import logging
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os
import time
from src.preprocessor.engagement_calculator import EngagementCalculator
from src.preprocessor.resource_logger import ResourceLogger
from src.preprocessor.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, interacoes, noticias, user_profiles, model):
        self.interacoes = interacoes
        self.noticias = noticias
        self.user_profiles = user_profiles
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.engagement_calculator = EngagementCalculator()
        self.resource_logger = ResourceLogger()
        self.cache_manager = CacheManager()
        logger.info(f"Predictor inicializado no dispositivo: {self.device}")

        # Pré-carrega embeddings de notícias na GPU
        self._preload_news_embeddings()
        # Pré-calcula e armazena recency_weights e global_engagement
        self._preload_recency_and_engagement()

    def _preload_news_embeddings(self):
        """Carrega embeddings de notícias na GPU uma única vez."""
        start_time = time.time()
        if self.noticias.empty or 'embedding' not in self.noticias.columns:
            logger.error("Dados de notícias inválidos ou embeddings ausentes")
            raise ValueError("Dados de notícias inválidos ou embeddings ausentes")
        logger.info(f"Carregando {len(self.noticias)} embeddings de notícias na GPU")
        self.news_embs = torch.tensor(self.noticias['embedding'].tolist(), dtype=torch.float32).to(self.device)
        logger.info(
            f"Embeddings de notícias pré-carregados: shape={self.news_embs.shape}, tempo={time.time() - start_time:.2f} segundos")

    def _preload_recency_and_engagement(self):
        """Pré-carrega recency_weights e global_engagement, ajustando recency com o tempo atual."""
        # Carrega a base de recência e ajusta com o tempo atual
        recency_base_file = os.path.join(self.cache_manager.cache_dir, 'recency_base.h5')
        if not os.path.exists(recency_base_file):
            logger.warning(f"Arquivo de base de recência não encontrado: {recency_base_file}; calculando do zero")
            logger.info("Calculando pesos de recência para notícias")
            start_time = time.time()
            issued_dates = pd.to_datetime(self.noticias['issued'], errors='coerce').dt.tz_localize(None)
            current_time = pd.Timestamp.now(tz=None).tz_localize(None)
            time_diff_days = torch.tensor(
                [(current_time - date).days if pd.notna(date) else float('nan') for date in issued_dates],
                dtype=torch.float32
            ).to(self.device)
            self.recency_weights = torch.where(
                torch.isnan(time_diff_days),
                torch.tensor(0.1, device=self.device),
                torch.exp(-time_diff_days / 30)
            )
            self.cache_manager.save_array(recency_base_file, time_diff_days.cpu().numpy())
            logger.info(
                f"Pesos de recência calculados e base salva em {recency_base_file}: shape={self.recency_weights.shape}, tempo={time.time() - start_time:.2f} segundos")
        else:
            logger.info(f"Carregando base de recência do cache: {recency_base_file}")
            time_diff_days = torch.tensor(self.cache_manager.load_array(recency_base_file), dtype=torch.float32).to(
                self.device)
            self.recency_weights = torch.where(
                torch.isnan(time_diff_days),
                torch.tensor(0.1, device=self.device),
                torch.exp(-time_diff_days / 30)
            )
            logger.debug(
                f"Pesos de recência ajustados: shape={self.recency_weights.shape}, min={self.recency_weights.min().item():.4f}, max={self.recency_weights.max().item():.4f}")

        # Cache para global_engagement
        engagement_cache_file = os.path.join(self.cache_manager.cache_dir, 'global_engagement.h5')
        self.page_to_idx = {page: idx for idx, page in enumerate(self.noticias['page'])}
        logger.debug(f"Mapeamento page_to_idx criado com {len(self.page_to_idx)} entradas")
        if not os.path.exists(engagement_cache_file):
            logger.warning(f"Arquivo de engajamento global não encontrado: {engagement_cache_file}; calculando do zero")
            logger.info("Calculando engajamento global para notícias")
            start_time = time.time()
            self.global_engagement = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)
            interaction_counts = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)

            for _, row in tqdm(self.interacoes.iterrows(), total=len(self.interacoes),
                               desc="Pré-processando interações", leave=False):
                hist = row['history'].split(', ')
                clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
                times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
                scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
                for h, c, t, s in zip(hist, clicks, times, scrolls):
                    if h in self.page_to_idx:
                        idx = self.page_to_idx[h]
                        engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                        self.global_engagement[idx] += engagement
                        interaction_counts[idx] += 1
            self.global_engagement = torch.where(interaction_counts > 0, self.global_engagement / interaction_counts,
                                                 torch.tensor(0.0, device=self.device))
            max_global_engagement = torch.max(self.global_engagement)
            logger.debug(f"Máximo de engajamento global: {max_global_engagement}")
            if max_global_engagement > 0:
                self.global_engagement = self.global_engagement / max_global_engagement
            self.cache_manager.save_array(engagement_cache_file, self.global_engagement.cpu().numpy())
            logger.debug(
                f"Engajamento global calculado: shape={self.global_engagement.shape}, min={self.global_engagement.min().item():.4f}, max={self.global_engagement.max().item():.4f}")
            logger.info(
                f"Engajamento global calculado e salvo em {engagement_cache_file}: shape={self.global_engagement.shape}, tempo={time.time() - start_time:.2f} segundos")
        else:
            logger.info(f"Carregando global_engagement do cache: {engagement_cache_file}")
            self.global_engagement = torch.tensor(self.cache_manager.load_array(engagement_cache_file),
                                                  dtype=torch.float32).to(self.device)
            logger.debug(
                f"Engajamento global carregado: shape={self.global_engagement.shape}, min={self.global_engagement.min().item():.4f}, max={self.global_engagement.max().item():.4f}")

    def predict(self, user_id: str, number_of_records=10) -> list[dict]:
        """Gera predições personalizadas para um usuário específico."""
        start_time = time.time()
        logger.info(f"Iniciando predição para usuário {user_id} com {number_of_records} registros solicitados")

        # Verifica se o usuário existe nos perfis
        if user_id not in self.user_profiles:
            logger.warning(f"Usuário {user_id} não encontrado nos perfis de usuário")
            raise ValueError(f"Usuário {user_id} não encontrado nos perfis")

        # Carrega embedding do usuário na GPU
        logger.debug(f"Carregando embedding do usuário {user_id}")
        user_emb = torch.tensor(self.user_profiles[user_id], dtype=torch.float32).to(self.device)
        logger.debug(f"Embedding do usuário carregado: shape={user_emb.shape}")

        # Calcula engajamento específico do usuário
        logger.info(f"Calculando engajamento específico para usuário {user_id}")
        specific_engagement = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)
        num_user_interactions = 0
        user_interactions = self.interacoes[self.interacoes['userId'] == user_id]
        if not user_interactions.empty:
            hist = user_interactions['history'].iloc[0].split(', ')
            clicks = [float(x) for x in user_interactions['numberOfClicksHistory'].iloc[0].split(', ')]
            times = [float(x) for x in user_interactions['timeOnPageHistory'].iloc[0].split(', ')]
            scrolls = [float(x) for x in user_interactions['scrollPercentageHistory'].iloc[0].split(', ')]
            num_user_interactions = len(hist)
            for h, c, t, s in zip(hist, clicks, times, scrolls):
                if h in self.page_to_idx:
                    idx = self.page_to_idx[h]
                    engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                    specific_engagement[idx] = engagement
            # Normalizar o engajamento específico para o intervalo [0, 1]
            max_specific_engagement = torch.max(specific_engagement)
            if max_specific_engagement > 0:
                specific_engagement = specific_engagement / max_specific_engagement
        logger.debug(
            f"Engajamento específico calculado: interações={num_user_interactions}, shape={specific_engagement.shape}")

        # Combinar engajamento específico e global com pesos dinâmicos
        # Se o usuário tem muitas interações, aumenta o peso do engajamento específico
        specific_weight = min(num_user_interactions / 5, 1.0)  # Aumenta até 1.0 com 5 interações
        global_weight = 1.0 - specific_weight
        engagement_weights = specific_weight * specific_engagement + global_weight * self.global_engagement
        logger.debug(
            f"Pesos de engajamento combinados: specific_weight={specific_weight:.2f}, global_weight={global_weight:.2f}")

        # Processa notícias em lotes para otimizar desempenho
        batch_size = 4096  # Aumentado para melhor uso da GPU
        total_noticias = len(self.noticias)
        scores_all = []

        for batch_start in tqdm(range(0, total_noticias, batch_size), desc=f"Processando notícias para {user_id[:10]}",
                                leave=False):
            batch_end = min(batch_start + batch_size, total_noticias)
            batch_news_embs = self.news_embs[batch_start:batch_end]
            batch_recency_weights = self.recency_weights[batch_start:batch_end]
            batch_engagement_weights = engagement_weights[batch_start:batch_end]
            logger.debug(f"Processando lote {batch_start}-{batch_end} de {total_noticias} notícias")

            # Calcula scores com o modelo
            with torch.no_grad():
                self.model.eval()
                batch_scores = self.model(user_emb.expand_as(batch_news_embs), batch_news_embs).squeeze()
                # Ajusta os scores com o peso de recência e engajamento do usuário
                batch_scores = batch_scores + (batch_scores * batch_recency_weights) + (
                            batch_scores * batch_engagement_weights * 2.0)  # Aumenta o peso do engajamento
            scores_all.append(batch_scores.cpu())
            # Libera memória da GPU
            del batch_news_embs, batch_scores
            torch.cuda.empty_cache()

        # Converte scores_all para tensor
        scores_all = torch.cat(scores_all, dim=0)
        logger.debug(f"Scores finais calculados: shape={scores_all.shape}")

        # Obtém os top índices, evitando duplicatas e adicionando diversidade
        seen_pages = set()
        top_indices_unique = []
        sorted_indices = torch.argsort(scores_all, descending=True).numpy()
        diversity_scores = torch.ones(len(scores_all), dtype=torch.float32).to(self.device)

        for idx in sorted_indices:
            page = self.noticias.iloc[idx]['page']
            if page not in seen_pages:
                seen_pages.add(page)
                top_indices_unique.append(idx)
                # Aplicar penalidade de diversidade com base na similaridade com notícias já selecionadas
                selected_embedding = self.news_embs[idx]
                for selected_idx in top_indices_unique[:-1]:
                    sim = torch.cosine_similarity(selected_embedding, self.news_embs[selected_idx], dim=0)
                    diversity_scores[idx] *= (1 - sim * 0.5)  # Penaliza se for muito semelhante
            if len(top_indices_unique) >= number_of_records:
                break

        top_indices = top_indices_unique[:number_of_records]
        top_news = self.noticias.iloc[top_indices]
        predictions = [
            {
                "page": row['page'],
                "title": row['title'],
                "link": row['url'],
                "issued": str(row['issued']) if pd.notna(row['issued']) else "Sem data",
            }
            for _, row in top_news.iterrows()
        ]

        elapsed = time.time() - start_time
        logger.info(
            f"Predições concluídas para {user_id} em {elapsed:.2f} segundos: {len(predictions)} recomendações geradas")
        # Log do uso da GPU após calcular os scores
        self.resource_logger.log_gpu_usage()
        return predictions


# Métodos auxiliares no CacheManager para lidar com arrays
def save_array(self, filepath, array):
    """Salva um array NumPy em um arquivo HDF5."""
    import h5py
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=array)
    logger.info(f"Array salvo em {filepath}")


def load_array(self, filepath):
    """Carrega um array NumPy de um arquivo HDF5."""
    import h5py
    with h5py.File(filepath, 'r') as f:
        array = f['data'][:]
    logger.info(f"Array carregado de {filepath}")
    return array


# Adicionar ao CacheManager (você precisará incluir isso na classe CacheManager)
CacheManager.save_array = save_array
CacheManager.load_array = load_array
