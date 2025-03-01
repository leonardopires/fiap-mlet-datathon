# src/api/metrics_calculator.py
import logging
import torch
import pandas as pd
import numpy as np
import time
import os
from src.preprocessor.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class MetricsCalculator:
    def __init__(self, state):
        self.state = state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_manager = CacheManager()
        logger.info(f"MetricsCalculator inicializado no dispositivo: {self.device}")

    def calculate_metrics(self, k=10, force_recalc=False):
        cache_file = 'data/cache/metrics_cache.h5'
        start_time = time.time()
        logger.info(f"Iniciando cálculo de métricas de avaliação para top-{k} na GPU")

        # Verifica cache se não forçar recálculo
        if not force_recalc and os.path.exists(cache_file):
            try:
                cached_metrics = self.cache_manager.load_metrics(cache_file)
                logger.info(f"Métricas carregadas do cache em {time.time() - start_time:.2f} segundos: {cached_metrics}")
                return cached_metrics
            except Exception as e:
                logger.warning(f"Falha ao carregar métricas do cache: {e}; recalculando")

        # Carrega interações e notícias
        interacoes = pd.read_hdf('data/cache/interacoes.h5', 'interacoes')
        noticias = self.state.NOTICIAS
        if noticias is None or len(noticias) == 0:
            logger.error("Nenhum dado de notícias encontrado no estado")
            raise ValueError("Estado de notícias não inicializado ou vazio")
        if len(noticias) < k:
            logger.warning(f"Número de notícias ({len(noticias)}) menor que k={k}; ajustando k para {len(noticias)}")

        user_interactions = interacoes.groupby('userId')['history'].apply(lambda x: len(x.iloc[0].split(', ')))
        active_users = user_interactions[user_interactions > 5].index  # Usuários com >5 interações
        user_ids = np.random.choice(active_users, size=min(5000, len(active_users)), replace=False)
        logger.info(f"Selecionados {len(user_ids)} usuários ativos de {len(active_users)} com >5 interações")
        logger.info(f"Carregados {len(interacoes)} registros de interações e {len(noticias)} notícias")

        logger.info(f"Processando métricas para {len(user_ids)} usuários")
        logger.info(f"IDs dos 10 usuários mais ativos: {user_ids[:10]}")

        precision_at_k_total = []
        recall_at_k_total = []
        mrr_total = []
        ils_total = []
        recommended_items = set()

        # Pré-carrega embeddings de notícias na GPU
        news_embs = torch.tensor(np.array(noticias['embedding'].tolist()), dtype=torch.float32).to(self.device)
        logger.debug(f"Embeddings de notícias ({news_embs.shape}) carregados na GPU")

        for idx, user_id in enumerate(user_ids):
            user_start_time = time.time()
            if user_id not in self.state.USER_PROFILES:
                logger.warning(f"Usuário {user_id} não encontrado nos perfis; pulando")
                continue

            # Carrega embedding do usuário na GPU
            user_emb = torch.tensor(self.state.USER_PROFILES[user_id], dtype=torch.float32).to(self.device)
            logger.debug(f"Embedding do usuário {user_id}: shape={user_emb.shape}")
            user_emb = user_emb.unsqueeze(0).expand(len(news_embs), -1)  # Expande explicitamente
            logger.debug(f"Embedding expandido: shape={user_emb.shape}")

            # Calcula scores na GPU
            with torch.no_grad():
                scores = self.state.REGRESSOR(user_emb, news_embs)
                scores = scores.squeeze(-1) + torch.rand(scores.shape, device=self.device) * 0.1  # Adicionar ruído
            logger.debug(f"Scores brutos com ruído: shape={scores.shape}")
            if scores.numel() == 0:
                logger.error(f"Scores vazio para usuário {user_id}")
                continue
            scores = scores.squeeze(-1)  # Remove dimensão extra [255603, 1] -> [255603]
            logger.debug(f"Scores após squeeze: shape={scores.shape}, len={len(scores)}")
            if len(scores) < k:
                logger.warning(
                    f"Usuário {user_id}: apenas {len(scores)} scores disponíveis, ajustando k para {len(scores)}")
            effective_k = min(k, len(scores))  # Ajusta k dinamicamente
            if effective_k == 0:
                logger.error(f"Não há scores válidos para usuário {user_id}; pulando")
                continue
            top_indices = torch.topk(scores, effective_k).indices  # Top-K na GPU
            top_pages = noticias['page'].iloc[top_indices.cpu().numpy()].tolist()
            recommended_items.update(top_pages)

            # Ground truth (histórico do usuário)
            user_history = interacoes[interacoes['userId'] == user_id]['history'].iloc[0].split(', ')
            relevant_items = set(user_history)
            logger.debug(f"Usuário {user_id}: {len(relevant_items)} itens relevantes no histórico")

            # Precisão@k e Recall@k
            top_k_set = set(top_pages)
            hits = len(top_k_set & relevant_items)
            precision_at_k = hits / k
            recall_at_k = hits / len(relevant_items) if relevant_items else 0
            precision_at_k_total.append(precision_at_k)
            recall_at_k_total.append(recall_at_k)

            # MRR
            for i, page in enumerate(reversed(top_pages)):
                if page in relevant_items:
                    mrr_total.append(1 / (i + 1))
                    break
            else:
                mrr_total.append(0)

            # ILS (similaridade intra-lista) na GPU
            top_embeddings = news_embs[top_indices]
            if len(top_embeddings) > 1:
                # Normaliza embeddings
                top_embeddings_norm = top_embeddings / torch.norm(top_embeddings, dim=1, keepdim=True)
                sim_matrix = top_embeddings_norm @ top_embeddings_norm.T  # Similaridade de cosseno
                ils = torch.triu(sim_matrix, diagonal=1).mean().item()  # Média do triângulo superior
                ils_total.append(ils)

            elapsed_user = time.time() - user_start_time
            logger.debug(f"Usuário {user_id} processado em {elapsed_user:.2f} segundos: "
                         f"Precisão@{k}={precision_at_k:.4f}, Recall@{k}={recall_at_k:.4f}")

        # Cobertura
        catalog_coverage = len(recommended_items) / len(noticias)
        logger.info(f"Cobertura do catálogo: {len(recommended_items)}/{len(noticias)} notícias recomendadas")

        # Calcula médias
        metrics = {
            "precision_at_k": np.mean(precision_at_k_total),
            "recall_at_k": np.mean(recall_at_k_total),
            "mrr": np.mean(mrr_total),
            "intra_list_similarity": np.mean(ils_total) if ils_total else 0,
            "catalog_coverage": catalog_coverage
        }

        elapsed_total = time.time() - start_time
        logger.info(f"Métricas calculadas em {elapsed_total:.2f} segundos: "
                    f"Precisão@{k}={metrics['precision_at_k']:.4f}, "
                    f"Recall@{k}={metrics['recall_at_k']:.4f}, "
                    f"MRR={metrics['mrr']:.4f}, "
                    f"ILS={metrics['intra_list_similarity']:.4f}, "
                    f"Cobertura={metrics['catalog_coverage']:.4f}")
        # Salva em cache
        self.cache_manager.save_metrics(cache_file, metrics)
        logger.info(f"Métricas salvas em {cache_file}")
        return metrics