# src/metrics_calculator.py
import logging
import torch
import pandas as pd
import numpy as np
import time
import os
from src.preprocessor.cache_manager import CacheManager
from src.preprocessor.engagement_calculator import EngagementCalculator
from src.preprocessor.resource_logger import ResourceLogger  # Importar ResourceLogger

logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self, state):
        self.state = state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_manager = CacheManager()
        # Adiciona o EngagementCalculator para calcular o engajamento de cada usuário
        self.engagement_calculator = EngagementCalculator()
        # Inicializa o ResourceLogger para monitoramento de recursos
        self.resource_logger = ResourceLogger()
        logger.info(f"MetricsCalculator inicializado no dispositivo: {self.device}")

    def calculate_metrics(self, k=10, force_recalc=False):
        cache_file = 'data/cache/metrics_cache.h5'
        start_time = time.time()
        logger.info(f"Iniciando cálculo de métricas de avaliação para top-{k} na GPU")

        # Verifica cache se não forçar recálculo
        if not force_recalc and os.path.exists(cache_file):
            try:
                cached_metrics = self.cache_manager.load_metrics(cache_file)
                logger.info(
                    f"Métricas carregadas do cache em {time.time() - start_time:.2f} segundos: {cached_metrics}")
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
        user_ids = np.random.choice(active_users, size=min(500, len(active_users)), replace=False)  # Reduzido para 500
        logger.info(f"Selecionados {len(user_ids)} usuários ativos de {len(active_users)} com >5 interações")
        logger.info(f"Carregados {len(interacoes)} registros de interações e {len(noticias)} notícias")

        logger.info(f"Processando métricas para {len(user_ids)} usuários")
        logger.info(f"IDs dos 10 usuários mais ativos: {user_ids[:10]}")

        precision_at_k_total = []
        recall_at_k_total = []
        mrr_total = []
        ils_total = []
        recommended_items = set()

        # Calcular pesos de recência
        issued_dates = pd.to_datetime(noticias['issued'], errors='coerce')
        current_time = pd.Timestamp.now()
        recency_weights = []
        for date in issued_dates:
            if pd.isna(date):
                recency_weights.append(0.1)  # Penaliza notícias sem data
            else:
                time_diff_days = (current_time - date).days
                recency = np.exp(-time_diff_days / 30)
                recency_weights.append(recency)
        recency_weights = torch.tensor(recency_weights, dtype=torch.float32).to(self.device)

        # Calcular o engajamento global (média de engajamento de todos os usuários para cada notícia)
        global_engagement = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
        interaction_counts = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
        page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        for _, row in interacoes.iterrows():
            hist = row['history'].split(', ')
            clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
            times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
            scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
            for h, c, t, s in zip(hist, clicks, times, scrolls):
                if h in page_to_idx:
                    idx = page_to_idx[h]
                    engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                    global_engagement[idx] += engagement
                    interaction_counts[idx] += 1
        # Evitar divisão por zero e calcular a média
        global_engagement = torch.where(interaction_counts > 0, global_engagement / interaction_counts, torch.tensor(0.0, device=self.device))
        # Normalizar o engajamento global para o intervalo [0, 1]
        max_global_engagement = torch.max(global_engagement)
        if max_global_engagement > 0:
            global_engagement = global_engagement / max_global_engagement

        # Colocar o REGRESSOR na GPU e em modo de avaliação
        if self.state.REGRESSOR is None:
            logger.error("REGRESSOR não está disponível")
            raise ValueError("REGRESSOR não está disponível")
        self.state.REGRESSOR.to(self.device)
        self.state.REGRESSOR.eval()

        batch_size_noticias = 1000  # Reduzido para 1000 para evitar problemas de memória na GPU
        total_notcias = len(noticias)

        for user_idx, user_id in enumerate(user_ids):
            user_start_time = time.time()
            if user_id not in self.state.USER_PROFILES:
                logger.warning(f"Usuário {user_id} não encontrado nos perfis; pulando")
                continue

            # Carrega embedding do usuário na GPU
            user_emb = torch.tensor(self.state.USER_PROFILES[user_id], dtype=torch.float32).to(self.device)
            logger.debug(f"Embedding do usuário {user_id}: shape={user_emb.shape}")

            # Calcula o engajamento do usuário com cada notícia no dataset de notícias
            user_interactions = interacoes[interacoes['userId'] == user_id]
            engagement_weights = global_engagement.clone()  # Começa com o engajamento global como fallback
            if not user_interactions.empty:
                hist = user_interactions['history'].iloc[0].split(', ')
                clicks = [float(x) for x in user_interactions['numberOfClicksHistory'].iloc[0].split(', ')]
                times = [float(x) for x in user_interactions['timeOnPageHistory'].iloc[0].split(', ')]
                scrolls = [float(x) for x in user_interactions['scrollPercentageHistory'].iloc[0].split(', ')]

                for h, c, t, s in zip(hist, clicks, times, scrolls):
                    if h in page_to_idx:
                        idx = page_to_idx[h]
                        engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                        engagement_weights[idx] = engagement  # Substitui o engajamento global pelo específico

            # Normalizar os engagement_weights para o intervalo [0, 1]
            max_engagement = torch.max(engagement_weights)
            if max_engagement > 0:
                engagement_weights = engagement_weights / max_engagement

            # Processar notícias em lotes
            scores_all = []

            for batch_start in range(0, total_notcias, batch_size_noticias):
                batch_end = min(batch_start + batch_size_noticias, total_notcias)
                batch_noticias = noticias.iloc[batch_start:batch_end]
                logger.debug(f"Processando lote de notícias {batch_start} a {batch_end} de {total_notcias}")

                # Carrega embeddings e recency weights do lote atual na GPU
                news_embs = torch.tensor(np.array(batch_noticias['embedding'].tolist()), dtype=torch.float32).to(
                    self.device)
                batch_recency_weights = recency_weights[batch_start:batch_end]
                batch_engagement_weights = engagement_weights[batch_start:batch_end]
                logger.debug(f"Embeddings de notícias do lote: shape={news_embs.shape}")

                # Expande o embedding do usuário para o lote atual
                user_emb_batch = user_emb.unsqueeze(0).expand(len(news_embs), -1)
                logger.debug(f"Embedding do usuário expandido: shape={user_emb_batch.shape}")

                # Calcula scores usando o REGRESSOR na GPU
                with torch.no_grad():
                    scores = self.state.REGRESSOR(user_emb_batch, news_embs)
                    scores = scores.squeeze(-1) + torch.rand(scores.shape, device=self.device) * 0.05
                    scores = scores + (scores * batch_recency_weights) + (scores * batch_engagement_weights)
                logger.debug(f"Scores do lote: shape={scores.shape}")
                scores_all.append(scores.cpu())  # Move para CPU para liberar memória na GPU

                # Log do uso da GPU a cada lote de notícias usando ResourceLogger
                if (batch_start // batch_size_noticias) % 10 == 0:  # Log a cada 10 lotes
                    self.resource_logger.log_gpu_usage()

                # Libera memória da GPU
                del news_embs, user_emb_batch, scores
                torch.cuda.empty_cache()

            # Converte scores_all para tensor e obtém os top-k índices
            scores_all = torch.cat(scores_all, dim=0)  # [total_notcias]
            logger.debug(f"Scores totais: shape={scores_all.shape}")

            # Obtém os top-k índices, evitando duplicatas
            seen_pages = set()
            top_indices_unique = []
            sorted_indices = torch.argsort(scores_all, descending=True).numpy()
            for idx in sorted_indices:
                page = noticias.iloc[idx]['page']
                if page not in seen_pages:
                    seen_pages.add(page)
                    top_indices_unique.append(idx)
                if len(top_indices_unique) >= k:
                    break

            top_indices = top_indices_unique[:k]
            top_pages = noticias['page'].iloc[top_indices].tolist()
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
            if len(top_pages) > 1:
                top_indices_tensor = torch.tensor([noticias[noticias['page'] == page].index[0] for page in top_pages],
                                                  dtype=torch.long)
                top_embeddings = torch.tensor(np.array(noticias['embedding'].iloc[top_indices_tensor].tolist()),
                                              dtype=torch.float32).to(self.device)
                if len(top_embeddings) > 1:
                    top_embeddings_norm = top_embeddings / torch.norm(top_embeddings, dim=1, keepdim=True)
                    sim_matrix = top_embeddings_norm @ top_embeddings_norm.T
                    ils = torch.triu(sim_matrix, diagonal=1).mean().item()
                    ils_total.append(ils)

                # Libera memória
                del top_embeddings, top_embeddings_norm, sim_matrix
                torch.cuda.empty_cache()

            elapsed_user = time.time() - user_start_time
            logger.debug(f"Usuário {user_id} processado em {elapsed_user:.2f} segundos: "
                         f"Precisão@{k}={precision_at_k:.4f}, Recall@{k}={recall_at_k:.4f}")

        # Cobertura
        catalog_coverage = len(recommended_items) / len(noticias)
        logger.info(f"Cobertura do catálogo: {len(recommended_items)}/{len(noticias)} notícias recomendadas")

        # Calcula médias
        metrics = {
            "precision_at_k": np.mean(precision_at_k_total) if precision_at_k_total else 0.0,
            "recall_at_k": np.mean(recall_at_k_total) if recall_at_k_total else 0.0,
            "mrr": np.mean(mrr_total) if mrr_total else 0.0,
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