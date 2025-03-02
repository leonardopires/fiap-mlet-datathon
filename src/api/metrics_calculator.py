# src/metrics_calculator.py
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
        self.device = torch.device("cpu")  # Forçar CPU para evitar erros de memória
        self.cache_manager = CacheManager()
        logger.info(f"MetricsCalculator inicializado no dispositivo: {self.device}")

    def calculate_metrics(self, k=10, force_recalc=False):
        cache_file = 'data/cache/metrics_cache.h5'
        start_time = time.time()
        logger.info(f"Iniciando cálculo de métricas de avaliação para top-{k} na CPU")

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

        # Configurar batching para embeddings de notícias
        batch_size_noticias = 5000  # Reduzido para 5000 notícias por vez
        total_notcias = len(noticias)

        for user_idx, user_id in enumerate(user_ids):
            user_start_time = time.time()
            if user_id not in self.state.USER_PROFILES:
                logger.warning(f"Usuário {user_id} não encontrado nos perfis; pulando")
                continue

            # Carrega embedding do usuário na CPU
            user_emb = torch.tensor(self.state.USER_PROFILES[user_id], dtype=torch.float32)
            logger.info(f"Embedding do usuário {user_id}: shape={user_emb.shape}")

            # Processar notícias em lotes
            scores_all = []

            for batch_start in range(0, total_notcias, batch_size_noticias):
                batch_end = min(batch_start + batch_size_noticias, total_notcias)
                batch_notcias = noticias.iloc[batch_start:batch_end]
                logger.info(f"Processando lote de notícias {batch_start} a {batch_end} de {total_notcias}")

                # Carrega embeddings do lote atual na CPU
                news_embs = torch.tensor(np.array(batch_notcias['embedding'].tolist()), dtype=torch.float32)
                logger.info(f"Embeddings de notícias do lote: shape={news_embs.shape}")

                # Expande o embedding do usuário para o lote atual
                user_emb_batch = user_emb.unsqueeze(0).expand(len(news_embs), -1)
                logger.info(f"Embedding do usuário expandido: shape={user_emb_batch.shape}")

                # Calcula scores na CPU usando similaridade de cosseno
                with torch.no_grad():
                    # Normaliza os embeddings
                    user_emb_norm = user_emb_batch / torch.norm(user_emb_batch, dim=1, keepdim=True)
                    news_embs_norm = news_embs / torch.norm(news_embs, dim=1, keepdim=True)
                    # Calcula similaridade de cosseno
                    scores = torch.sum(user_emb_norm * news_embs_norm, dim=1)
                    # Adiciona ruído
                    scores = scores + torch.rand(scores.shape) * 0.1
                logger.info(f"Scores do lote: shape={scores.shape}")
                scores_all.append(scores)

                # Libera memória
                del news_embs, user_emb_batch, user_emb_norm, news_embs_norm, scores

            # Converte scores_all para tensor e obtém os top-k índices
            scores_all = torch.cat(scores_all, dim=0)  # [total_notcias]
            logger.info(f"Scores totais: shape={scores_all.shape}")

            # Obtém os top-k índices
            effective_k = min(k, len(scores_all))
            if effective_k == 0:
                logger.error(f"Não há scores válidos para usuário {user_id}; pulando")
                continue
            top_indices = torch.topk(scores_all, effective_k).indices
            top_pages = noticias['page'].iloc[top_indices.numpy()].tolist()
            recommended_items.update(top_pages)

            # Ground truth (histórico do usuário)
            user_history = interacoes[interacoes['userId'] == user_id]['history'].iloc[0].split(', ')
            relevant_items = set(user_history)
            logger.info(f"Usuário {user_id}: {len(relevant_items)} itens relevantes no histórico")

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

            # ILS (similaridade intra-lista) na CPU
            if len(top_pages) > 1:
                top_indices_tensor = torch.tensor([noticias[noticias['page'] == page].index[0] for page in top_pages], dtype=torch.long)
                top_embeddings = torch.tensor(np.array(noticias['embedding'].iloc[top_indices_tensor].tolist()), dtype=torch.float32)
                if len(top_embeddings) > 1:
                    top_embeddings_norm = top_embeddings / torch.norm(top_embeddings, dim=1, keepdim=True)
                    sim_matrix = top_embeddings_norm @ top_embeddings_norm.T
                    ils = torch.triu(sim_matrix, diagonal=1).mean().item()
                    ils_total.append(ils)

                # Libera memória
                del top_embeddings, top_embeddings_norm, sim_matrix

            elapsed_user = time.time() - user_start_time
            logger.info(f"Usuário {user_id} processado em {elapsed_user:.2f} segundos: "
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