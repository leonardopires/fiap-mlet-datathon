import logging
import torch
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
from src.preprocessor.cache_manager import CacheManager
from src.preprocessor.engagement_calculator import EngagementCalculator
from src.preprocessor.resource_logger import ResourceLogger

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
        logger.info("Calculando pesos de recência e engajamento global")
        # Carrega a base de recência e ajusta com o tempo atual
        recency_base_file = 'data/cache/recency_base.h5'
        if not os.path.exists(recency_base_file):
            logger.warning(f"Arquivo de base de recência não encontrado: {recency_base_file}; calculando do zero")
            issued_dates = pd.to_datetime(noticias['issued'], errors='coerce')
            current_time = pd.Timestamp.now(tz=None).tz_localize(None)  # Remove fuso horário de current_time
            # Vetorizar o cálculo de recency_weights na GPU
            time_diff_days = torch.zeros(len(issued_dates), dtype=torch.float32)
            for i, date in enumerate(issued_dates):
                if pd.isna(date):
                    time_diff_days[i] = float('nan')
                else:
                    date = date.tz_localize(None)
                    time_diff_days[i] = (current_time - date).days
            self.cache_manager.save_array(recency_base_file, time_diff_days.cpu().numpy())
            logger.info(f"Base de recência salva em {recency_base_file}")
        else:
            logger.info(f"Carregando base de recência do cache: {recency_base_file}")
            time_diff_days = torch.tensor(self.cache_manager.load_array(recency_base_file), dtype=torch.float32)

        # Mover para GPU e calcular recency_weights
        time_diff_days = time_diff_days.to(self.device)
        recency_weights = torch.where(
            torch.isnan(time_diff_days),
            torch.tensor(0.1, device=self.device),
            torch.exp(-time_diff_days / 30)
        )
        logger.info(f"Calculados pesos de recência para {len(recency_weights)} notícias")
        logger.info(f"Recency weights: shape={recency_weights.shape}")

        # Carrega global_engagement do cache
        engagement_cache_file = 'data/cache/global_engagement.h5'
        page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        logger.info(f"Page to index: {len(page_to_idx)} páginas mapeadas")
        if not os.path.exists(engagement_cache_file):
            logger.warning(f"Arquivo de engajamento global não encontrado: {engagement_cache_file}; calculando do zero")
            logger.info("Calculando engajamento global")
            global_engagement = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
            logger.info(f"Global engagement: shape={global_engagement.shape}")
            interaction_counts = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
            logger.info(f"Interaction counts: shape={interaction_counts.shape}")

            logger.info("Calculando engajamento global")
            # Pré-processar dados para acumulação na GPU
            indices = []
            engagements = []
            for _, row in interacoes.iterrows():
                hist = row['history'].split(', ')
                clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
                times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
                scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
                for h, c, t, s in zip(hist, clicks, times, scrolls):
                    if h in page_to_idx:
                        idx = page_to_idx[h]
                        engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                        indices.append(idx)
                        engagements.append(engagement)
            # Acumular na GPU usando scatter_add_
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)
            engagements = torch.tensor(engagements, dtype=torch.float32).to(self.device)
            global_engagement.scatter_add_(0, indices, engagements)
            interaction_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
            logger.info("Engajamento global calculado")
            # Evitar divisão por zero e calcular a média
            logger.info("Normalizando engajamento global")
            global_engagement = torch.where(interaction_counts > 0, global_engagement / interaction_counts,
                                            torch.tensor(0.0, device=self.device))
            logger.info(f"Global engagement normalizado: shape={global_engagement.shape}")
            # Normalizar o engajamento global para o intervalo [0, 1]
            max_global_engagement = torch.max(global_engagement)
            logger.info(f"Máximo de engajamento global: {max_global_engagement}")
            if max_global_engagement > 0:
                global_engagement = global_engagement / max_global_engagement
            logger.info("Engajamento global normalizado")
            self.cache_manager.save_array(engagement_cache_file, global_engagement.cpu().numpy())
            logger.info(f"Engajamento global salvo em {engagement_cache_file}")
        else:
            logger.info(f"Carregando global_engagement do cache: {engagement_cache_file}")
            global_engagement = torch.tensor(self.cache_manager.load_array(engagement_cache_file),
                                             dtype=torch.float32).to(self.device)
            logger.info(f"Engajamento global carregado: shape={global_engagement.shape}")

        # Colocar o REGRESSOR na GPU e em modo de avaliação
        logger.info("Calculando métricas de avaliação para cada usuário")
        if self.state.REGRESSOR is None:
            logger.error("REGRESSOR não está disponível")
            raise ValueError("REGRESSOR não está disponível")
        self.state.REGRESSOR.to(self.device)
        self.state.REGRESSOR.eval()
        logger.info("REGRESSOR movido para a GPU e em modo de avaliação")

        batch_size_noticias = 1000  # Reduzido para 1000 para evitar problemas de memória na GPU
        total_notcias = len(noticias)

        for user_idx, user_id in enumerate(tqdm(user_ids, desc="Processando usuários", unit="user")):
            user_start_time = time.time()
            if user_id not in self.state.USER_PROFILES:
                logger.warning(f"Usuário {user_id} não encontrado nos perfis; pulando")
                continue

            # Carrega embedding do usuário na GPU
            user_emb = torch.tensor(self.state.USER_PROFILES[user_id], dtype=torch.float32).to(self.device)
            logger.debug(f"Embedding do usuário {user_id}: shape={user_emb.shape}")

            # Calcula o engajamento do usuário com cada notícia no dataset de notícias
            user_interactions = interacoes[interacoes['userId'] == user_id]
            specific_engagement = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
            num_user_interactions = 0
            if not user_interactions.empty:
                hist = user_interactions['history'].iloc[0].split(', ')
                clicks = [float(x) for x in user_interactions['numberOfClicksHistory'].iloc[0].split(', ')]
                times = [float(x) for x in user_interactions['timeOnPageHistory'].iloc[0].split(', ')]
                scrolls = [float(x) for x in user_interactions['scrollPercentageHistory'].iloc[0].split(', ')]
                num_user_interactions = len(hist)

                for h, c, t, s in zip(hist, clicks, times, scrolls):
                    if h in page_to_idx:
                        idx = page_to_idx[h]
                        engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                        specific_engagement[idx] = engagement

            # Normalizar o engajamento específico para o intervalo [0, 1]
            max_specific_engagement = torch.max(specific_engagement)
            if max_specific_engagement > 0:
                specific_engagement = specific_engagement / max_specific_engagement

            # Combinar engajamento específico e global com pesos dinâmicos
            specific_weight = min(num_user_interactions / 5, 1.0)  # Aumenta até 1.0 com 5 interações
            global_weight = 1.0 - specific_weight
            engagement_weights = specific_weight * specific_engagement + global_weight * global_engagement

            # Processar notícias em lotes
            scores_all = []

            for batch_start in tqdm(range(0, total_notcias, batch_size_noticias),
                                    desc=f"Notícias do usuário {user_id[:10]} ({user_idx}/{len(user_ids)})",
                                    leave=False, unit="batch"):
                batch_end = min(batch_start + batch_size_noticias, total_notcias)
                batch_noticias = noticias.iloc[batch_start:batch_end]
                batch_size_actual = batch_end - batch_start  # Calcula o tamanho real do lote
                logger.debug(f"Processando lote de notícias {batch_start} a {batch_end} de {total_notcias}")

                # Carrega embeddings e recency weights do lote atual na GPU
                news_embs = torch.tensor(np.array(batch_noticias['embedding'].tolist()), dtype=torch.float32).to(
                    self.device)
                batch_recency_weights = recency_weights[batch_start:batch_end]
                batch_engagement_weights = engagement_weights[batch_start:batch_end]
                logger.debug(f"Embeddings de notícias do lote: shape={news_embs.shape}")
                logger.debug(f"Batch recency weights: shape={batch_recency_weights.shape}")
                logger.debug(f"Batch engagement weights: shape={batch_engagement_weights.shape}")

                # Expande o embedding do usuário para o lote atual
                user_emb_batch = user_emb.unsqueeze(0).expand(batch_size_actual, -1)
                logger.debug(f"Embedding do usuário expandido: shape={user_emb_batch.shape}")

                # Calcula scores usando o REGRESSOR na GPU
                with torch.no_grad():
                    scores = self.state.REGRESSOR(user_emb_batch, news_embs)
                    if scores.dim() == 2 and scores.shape[0] == scores.shape[1]:
                        # Caso retorne uma matriz de similaridade [batch_size, batch_size]
                        scores = torch.diagonal(scores)  # Extrai a diagonal
                    else:
                        scores = scores.squeeze(-1)  # Caso correto [batch_size, 1] -> [batch_size]
                    scores = scores + torch.rand(scores.shape, device=self.device) * 0.05
                    scores = scores + (scores * batch_recency_weights) + (scores * batch_engagement_weights * 2.0)
                logger.debug(f"Scores do lote: shape={scores.shape}, size={batch_size_actual}")
                # Verificar se scores é 1D
                if scores.dim() != 1:
                    logger.error(f"Scores tensor is not 1D: shape={scores.shape}")
                    raise ValueError(f"Scores tensor is not 1D: shape={scores.shape}")
                scores_all.append(scores.cpu())  # Move para CPU para liberar memória na GPU

                # Libera memória da GPU
                del news_embs, user_emb_batch, scores
                torch.cuda.empty_cache()

            # Converte scores_all para tensor e obtém os top-k índices com diversidade
            # Log dos shapes antes da concatenação
            logger.debug(f"Shapes de scores_all antes da concatenação: {[s.shape for s in scores_all]}")
            scores_all = torch.cat(scores_all, dim=0)  # [total_notcias]
            logger.debug(f"Scores totais: shape={scores_all.shape}")

            seen_pages = set()
            top_indices_unique = []
            sorted_indices = torch.argsort(scores_all, descending=True).numpy()
            diversity_scores = torch.ones(len(scores_all), dtype=torch.float32).to(self.device)

            for idx in sorted_indices:
                page = noticias.iloc[idx]['page']
                if page not in seen_pages:
                    seen_pages.add(page)
                    top_indices_unique.append(idx)
                    # Aplicar penalidade de diversidade com base na similaridade com notícias já selecionadas
                    selected_embedding = torch.tensor(noticias['embedding'].iloc[idx], dtype=torch.float32).to(
                        self.device)
                    for selected_idx in top_indices_unique[:-1]:
                        other_embedding = torch.tensor(noticias['embedding'].iloc[selected_idx],
                                                       dtype=torch.float32).to(self.device)
                        sim = torch.cosine_similarity(selected_embedding, other_embedding, dim=0)
                        diversity_scores[idx] *= (1 - sim * 0.5)  # Penaliza se for muito semelhante
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
            logger.debug(
                f"Usuário {user_id} processado em {elapsed_user:.2f} segundos: Precisão@{k}={precision_at_k:.4f}, Recall@{k}={recall_at_k:.4f}")

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
