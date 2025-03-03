import logging
import torch
import pandas as pd
from src.preprocessor.engagement_calculator import EngagementCalculator
from src.preprocessor.resource_logger import ResourceLogger  # Importar ResourceLogger

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, interacoes, noticias, user_profiles, model):
        self.interacoes = interacoes
        self.noticias = noticias
        self.user_profiles = user_profiles
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Inicializa o EngagementCalculator para calcular o engajamento do usuário
        self.engagement_calculator = EngagementCalculator()
        # Inicializa o ResourceLogger para monitoramento de recursos
        self.resource_logger = ResourceLogger()

    def predict(self, user_id: str, number_of_records=10) -> list[dict]:
        logger.info(f"Gerando predições para usuário {user_id}")
        user_emb = torch.tensor(self.user_profiles[user_id], dtype=torch.float32).to(self.device)

        # Carrega embeddings de notícias e datas
        news_embs = torch.tensor(self.noticias['embedding'].tolist(), dtype=torch.float32).to(self.device)
        issued_dates = pd.to_datetime(self.noticias['issued'],
                                      errors='coerce')  # Converte datas, tratando valores inválidos como NaT
        # Calcula a recência como um fator de peso (0 a 1)
        current_time = pd.Timestamp.now()
        recency_weights = []
        for date in issued_dates:
            if pd.isna(date):
                recency_weights.append(0.1)  # Penaliza notícias sem data
            else:
                time_diff_days = (current_time - date).days
                recency = np.exp(-time_diff_days / 30)  # Decay exponencial com meia-vida de 30 dias
                recency_weights.append(recency)
        recency_weights = torch.tensor(recency_weights, dtype=torch.float32).to(self.device)

        # Calcula o engajamento global (média de engajamento de todos os usuários para cada notícia)
        global_engagement = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)
        interaction_counts = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)
        page_to_idx = {page: idx for idx, page in enumerate(self.noticias['page'])}
        for _, row in self.interacoes.iterrows():
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
        global_engagement = torch.where(interaction_counts > 0, global_engagement / interaction_counts,
                                        torch.tensor(0.0, device=self.device))
        # Normalizar o engajamento global para o intervalo [0, 1]
        max_global_engagement = torch.max(global_engagement)
        if max_global_engagement > 0:
            global_engagement = global_engagement / max_global_engagement

        # Calcula o engajamento do usuário com cada notícia no dataset de notícias
        user_interactions = self.interacoes[self.interacoes['userId'] == user_id]
        specific_engagement = torch.zeros(len(self.noticias), dtype=torch.float32).to(self.device)
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
        # Se o usuário tem muitas interações, aumenta o peso do engajamento específico
        specific_weight = min(num_user_interactions / 5, 1.0)  # Aumenta até 1.0 com 5 interações
        global_weight = 1.0 - specific_weight
        engagement_weights = specific_weight * specific_engagement + global_weight * global_engagement

        # Calcula scores com o modelo
        with torch.no_grad():
            self.model.eval()
            scores = self.model(user_emb.expand_as(news_embs), news_embs).squeeze()
            # Ajusta os scores com o peso de recência e engajamento do usuário
            scores = scores + (scores * recency_weights) + (
                        scores * engagement_weights * 2.0)  # Aumenta o peso do engajamento
            # Log do uso da GPU após calcular os scores
            self.resource_logger.log_gpu_usage()

        # Obtém os top índices, evitando duplicatas e adicionando diversidade
        seen_pages = set()
        top_indices_unique = []
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        diversity_scores = torch.ones(len(scores), dtype=torch.float32).to(self.device)

        for idx in sorted_indices:
            page = self.noticias.iloc[idx]['page']
            if page not in seen_pages:
                seen_pages.add(page)
                top_indices_unique.append(idx)
                # Aplicar penalidade de diversidade com base na similaridade com notícias já selecionadas
                selected_embedding = news_embs[idx]
                for selected_idx in top_indices_unique[:-1]:
                    sim = torch.cosine_similarity(selected_embedding, news_embs[selected_idx], dim=0)
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

        return predictions
