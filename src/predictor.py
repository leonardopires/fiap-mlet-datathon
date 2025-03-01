import logging
import torch
import pandas as pd

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, interacoes, noticias, user_profiles, model):
        self.interacoes = interacoes
        self.noticias = noticias
        self.user_profiles = user_profiles
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, user_id: str) -> list[dict]:
        logger.info(f"Gerando predições para usuário {user_id}")
        user_emb = torch.tensor(self.user_profiles[user_id], dtype=torch.float32).to(self.device)

        news_embs = torch.tensor(self.noticias['embedding'].tolist(), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.model.eval()
            scores = self.model(user_emb.expand_as(news_embs), news_embs).squeeze()

        top_indices = torch.topk(scores, 10).indices.cpu().numpy()
        top_news = self.noticias.iloc[top_indices]
        predictions = [
            {
                "page": row['page'],
                "title": row['title'],
                "link": row['url'],
            }
            for _, row in top_news.iterrows()
        ]

        return predictions
