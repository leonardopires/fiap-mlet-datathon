import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, user_profiles: dict, regressor):
        self.interacoes = interacoes
        self.noticias = noticias
        self.user_profiles = user_profiles
        self.regressor = regressor

    def predict(self, user_id: str, k: int = 10) -> list:
        logger.info(f"Gerando predições para user_id: {user_id}")
        if user_id not in self.user_profiles:
            logger.info("Usuário não encontrado, usando recomendações populares...")
            popular = self.interacoes['history'].str.split(', ').explode().value_counts().head(k).index
            return popular.tolist()
        user_emb = self.user_profiles[user_id]
        candidates = self.noticias[~self.noticias['page'].isin(self.interacoes[self.interacoes['userId'] == user_id]['history'].str.split(', ').iloc[0])]
        logger.info(f"Selecionados {len(candidates)} candidatos para predição...")
        X = [np.concatenate([user_emb, emb]) for emb in candidates['embedding']]
        scores = self.regressor.predict(X)
        top_indices = np.argsort(scores)[-k:][::-1]
        predictions = candidates.iloc[top_indices]['page'].tolist()
        logger.info(f"Predições geradas: {predictions}")
        return predictions

