import logging
import pandas as pd
from sklearn.linear_model import Ridge
import joblib

logger = logging.getLogger(__name__)

class Trainer:
    def train(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, user_profiles: dict, validacao_file: str) -> Ridge:
        logger.info(f"Carregando validação de {validacao_file}...")
        validacao = pd.read_csv(validacao_file)
        X, y = [], []
        logger.info("Preparando dados para treinamento...")
        for i, row in validacao.iterrows():
            if i % 1000 == 0:
                logger.info(f"Processados {i} registros de validação...")
            user_id = row['userId']
            news_id = row['history']
            if user_id in user_profiles and news_id in noticias['page'].values:
                user_emb = user_profiles[user_id]
                news_emb = noticias[noticias['page'] == news_id]['embedding'].values[0]
                X.append(np.concatenate([user_emb, news_emb]))
                y.append(row['relevance'])
        if X:
            logger.info("Treinando modelo Ridge...")
            regressor = Ridge(alpha=1.0)
            regressor.fit(X, y)
            joblib.dump(regressor, 'regressor.pkl')
            logger.info("Modelo treinado e salvo como 'regressor.pkl'")
            return regressor
        logger.warning("Dados insuficientes para treinamento.")
        return None
