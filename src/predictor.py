import logging
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, user_profiles: dict, regressor):
        """
        Inicializa o preditor com dados e modelo treinado.

        Args:
            interacoes (pd.DataFrame): Dados de interações dos usuários.
            noticias (pd.DataFrame): Dados das notícias com embeddings.
            user_profiles (dict): Perfis de usuário pré-processados (user_id -> embedding).
            regressor: Modelo de regressão treinado (ex.: Ridge).
        """
        init_start = time.time()
        logger.info("Iniciando inicialização do Predictor")
        self.interacoes = interacoes
        self.noticias = noticias
        self.user_profiles = user_profiles
        self.regressor = regressor
        elapsed = time.time() - init_start
        logger.info(
            f"Predictor inicializado em {elapsed:.2f} segundos com {len(user_profiles)} perfis e {len(noticias)} notícias")

    def predict(self, user_id: str, k: int = 10) -> list:
        """
        Gera as top k recomendações de notícias para um usuário.

        Args:
            user_id (str): UUID do usuário.
            k (int): Número de recomendações a retornar (padrão: 10).

        Returns:
            list: Lista de dicionários com informações das notícias recomendadas.
        """
        predict_start = time.time()
        logger.info(f"Iniciando predição para o user_id: {user_id} com k={k}")

        if user_id not in self.user_profiles:
            logger.warning(f"Usuário {user_id} não encontrado nos perfis pré-processados")
            fallback_start = time.time()
            popular = self.interacoes['history'].str.split(', ').explode().value_counts().head(k).index
            predictions = [
                {
                    'page': page,
                    'title': self.noticias[self.noticias['page'] == page]['title'].iloc[0],
                    'link': self.noticias[self.noticias['page'] == page].get('link', 'N/A').iloc[0]
                }
                for page in popular.tolist()
            ]
            elapsed = time.time() - fallback_start
            total_elapsed = time.time() - predict_start
            logger.info(f"Predições populares geradas em {elapsed:.2f} segundos (total: {total_elapsed:.2f} segundos)")
            return predictions

        logger.debug(f"Extraindo embedding e histórico do usuário {user_id}")
        user_emb = self.user_profiles[user_id]
        user_history = self.interacoes[self.interacoes['userId'] == user_id]['history'].str.split(', ').iloc[0]

        filter_start = time.time()
        candidates = self.noticias[~self.noticias['page'].isin(user_history)]
        logger.info(
            f"Selecionados {len(candidates)} candidatos para predição do usuário {user_id} em {time.time() - filter_start:.2f} segundos")

        build_start = time.time()
        logger.debug(f"Construindo array de entrada para predição com {len(candidates)} candidatos")
        X = np.array([np.concatenate([user_emb, emb]) for emb in candidates['embedding']])
        build_elapsed = time.time() - build_start
        logger.info(f"Dados de entrada construídos com {len(X)} amostras em {build_elapsed:.2f} segundos")

        predict_time = time.time()
        scores = self.regressor.predict(X)
        predict_elapsed = time.time() - predict_time
        logger.info(f"Pontuações previstas para {len(scores)} candidatos em {predict_elapsed:.2f} segundos")

        top_start = time.time()
        top_indices = np.argsort(scores)[-k:][::-1]
        predictions = [
            {
                'page': candidates['page'].iloc[idx],
                'title': candidates['title'].iloc[idx],
                'link': candidates.get('link', pd.Series(['N/A'] * len(candidates))).iloc[idx]
            }
            for idx in top_indices
        ]
        top_elapsed = time.time() - top_start
        total_elapsed = time.time() - predict_start
        logger.info(f"Top {k} recomendações selecionadas em {top_elapsed:.2f} segundos")
        logger.info(
            f"Predição concluída para {user_id}: {len(predictions)} recomendações geradas em {total_elapsed:.2f} segundos")
        return predictions