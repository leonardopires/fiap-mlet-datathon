import logging
from typing import Optional, List

from fastapi import HTTPException
from src.trainer import Trainer
from src.predictor import Predictor
from .state_manager import StateManager
import time
import joblib
import os
import re

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, trainer: Trainer, predictor_class: type):
        self.trainer = trainer
        self.predictor_class = predictor_class
        self.cache_dir = 'data/cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def train_model(self, state: StateManager, validation_file: str, force_retrain: bool = False) -> None:
        regressor_file = os.path.join(self.cache_dir, 'regressor.pkl')
        if not force_retrain and os.path.exists(regressor_file):
            logger.info("Modelo treinado encontrado em regressor.pkl; carregando modelo existente")
            state.REGRESSOR = joblib.load(regressor_file)
            state.PREDICTOR = self.predictor_class(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES, state.REGRESSOR)
            logger.info("Modelo pré-treinado carregado com sucesso; pronto para predições e cálculo de métricas")
            return

        start_time = time.time()
        logger.info("Iniciando treinamento do modelo")
        state.REGRESSOR = self.trainer.train(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES, validation_file)
        if state.REGRESSOR:
            joblib.dump(state.REGRESSOR, regressor_file)
            logger.info(f"Modelo treinado salvo em {regressor_file}")
            state.PREDICTOR = self.predictor_class(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES, state.REGRESSOR)
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
        else:
            raise HTTPException(status_code=500, detail="Falha no treinamento: dados insuficientes")

    def predict(self, state: StateManager, user_id: str, keywords: Optional[List[str]] = None) -> list[dict]:
        if state.PREDICTOR is None:
            logger.warning("Modelo não treinado")
            raise HTTPException(status_code=400, detail="Modelo não treinado")

        padrao = r"/noticia/(\d{4}/\d{2}/\d{2})/"
        start_time = time.time()
        if user_id not in state.USER_PROFILES:
            logger.info(f"Usuário {user_id} não encontrado; aplicando cold-start")
            popular_news = self.trainer.handle_cold_start(state.NOTICIAS, keywords)
            predictions = [{
                "page": page,
                "title": state.NOTICIAS[state.NOTICIAS['page'] == page]['title'].iloc[0],
                "link": state.NOTICIAS[state.NOTICIAS['page'] == page]['url'].iloc[0],
            } for page in popular_news]
        else:
            predictions = state.PREDICTOR.predict(user_id)
        elapsed = time.time() - start_time
        logger.info(f"Predições geradas para {user_id} em {elapsed:.2f} segundos")
        return predictions