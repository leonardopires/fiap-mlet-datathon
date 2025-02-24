import logging
from fastapi import HTTPException
from src.trainer import Trainer
from src.predictor import Predictor
from .state_manager import StateManager
import time

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Cuida do treinamento do modelo que faz recomendações e das predições para os usuários.
    """

    def __init__(self, trainer: Trainer, predictor_class: type):
        """
        Configura o gerenciador do modelo.

        Args:
            trainer: Ferramenta para treinar o modelo.
            predictor_class: Tipo da classe que faz recomendações.
        """
        self.trainer = trainer
        self.predictor_class = predictor_class

    def train_model(self, state: StateManager, validation_file: str) -> None:
        """
        Treina o modelo com os dados preparados.

        Args:
            state: Onde os dados e o modelo são guardados.
            validation_file: Arquivo usado para testar o modelo.
        """
        start_time = time.time()
        logger.info("Iniciando treinamento do modelo")

        # Treina o modelo com os dados de interações, notícias e perfis
        state.REGRESSOR = self.trainer.train(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES, validation_file)
        if state.REGRESSOR:
            # Se o treinamento deu certo, cria o sistema de recomendações
            state.PREDICTOR = self.predictor_class(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES,
                                                   state.REGRESSOR)
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
        else:
            raise HTTPException(status_code=500, detail="Falha no treinamento: dados insuficientes")

    def predict(self, state: StateManager, user_id: str) -> list[dict]:
        """
        Faz recomendações de notícias para um usuário.

        Args:
            state: Onde o modelo treinado está guardado.
            user_id: O ID do usuário para quem queremos recomendar.

        Returns:
            list[dict]: Lista de notícias recomendadas.
        """
        if state.PREDICTOR is None:
            logger.warning("Modelo não treinado")
            raise HTTPException(status_code=400, detail="Modelo não treinado")

        start_time = time.time()
        # Usa o modelo para prever quais notícias o usuário vai gostar
        predictions = state.PREDICTOR.predict(user_id)
        elapsed = time.time() - start_time
        logger.info(f"Predições geradas para {user_id} em {elapsed:.2f} segundos")
        return predictions
