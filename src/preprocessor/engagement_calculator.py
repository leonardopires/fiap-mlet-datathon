import logging
import numpy as np

logger = logging.getLogger(__name__)


class EngagementCalculator:
    """
    Esta classe calcula o quanto um usuário gostou de uma notícia com base em cliques, tempo gasto
    e quanto ele rolou a página. Também calcula quão recente é uma interação.
    """

    def calculate_engagement(self, clicks: float, time: float, scroll: float) -> float:
        """
        Calcula uma pontuação para o engajamento do usuário com uma notícia.

        Args:
            clicks (float): Quantos cliques o usuário deu na página.
            time (float): Tempo (em milissegundos) que o usuário passou na página.
            scroll (float): Percentual (0-100) da página que o usuário rolou.

        Returns:
            float: Uma pontuação que mostra o nível de interesse do usuário.
        """
        # Dá pesos diferentes para cada ação:
        # - Cliques contam 30% (multiplica por 0.3)
        # - Tempo (convertido para segundos) conta 50% (divide por 1000 e multiplica por 0.5)
        # - Rolagem conta 20% (multiplica por 0.2)
        engagement = (clicks * 0.3) + (time / 1000 * 1.5) + (scroll * 0.2)  # Aumentar peso do tempo
        return engagement / 10  # Normalizar para escala menor

    def calculate_recency(self, timestamp: int, max_timestamp: int) -> float:
        """
        Calcula quão recente é uma interação comparada ao momento mais recente.

        Args:
            timestamp (int): Hora da interação (em milissegundos).
            max_timestamp (int): Hora mais recente no histórico (em milissegundos).

        Returns:
            float: Um número entre 0 e 1, onde 1 significa "muito recente".
        """
        # Calcula a diferença de tempo em dias
        time_diff = (max_timestamp - timestamp) / (1000 * 60 * 60 * 24)  # Converte milissegundos para dias
        # Usa uma fórmula (exponencial) para dar mais peso a interações recentes
        # Quanto maior a diferença, menor o valor; "7" é como uma "meia-vida" em dias
        recency = np.exp(-time_diff / 7)
        return recency