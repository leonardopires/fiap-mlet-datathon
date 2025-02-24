import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    Guarda as informações importantes que o programa precisa compartilhar entre suas partes,
    como os dados e o modelo treinado.
    """

    def __init__(self):
        # Essas variáveis são como "caixas" onde guardamos os dados
        self.INTERACOES = None  # Tabela com interações dos usuários
        self.NOTICIAS = None  # Tabela com notícias
        self.USER_PROFILES = None  # Perfis dos usuários
        self.REGRESSOR = None  # O modelo treinado
        self.PREDICTOR = None  # O sistema que faz recomendações

    def is_initialized(self) -> bool:
        """
        Verifica se todas as "caixas" estão cheias (se os dados estão prontos).

        Returns:
            bool: True se tudo está pronto, False se algo falta.
        """
        return all(
            x is not None for x in [self.INTERACOES, self.NOTICIAS, self.USER_PROFILES, self.REGRESSOR, self.PREDICTOR])

    def reset(self):
        """Limpa todas as 'caixas', voltando ao estado inicial."""
        self.INTERACOES = None
        self.NOTICIAS = None
        self.USER_PROFILES = None
        self.REGRESSOR = None
        self.PREDICTOR = None
