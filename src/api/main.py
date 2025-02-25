import logging
import uvicorn
from .api_server import APIServer

# Configura o logger para mostrar mensagens úteis com data e hora
logger = logging.getLogger(__name__)


def create_app():
    """
    Cria e configura o aplicativo FastAPI para ser usado pelo servidor.

    Returns:
        FastAPI: O aplicativo configurado com todas as rotas.
    """
    logger.info("Criando o aplicativo FastAPI a partir do módulo src.api.main")
    server = APIServer()  # Cria o servidor com todas as configurações
    # Carrega dados persistentes, se disponíveis, mas não força pré-processamento ou treinamento
    if server.data_initializer.load_persisted_data(server.state):
        logger.info("Dados persistentes carregados na inicialização")
    else:
        logger.info("Nenhum dado persistente encontrado; aguardando chamada ao /train")
    return server.app  # Retorna o objeto FastAPI configurado


if __name__ == "__main__":
    """Executa o servidor diretamente se este arquivo for chamado sozinho."""
    logger.info("Iniciando servidor FastAPI diretamente via main.py")
    app = create_app()  # Cria o app
    # Inicia o servidor na porta 8000, acessível por qualquer computador
    uvicorn.run(app, host="0.0.0.0", port=8000)
