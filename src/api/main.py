import logging
import uvicorn
import os
from .api_server import APIServer

# Configura o logger para mostrar mensagens úteis com data e hora
logger = logging.getLogger(__name__)

# Obtém a variável de ambiente AUTO_INIT, que define se o modelo deve ser treinado automaticamente
AUTO_INIT = os.getenv("AUTO_INIT", "true").lower() == "true"


def create_app(auto_init: bool = AUTO_INIT):
    """
    Cria e configura o aplicativo FastAPI para ser usado pelo servidor.

    Args:
        auto_init (bool): Se True, inicializa automaticamente os dados e treina o modelo na criação se necessário.

    Returns:
        FastAPI: O aplicativo configurado com todas as rotas.
    """
    logger.debug(f"Configuração de inicialização: AUTO_INIT={AUTO_INIT}, auto_init={auto_init}")

    logger.info("Criando o aplicativo FastAPI a partir do módulo src.api.main")
    server = APIServer()

    # Tenta carregar dados persistentes; se falhar e AUTO_INIT estiver ativado, inicializa
    loaded = server.data_initializer.load_persisted_data(server.state)

    if not loaded or server.state.REGRESSOR is None:
        if auto_init:
            if not loaded:
                logger.info("Nenhum dado persistente completo encontrado; inicializando automaticamente")
            elif server.state.REGRESSOR is None:
                logger.info("Dados persistentes carregados, mas modelo ausente; iniciando treinamento")
            server.data_initializer.initialize_data(server.state, force_reprocess=False)
            if not os.path.exists('data/validacao_kaggle.csv'):
                logger.info("Gerando validacao_kaggle.csv")
                current_dir = os.getcwd()
                os.chdir('data')
                result = os.system("python convert_kaggle.py")
                os.chdir(current_dir)
                if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
                    logger.error("Falha ao gerar validacao_kaggle.csv")
                    raise RuntimeError("Falha ao gerar validacao_kaggle.csv")
            server.model_manager.train_model(server.state, 'data/validacao_kaggle.csv', force_retrain=False)
            logger.info("Inicialização e treinamento automáticos concluídos")
        else:
            logger.info("Dados persistentes incompletos e AUTO_INIT desativado; aguardando /train")
    else:
        logger.info("Dados persistentes e modelo carregados na inicialização")

    return server.app


if __name__ == "__main__":
    """Executa o servidor diretamente se este arquivo for chamado sozinho."""
    logger.info("Iniciando servidor FastAPI diretamente via main.py")
    app = create_app(auto_init=AUTO_INIT)  # Cria o app
    # Inicia o servidor na porta 8000, acessível por qualquer computador
    uvicorn.run(app, host="0.0.0.0", port=8000)
