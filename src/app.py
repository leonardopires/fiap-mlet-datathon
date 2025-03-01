import logging
import os

from src.api.main import create_app  # Importa a função que cria o app do main

# Obtém a variável de ambiente AUTO_INIT, que define se o modelo deve ser treinado automaticamente
AUTO_INIT = os.getenv("AUTO_INIT", "true").lower() == "true"

# Configura o logger para mostrar mensagens úteis com data e hora
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Cria o aplicativo FastAPI chamando a função do main
app = create_app(auto_init=AUTO_INIT)

if __name__ == "__main__":
    """Executa o servidor diretamente, fora do Docker."""
    import uvicorn

    logger.info("Iniciando servidor FastAPI diretamente via app.py")
    # Inicia o servidor na porta 8000, acessível por qualquer computador
    uvicorn.run(app, host="0.0.0.0", port=8000)
