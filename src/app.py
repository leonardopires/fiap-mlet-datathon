import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import psycopg2
from src.data_loader import DataLoader
from src.load_data_to_postgress import load_data_to_postgres
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.predictor import Predictor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Recomendador G1",
    description="API para recomendação de notícias do G1 baseada em histórico de usuários.",
    version="1.0.0"
)

# Dependências
data_loader = DataLoader()
preprocessor = Preprocessor()
trainer = Trainer()

# Variáveis globais
INTERACOES = None
NOTICIAS = None
USER_PROFILES = None
REGRESSOR = None
PREDICTOR = None

class UserRequest(BaseModel):
    user_id: str

def initialize_data():
    global INTERACOES, NOTICIAS, USER_PROFILES
    if INTERACOES is None or NOTICIAS is None or USER_PROFILES is None:
        logger.info("Inicializando dados...")
        if not os.path.exists('data/validacao.csv'):
            logger.error("Arquivo data/validacao.csv não encontrado. Execute download_data.py primeiro.")
            raise FileNotFoundError("data/validacao.csv não encontrado.")
        INTERACOES = data_loader.load_and_concat_files('data/files/treino/treino_parte*.csv')
        NOTICIAS = data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')
        INTERACOES, NOTICIAS, USER_PROFILES = preprocessor.preprocess(INTERACOES, NOTICIAS)
        logger.info("Carregando dados no PostgreSQL...")
        load_data_to_postgres()
        logger.info("Dados inicializados e carregados no PostgreSQL com sucesso.")
    else:
        logger.info("Dados já inicializados, pulando inicialização.")

@app.get("/")
async def read_root():
    file_path = "static/index.html"
    if os.path.exists(file_path):
        logger.info(f"Servindo {file_path}")
        return FileResponse(file_path)
    else:
        logger.error(f"Arquivo {file_path} não encontrado")
        return {"detail": "Arquivo index.html não encontrado"}

@app.post("/train")
async def train_model_endpoint():
    global REGRESSOR, PREDICTOR
    logger.info("Iniciando treinamento via endpoint...")
    # Garante que os dados estejam inicializados
    initialize_data()
    if not os.path.exists('data/validacao.csv'):
        logger.error("Arquivo data/validacao.csv não encontrado.")
        raise HTTPException(status_code=500, detail="Arquivo data/validacao.csv não encontrado. Execute download_data.py.")
    if not os.path.exists('data/validacao_kaggle.csv'):
        logger.info("Gerando data/validacao_kaggle.csv com data/convert_kaggle.py...")
        current_dir = os.getcwd()
        os.chdir('data')
        result = os.system("python convert_kaggle.py")
        os.chdir(current_dir)
        if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
            logger.error("Falha ao gerar data/validacao_kaggle.csv.")
            raise HTTPException(status_code=500, detail="Falha ao gerar data/validacao_kaggle.csv.")
    REGRESSOR = trainer.train(INTERACOES, NOTICIAS, USER_PROFILES, 'data/validacao_kaggle.csv')
    if REGRESSOR:
        PREDICTOR = Predictor(INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR)
        return {"message": "Modelo treinado com sucesso"}
    logger.error("Falha no treinamento: dados insuficientes.")
    raise HTTPException(status_code=500, detail="Falha no treinamento: dados insuficientes.")

@app.post("/predict", response_model=dict)
async def get_prediction(request: UserRequest):
    if PREDICTOR is None:
        logger.warning("Modelo não treinado. Execute /train primeiro.")
        raise HTTPException(status_code=400, detail="Modelo não treinado. Use o endpoint /train.")
    logger.info(f"Requisição recebida para user_id: {request.user_id}")
    predictions = PREDICTOR.predict(request.user_id)
    logger.info("Requisição concluída.")
    return {"user_id": request.user_id, "acessos_futuros": predictions}

app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicialização inicial apenas para testes locais sem Docker
if __name__ == "__main__":
    initialize_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)