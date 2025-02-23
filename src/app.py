import logging
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.predictor import Predictor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração da API com FastAPI
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

# Modelo Pydantic para entrada da API
class UserRequest(BaseModel):
    user_id: str

# Inicialização dos dados
def initialize_data():
    global INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR, PREDICTOR
    if not os.path.exists('data/validacao_kaggle.csv'):
        logger.info("Gerando validacao_kaggle.csv...")
        os.system("python data/convert_kaggle.py")
    INTERACOES = data_loader.load_and_concat_files('data/files/treino/treino_parte*.csv')
    NOTICIAS = data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')
    INTERACOES, NOTICIAS, USER_PROFILES = preprocessor.preprocess(INTERACOES, NOTICIAS)

# Rota para o microsite
@app.get("/")
async def read_root():
    file_path = "static/index.html"
    if os.path.exists(file_path):
        logger.info(f"Servindo {file_path}")
        return FileResponse(file_path)
    else:
        logger.error(f"Arquivo {file_path} não encontrado")
        return {"detail": "Arquivo index.html não encontrado"}

# Endpoint para treinamento
@app.post("/train")
async def train_model_endpoint():
    global REGRESSOR, PREDICTOR
    logger.info("Iniciando treinamento via endpoint...")
    REGRESSOR = trainer.train(INTERACOES, NOTICIAS, USER_PROFILES, 'data/validacao_kaggle.csv')
    if REGRESSOR:
        PREDICTOR = Predictor(INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR)
        return {"message": "Modelo treinado com sucesso"}
    return {"message": "Falha no treinamento, dados insuficientes"}

# Endpoint para predição
@app.post("/predict", response_model=dict)
async def get_prediction(request: UserRequest):
    if PREDICTOR is None:
        logger.warning("Modelo não treinado. Execute /train primeiro.")
        return {"detail": "Modelo não treinado. Use o endpoint /train."}
    logger.info(f"Requisição recebida para user_id: {request.user_id}")
    predictions = PREDICTOR.predict(request.user_id)
    logger.info("Requisição concluída.")
    return {"user_id": request.user_id, "acessos_futuros": predictions}

# Monta a pasta static para arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    initialize_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)
