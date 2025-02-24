import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.predictor import Predictor
from typing import Optional, Any
import time

# Configura o logger para mensagens detalhadas
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Instancia o aplicativo FastAPI com metadados básicos
app = FastAPI(
    title="Recomendador G1",
    description="API para recomendação de notícias do G1 baseada em histórico de usuários",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa as dependências do backend
data_loader = DataLoader()
preprocessor = Preprocessor()
trainer = Trainer()

# Variáveis globais para armazenar estado entre requisições
INTERACOES = None
NOTICIAS = None
USER_PROFILES = None
REGRESSOR = None
PREDICTOR = None


class UserRequest(BaseModel):
    """
    Modelo Pydantic para requisições de predição.

    Attributes:
        user_id (str): UUID do usuário para o qual gerar recomendações.
    """
    user_id: str


class TrainRequest(BaseModel):
    """
    Modelo Pydantic para requisições de treinamento.

    Attributes:
        subsample_frac (Optional[float]): Fração dos dados a processar (0-1), opcional.
        force_reprocess (Optional[bool]): Força reprocessamento ignorando cache, opcional.
    """
    subsample_frac: Optional[float] = None
    force_reprocess: Optional[bool] = False


def initialize_data(subsample_frac: Optional[float] = None, force_reprocess: Optional[bool] = False):
    """
    Inicializa os dados globais, carregando ou processando interações e notícias.

    Args:
        subsample_frac (Optional[float]): Fração dos dados a processar, se especificada.
        force_reprocess (Optional[bool]): Força reprocessamento completo, ignorando cache.
    """
    global INTERACOES, NOTICIAS, USER_PROFILES
    init_start_time = time.time()
    logger.info("Verificando necessidade de inicialização dos dados globais")
    processed_flag_path = 'data/cache/processed_flag.txt'

    if force_reprocess or not os.path.exists(processed_flag_path):
        logger.info("Dados precisam ser inicializados ou reprocessados")
        # Verifica a existência do arquivo de validação básico
        if not os.path.exists('data/validacao.csv'):
            logger.error("Arquivo de validação data/validacao.csv não encontrado. Execute download_data.py primeiro")
            raise FileNotFoundError("data/validacao.csv não encontrado")

        # Carrega os dados brutos de interações
        load_inter_start = time.time()
        logger.info("Iniciando carregamento dos arquivos de interações")
        INTERACOES = data_loader.load_and_concat_files('data/files/treino/treino_parte*.csv')
        elapsed_inter = time.time() - load_inter_start
        logger.info(f"Interações carregadas: {len(INTERACOES)} registros em {elapsed_inter:.2f} segundos")

        # Carrega os dados brutos de notícias
        load_noticias_start = time.time()
        logger.info("Iniciando carregamento dos arquivos de notícias")
        NOTICIAS = data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')
        elapsed_noticias = time.time() - load_noticias_start
        logger.info(f"Notícias carregadas: {len(NOTICIAS)} registros em {elapsed_noticias:.2f} segundos")

        # Pré-processa os dados com o Preprocessor
        preprocess_start = time.time()
        logger.info("Iniciando pré-processamento dos dados com o Preprocessor")
        INTERACOES, NOTICIAS, USER_PROFILES = preprocessor.preprocess(
            INTERACOES, NOTICIAS, subsample_frac=subsample_frac, force_reprocess=force_reprocess
        )
        preprocess_elapsed = time.time() - preprocess_start
        logger.info(f"Pré-processamento concluído em {preprocess_elapsed:.2f} segundos")
    else:
        logger.info("Dados globais já inicializados, pulando inicialização")

    total_elapsed = time.time() - init_start_time
    logger.info(f"Inicialização dos dados concluída em {total_elapsed:.2f} segundos")
    # creates flag file to indicate that data has been processed
    with open(processed_flag_path, 'w') as f:
        f.write('Data has been processed')



@app.get("/")
async def read_root():
    """
    Redireciona a rota raiz para a aplicação React em localhost:3000.

    Returns:
        RedirectResponse: Redirecionamento para a interface React.
    """
    start_time = time.time()
    logger.info("Requisição recebida na rota raiz, redirecionando para React")
    response = RedirectResponse(url="http://localhost:3000")
    elapsed = time.time() - start_time
    logger.info(f"Redirecionamento para localhost:3000 concluído em {elapsed:.2f} segundos")
    return response


@app.post("/train")
async def train_model_endpoint(request: TrainRequest = None):
    """
    Endpoint para treinar o modelo de recomendação.

    Args:
        request (TrainRequest, opcional): Parâmetros de treinamento (subsample_frac, force_reprocess).

    Returns:
        dict: Mensagem de sucesso ou erro via HTTPException.
    """
    global REGRESSOR, PREDICTOR
    train_start_time = time.time()
    logger.info("Iniciando treinamento via endpoint")

    # Extrai parâmetros da requisição, usando valores padrão se não fornecidos
    subsample_frac = request.subsample_frac if request else None
    force_reprocess = request.force_reprocess if request else False
    logger.debug(f"Parâmetros recebidos: subsample_frac={subsample_frac}, force_reprocess={force_reprocess}")

    # Inicializa os dados com os parâmetros especificados
    logger.info("Chamando inicialização dos dados")
    initialize_data(subsample_frac=subsample_frac, force_reprocess=force_reprocess)

    # Verifica a existência do arquivo de validação básico
    if not os.path.exists('data/validacao.csv'):
        logger.error("Arquivo de validação data/validacao.csv não encontrado")
        raise HTTPException(status_code=500,
                            detail="Arquivo data/validacao.csv não encontrado. Execute download_data.py")

    # Gera o arquivo de validação para Kaggle se necessário
    if not os.path.exists('data/validacao_kaggle.csv'):
        logger.info("Arquivo data/validacao_kaggle.csv não encontrado, gerando com convert_kaggle.py")
        convert_start = time.time()
        current_dir = os.getcwd()
        os.chdir('data')
        result = os.system("python convert_kaggle.py")
        os.chdir(current_dir)
        convert_elapsed = time.time() - convert_start
        if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
            logger.error(f"Falha ao gerar data/validacao_kaggle.csv em {convert_elapsed:.2f} segundos")
            raise HTTPException(status_code=500, detail="Falha ao gerar data/validacao_kaggle.csv")
        logger.info(f"Arquivo data/validacao_kaggle.csv gerado em {convert_elapsed:.2f} segundos")

    # Treina o modelo com os dados pré-processados
    train_model_start = time.time()
    logger.info("Iniciando treinamento do modelo com o Trainer")
    REGRESSOR = trainer.train(INTERACOES, NOTICIAS, USER_PROFILES, 'data/validacao_kaggle.csv')
    train_model_elapsed = time.time() - train_model_start
    if REGRESSOR:
        logger.info(f"Modelo treinado com sucesso em {train_model_elapsed:.2f} segundos")
        # Cria o preditor se o treinamento for bem-sucedido
        PREDICTOR = Predictor(INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR)
        total_elapsed = time.time() - train_start_time
        logger.info(f"Treinamento concluído em {total_elapsed:.2f} segundos")
        return {"message": "Modelo treinado com sucesso"}
    logger.error(f"Falha no treinamento após {train_model_elapsed:.2f} segundos: dados insuficientes")
    raise HTTPException(status_code=500, detail="Falha no treinamento: dados insuficientes")


@app.post("/predict", response_model=dict)
async def get_prediction(request: UserRequest):
    """
    Endpoint para gerar predições de notícias para um usuário.

    Args:
        request (UserRequest): Requisição contendo o user_id.

    Returns:
        dict: Resposta com user_id e lista de acessos futuros recomendados.
    """
    predict_start_time = time.time()
    logger.info(f"Requisição de predição recebida para user_id: {request.user_id}")

    # Verifica se o modelo foi treinado
    if PREDICTOR is None:
        logger.warning("Modelo não treinado detectado ao tentar predizer")
        raise HTTPException(status_code=400, detail="Modelo não treinado. Use o endpoint /train")

    # Gera as predições usando o Predictor
    predict_time = time.time()
    predictions = PREDICTOR.predict(request.user_id)
    predict_elapsed = time.time() - predict_time
    total_elapsed = time.time() - predict_start_time
    logger.info(
        f"Predições geradas para user_id {request.user_id} em {predict_elapsed:.2f} segundos (total: {total_elapsed:.2f} segundos)")
    return {"user_id": request.user_id, "acessos_futuros": predictions}


if __name__ == "__main__":
    """
    Ponto de entrada para execução direta do servidor (fora do Docker).
    """
    logger.info("Iniciando execução direta do servidor FastAPI")
    init_start = time.time()
    initialize_data()
    init_elapsed = time.time() - init_start
    logger.info(f"Dados inicializados para execução direta em {init_elapsed:.2f} segundos")
    uvicorn.run(app, host="0.0.0.0", port=8000)