import logging
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from logging.handlers import RotatingFileHandler
import os
from .state_manager import StateManager
from .data_initializer import DataInitializer
from .model_manager import ModelManager
from .models import TrainRequest, UserRequest, PredictionResponse
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.predictor import Predictor

# Configura o logger raiz para capturar todos os logs, incluindo os do Uvicorn
log_path = os.path.join('logs', 'app.log')
# Verifica se a pasta logs existe. Caso contrário, cria ela
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='recomendador-g1 | %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)

# Obtém o logger raiz
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Configura o logger do Uvicorn para usar o mesmo handler
uvicorn_logger = logging.getLogger('uvicorn')
uvicorn_logger.handlers = root_logger.handlers
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.propagate = False  # Evita propagação duplicada

# Obtém o logger para o módulo atual
logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI(
            title="Recomendador G1",
            description="API para recomendação de notícias do G1 baseada em histórico de usuários",
            version="1.0.0"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.state = StateManager()
        self.data_initializer = DataInitializer(DataLoader(), Preprocessor())
        self.model_manager = ModelManager(Trainer(), Predictor)
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        async def read_root():
            start_time = time.time()
            logger.info("Redirecionando para React")
            response = RedirectResponse(url="http://localhost:3000")
            elapsed = time.time() - start_time
            logger.info(f"Redirecionamento concluído em {elapsed:.2f} segundos")
            return response

        @self.app.post("/train")
        async def train_model_endpoint(request: TrainRequest = None):
            import os
            start_time = time.time()
            logger.info("Iniciando treinamento via endpoint")
            subsample_frac = request.subsample_frac if request else None
            force_reprocess = request.force_reprocess if request else False
            force_retrain = request.force_retrain if request else False

            cache_dir = 'data/cache'
            interacoes_file = os.path.join(cache_dir, 'interacoes.h5')
            noticias_file = os.path.join(cache_dir, 'noticias.h5')
            user_profiles_file = os.path.join(cache_dir, 'user_profiles_final.h5')
            data_files_exist = all(os.path.exists(f) for f in [interacoes_file, noticias_file, user_profiles_file])

            if force_reprocess or not data_files_exist:
                logger.info("Pré-processamento necessário; iniciando initialize_data")
                self.data_initializer.initialize_data(self.state, subsample_frac, force_reprocess)
            else:
                if self.data_initializer.load_persisted_data(self.state):
                    logger.info("Dados persistentes carregados; pulando pré-processamento")
                else:
                    logger.error("Dados persistentes encontrados, mas falha ao carregar")
                    raise HTTPException(status_code=500, detail="Falha ao carregar dados persistentes existentes")

            if not os.path.exists('data/validacao.csv'):
                raise HTTPException(status_code=500, detail="data/validacao.csv não encontrado")

            if not os.path.exists('data/validacao_kaggle.csv'):
                logger.info("Gerando validacao_kaggle.csv")
                current_dir = os.getcwd()
                os.chdir('data')
                result = os.system("python convert_kaggle.py")
                os.chdir(current_dir)
                if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
                    raise HTTPException(status_code=500, detail="Falha ao gerar validacao_kaggle.csv")

            self.model_manager.train_model(self.state, 'data/validacao_kaggle.csv', force_retrain)
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
            return {"message": "Modelo treinado com sucesso"}

        @self.app.post("/predict", response_model=PredictionResponse)
        async def get_prediction(request: UserRequest):
            start_time = time.time()
            logger.info(f"Requisição de predição para {request.user_id}")
            predictions = self.model_manager.predict(self.state, request.user_id, request.keywords)
            elapsed = time.time() - start_time
            logger.info(f"Predição concluída em {elapsed:.2f} segundos")
            return {"user_id": request.user_id, "acessos_futuros": predictions}

        @self.app.get("/logs")
        async def get_logs():
            start_time = time.time()
            log_lines = []
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()
                log_lines = [line.strip() for line in log_lines if line.strip()]
                log_lines = log_lines[-2000:] if len(log_lines) > 2000 else log_lines
            except FileNotFoundError:
                logger.warning(f"Arquivo de log {log_file} não encontrado ainda.")
                log_lines = ["Nenhum log disponível ainda."]
            except Exception as e:
                logger.error(f"Erro ao ler logs: {e}")
                log_lines = ["Erro ao carregar logs do servidor."]
            elapsed = time.time() - start_time
            return {"logs": log_lines}

if __name__ == "__main__":
    server = APIServer()
    uvicorn.run(server.app, host="0.0.0.0", port=8000, log_config=None)