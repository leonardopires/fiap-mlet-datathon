import asyncio
import logging
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, WebSocketException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from logging.handlers import RotatingFileHandler
import os

from starlette.websockets import WebSocketState, WebSocketDisconnect

from src.data_loader import DataLoader
from src.predictor import Predictor
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from .state_manager import StateManager
from .data_initializer import DataInitializer
from .metrics_calculator import MetricsCalculator
from .model_manager import ModelManager
from .models import TrainRequest, UserRequest, PredictionResponse

# Configura o logger raiz para capturar todos os logs, incluindo os do Uvicorn
log_path = os.path.join('data/logs', 'app.log')
os.makedirs('data', exist_ok=True)
os.makedirs('data/logs', exist_ok=True)

log_handler = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=5)

if os.path.exists(log_path):
    log_handler.doRollover()

logging.basicConfig(
    level=logging.INFO,
    format='recomendador-g1 | %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler(sys.stdout)
    ]
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

uvicorn_logger = logging.getLogger('uvicorn')
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.propagate = False

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
            allow_headers=["*"],
            allow_credentials=True,
            allow_methods=["*"],
        )
        self.state = StateManager()
        self.data_initializer = DataInitializer(DataLoader(), Preprocessor())
        self.model_manager = ModelManager(Trainer(), Predictor)
        self.metrics_calculator = MetricsCalculator(self.state)
        self.training_status = {"running": False, "progress": "idle", "error": None}
        self.metrics_status = {"running": False, "progress": "idle", "error": None}
        self.prediction_status = {"running": False, "progress": "idle", "error": None}  # Novo estado para predição
        self.setup_routes()

    async def _handle_websocket(self, websocket: WebSocket, callback: callable, callback_name: str):
        """Gerencia conexão WebSocket, executa a callback e trata erros de forma genérica."""
        await websocket.accept()
        logger.debug(f"WebSocket de {callback_name} conectado")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await callback(websocket)
            else:
                logger.info(f"WebSocket de {callback_name} desconectado")
        except WebSocketDisconnect as re:
            logger.debug(f"WebSocket de {callback_name} desconectado normalmente:")
        except Exception as ex:
            logger.error(f"Erro no WebSocket de {callback_name}:", ex)
            raise ex
        finally:
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    await websocket.close()
            except Exception as e:
                logger.debug(f"Ignorando erro ao fechar WebSocket de {callback_name}:", e)

    def _read_and_filter_logs(self, limit: int = 2000) -> list[str]:
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            filtered_lines = [
                line.strip() for line in log_lines
                if line.strip() and all(x not in line for x in ["GET /logs HTTP/1.1", "POST /logs HTTP/1.1"])
            ]
            return filtered_lines[-limit:] if len(filtered_lines) > limit else filtered_lines
        except FileNotFoundError:
            logger.warning(f"Arquivo de log {log_path} não encontrado ainda.")
            return ["Nenhum log disponível ainda."]
        except Exception as e:
            logger.error(f"Erro ao ler logs: {e}")
            return ["Erro ao carregar logs do servidor."]

    def _train_model_background(self, request: TrainRequest):
        self.training_status["running"] = True
        self.training_status["progress"] = "starting"
        self.training_status["error"] = None
        try:
            start_time = time.time()
            logger.info("Iniciando treinamento em background")
            subsample_frac = request.subsample_frac if request else None
            force_reprocess = request.force_reprocess if request else False
            force_retrain = request.force_retrain if request else False

            cache_dir = 'data/cache'
            interacoes_file = os.path.join(cache_dir, 'interacoes.h5')
            noticias_file = os.path.join(cache_dir, 'noticias.h5')
            user_profiles_file = os.path.join(cache_dir, 'user_profiles_final.h5')
            data_files_exist = all(os.path.exists(f) for f in [interacoes_file, noticias_file, user_profiles_file])

            if force_reprocess or not data_files_exist:
                self.training_status["progress"] = "preprocessing"
                logger.info("Pré-processamento necessário; iniciando initialize_data")
                self.data_initializer.initialize_data(self.state, subsample_frac, force_reprocess)
            else:
                if self.data_initializer.load_persisted_data(self.state):
                    logger.info("Dados persistentes carregados; pulando pré-processamento")
                else:
                    self.training_status["error"] = "Falha ao carregar dados persistentes"
                    raise Exception("Dados persistentes incompletos ou corrompidos. Force novo treinamento.")

            if not os.path.exists('data/validacao.csv'):
                self.training_status["error"] = "data/validacao.csv não encontrado"
                raise Exception("Arquivo de validação não encontrado em data/validacao.csv.")

            if not os.path.exists('data/validacao_kaggle.csv'):
                self.training_status["progress"] = "converting kaggle data"
                logger.info("Gerando validacao_kaggle.csv")
                current_dir = os.getcwd()
                os.chdir('data')
                result = os.system("python convert_kaggle.py")
                os.chdir(current_dir)
                if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
                    self.training_status["error"] = "Falha ao gerar validacao_kaggle.csv"
                    raise Exception("Falha ao gerar data/validacao_kaggle.csv")

            self.training_status["progress"] = "training"
            self.model_manager.train_model(self.state, 'data/validacao_kaggle.csv', force_retrain)
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
            self.training_status["progress"] = "completed"
        except Exception as e:
            self.training_status["error"] = str(e)
            logger.error(f"Erro durante treinamento: {e}")
        finally:
            self.training_status["running"] = False

    def _calculate_metrics_background(self, force_recalc: bool):
        self.metrics_status["running"] = True
        self.metrics_status["progress"] = "starting"
        self.metrics_status["error"] = None
        try:
            start_time = time.time()
            logger.info("Calculando métricas em background")
            if not self.state.is_initialized():
                self.metrics_status["error"] = "Modelo ou dados não carregados"
                logger.error("Modelo ou dados não carregados")
                raise Exception("Model or data not initialized")
            self.metrics_status["progress"] = "calculating"
            metrics = self.metrics_calculator.calculate_metrics(k=10, force_recalc=force_recalc)
            elapsed = time.time() - start_time
            logger.info(f"Métricas calculadas em {elapsed:.2f} segundos")
            self.state.metrics = metrics
            self.metrics_status["progress"] = "completed"
        except Exception as e:
            self.metrics_status["error"] = str(e)
            logger.error(f"Erro ao calcular métricas: {e}")
        finally:
            self.metrics_status["running"] = False

    def _predict_background(self, request: UserRequest):
        self.prediction_status["running"] = True
        self.prediction_status["progress"] = "starting"
        self.prediction_status["error"] = None
        try:
            start_time = time.time()
            logger.info(f"Requisição de predição para {request.user_id} em background")
            self.prediction_status["progress"] = "predicting"
            predictions = self.model_manager.predict(self.state, request.user_id, keywords=request.keywords)
            elapsed = time.time() - start_time
            logger.info(f"Predição concluída em {elapsed:.2f} segundos")
            self.prediction_status["progress"] = "completed"
            return predictions
        except Exception as e:
            self.prediction_status["error"] = str(e)
            logger.error(f"Erro durante predição: {e}")
            raise
        finally:
            self.prediction_status["running"] = False

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
        async def train_model_endpoint(request: TrainRequest = None, background_tasks: BackgroundTasks = None):
            logger.info("Requisição de treinamento recebida")
            background_tasks.add_task(self._train_model_background, request)
            return {"message": "Treinamento iniciado em background; acompanhe via logs ou /train/status"}

        @self.app.get("/train/status")
        async def get_train_status():
            return self.training_status

        @self.app.post("/predict")
        async def get_prediction(request: UserRequest, background_tasks: BackgroundTasks = None):
            if self.prediction_status["running"]:
                raise HTTPException(status_code=429,
                                    detail="Uma solicitação de predição já está em andamento. Acompanhe via /predict/status.")

            background_tasks.add_task(self._predict_background, request)
            return {"message": "Predição iniciada; acompanhe via /predict/status"}

        @self.app.post("/predict_foreground", response_model=PredictionResponse)
        async def get_prediction_foreground(request: UserRequest):
            if self.prediction_status["running"]:
                raise HTTPException(status_code=429,
                                    detail="Uma solicitação de predição já está em andamento. Acompanhe via /predict/status.")

            start_time = time.time()
            logger.info(f"Requisição de predição para {request.user_id}")
            predictions = self.model_manager.predict(self.state, request.user_id, number_of_records=20, keywords=request.keywords)
            elapsed = time.time() - start_time
            logger.info(f"Predição concluída em {elapsed:.2f} segundos")
            return {"user_id": request.user_id, "acessos_futuros": predictions}

        @self.app.get("/predict/status")
        async def get_predict_status():
            return self.prediction_status

        @self.app.get("/logs")
        async def get_logs():
            log_lines = self._read_and_filter_logs()
            return {"logs": log_lines}

        @self.app.websocket("/ws/logs")
        async def websocket_logs(websocket: WebSocket):
            async def logs_callback(ws):
                initial_logs = self._read_and_filter_logs()
                for line in initial_logs:
                    await ws.send_text(line)
                last_pos = os.path.getsize(log_path)
                while True:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        if new_lines:
                            filtered_new_lines = [
                                line.strip() for line in new_lines
                                if line.strip() and all(
                                    x not in line for x in ["GET /logs HTTP/1.1", "POST /logs HTTP/1.1"])
                            ]
                            for line in filtered_new_lines:
                                await ws.send_text(line)
                        last_pos = f.tell()
                    await asyncio.sleep(1)

            await self._handle_websocket(websocket, logs_callback, "logs")

        @self.app.get("/metrics", response_model=dict)
        async def get_metrics(
                force_recalc: bool = False,
                fetch_only_existing: bool = False,
                background_tasks: BackgroundTasks = None
        ):
            logger.info("Requisição para obter métricas recebida")
            if hasattr(self.state, 'metrics') and not force_recalc:
                return {"metrics": self.state.metrics}
            if fetch_only_existing:
                return {"metrics": None}  # Retorna null se não houver métricas salvas
            background_tasks.add_task(self._calculate_metrics_background, force_recalc)
            return {"message": "Cálculo de métricas iniciado em background; acompanhe via /metrics/status"}

        @self.app.get("/metrics/status")
        async def get_metrics_status():
            return self.metrics_status

        @self.app.websocket("/ws/status")
        async def websocket_status(websocket: WebSocket):
            async def status_callback(ws):
                while True:
                    status = {
                        "training": self.training_status,
                        "metrics": self.metrics_status
                    }
                    await ws.send_json(status)
                    await asyncio.sleep(1)

            await self._handle_websocket(websocket, status_callback, "status")

        @self.app.websocket("/ws/predict/status")
        async def websocket_predict_status(websocket: WebSocket):
            async def predict_status_callback(ws):
                while True:
                    status = self.prediction_status
                    await ws.send_json(status)
                    await asyncio.sleep(1)

            await self._handle_websocket(websocket, predict_status_callback, "predict/status")


if __name__ == "__main__":
    server = APIServer()
    uvicorn.run(server.app, host="0.0.0.0", port=8000, log_config=None)
