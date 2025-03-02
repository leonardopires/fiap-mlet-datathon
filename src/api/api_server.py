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
import requests
from dotenv import load_dotenv
import threading
import hashlib
import json

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

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o logger raiz para capturar todos os logs, incluindo os do Uvicorn
log_path = os.path.join('logs', 'app.log')
os.makedirs('logs', exist_ok=True)

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

# Definição das faixas de desempenho (baseadas no frontend para consistência)
METRICS_RANGES = {
    "precision_at_k": [
        {"range": [0, 0.05], "category": "ruim"},
        {"range": [0.05, 0.1], "category": "aceitável"},
        {"range": [0.1, 0.2], "category": "bom"},
        {"range": [0.2, 1], "category": "excelente"},
    ],
    "recall_at_k": [
        {"range": [0, 0.05], "category": "ruim"},
        {"range": [0.05, 0.08], "category": "aceitável"},
        {"range": [0.08, 0.15], "category": "bom"},
        {"range": [0.15, 1], "category": "excelente"},
    ],
    "mrr": [
        {"range": [0, 0.1], "category": "ruim"},
        {"range": [0.1, 0.2], "category": "aceitável"},
        {"range": [0.2, 0.5], "category": "bom"},
        {"range": [0.5, 1], "category": "excelente"},
    ],
    "intra_list_similarity": [
        {"range": [0.4, 1], "category": "ruim"},
        {"range": [0.3, 0.4], "category": "aceitável"},
        {"range": [0.15, 0.3], "category": "bom"},
        {"range": [0, 0.15], "category": "excelente"},
    ],
    "catalog_coverage": [
        {"range": [0, 0.01], "category": "ruim"},
        {"range": [0.01, 0.02], "category": "aceitável"},
        {"range": [0.02, 0.05], "category": "bom"},
        {"range": [0.05, 1], "category": "excelente"},
    ],
}

METRICS_METADATA = {
    "precision_at_k": {
        "label": "Precisão top 10 (P@10)",
        "description": "Percentual de itens relevantes entre os 10 primeiros recomendados",
        "is_percentage": True
    },
    "recall_at_k": {
        "label": "Recall top 10 (R@10)",
        "description": "Percentual de itens relevantes recuperados entre os 10 primeiros em relação ao total de itens relevantes",
        "is_percentage": True
    },
    "mrr": {
        "label": "Mean Reciprocal Rank (MRR)",
        "description": "Média da posição inversa do primeiro item relevante (entre 0 e 1)",
        "is_percentage": False
    },
    "intra_list_similarity": {
        "label": "Intra-List Similarity (ILS)",
        "description": "Similaridade média entre os itens recomendados, indicando diversidade (0 a 1)",
        "is_percentage": False
    },
    "catalog_coverage": {
        "label": "Cobertura de Catálogo (CC)",
        "description": "Percentual de itens do catálogo que o sistema consegue recomendar",
        "is_percentage": True
    }
}


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
        self.prediction_status = {"running": False, "progress": "idle", "error": None}
        self.interpretation_status = {"running": False, "progress": "idle", "error": None, "interpretation": None}
        self.setup_routes()

        # Criar diretório de cache para interpretações
        self.cache_dir = 'data/cache/interpretations'
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_file_path(self, prompt_hash: str) -> str:
        """Gera o caminho do arquivo de cache baseado no hash do prompt."""
        return os.path.join(self.cache_dir, f"{prompt_hash}.json")

    def _cache_interpretation(self, prompt: str, interpretation: str):
        """Salva o prompt e a interpretação em um arquivo de cache."""
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_file = self._get_cache_file_path(prompt_hash)
        cache_data = {"prompt": prompt, "interpretation": interpretation}
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def _get_cached_interpretation(self, prompt: str):
        """Verifica se o prompt está em cache e retorna a interpretação, se existir."""
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_file = self._get_cache_file_path(prompt_hash)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                if cache_data["prompt"] == prompt:
                    logger.info("Interpretação encontrada no cache")
                    return cache_data["interpretation"]
        return None

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
            predictions = self.model_manager.predict(self.state, request.user_id, request.keywords)
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

    def _interpret_metrics_background(self, metrics: dict):
        self.interpretation_status["running"] = True
        self.interpretation_status["progress"] = "starting"
        self.interpretation_status["error"] = None
        self.interpretation_status["interpretation"] = None
        try:
            start_time = time.time()
            logger.info("Iniciando interpretação de métricas em background")

            # Gerar o prompt dinamicamente
            metrics_text = ""
            for field, metadata in METRICS_METADATA.items():
                value = metrics.get(field)
                if value is None:
                    display_value = "N/A"
                else:
                    display_value = f"{value * 100:.2f}%" if metadata["is_percentage"] else f"{value:.4f}"
                metrics_text += f"- {metadata['label']}: {display_value}\n  ({metadata['description']})\n"

            ranges_text = ""
            for field, ranges in METRICS_RANGES.items():
                metadata = METRICS_METADATA[field]
                ranges_formatted = ", ".join(
                    f"{r['category']} ({r['range'][0]}{'%' if metadata['is_percentage'] else ''}-"
                    f"{'' if r['range'][1] == 1 else r['range'][1]}{'%' if metadata['is_percentage'] else ''})"
                    for r in ranges
                )
                ranges_text += f"- {metadata['label']}: {ranges_formatted}\n"

            prompt = f"""
**Solicitação de Interpretação de Métricas de Sistema de Recomendação**

**Contexto do Projeto:**
Este é um sistema de recomendação de notícias do G1 que utiliza dados de interação de usuários para prever a próxima notícia que eles lerão. O sistema enfrenta desafios como cold-start (usuários ou notícias com poucas interações), a necessidade de priorizar recência (notícias recentes são mais relevantes), e personalização (melhorar recomendações para usuários com mais dados). O modelo foi treinado e avaliado com as seguintes métricas:

**Métricas Atuais:**
{metrics_text}

**Faixas de Desempenho Definidas:**
{ranges_text}

**Solicitação:**
Por favor, analise as métricas fornecidas no contexto do projeto e das faixas de desempenho definidas. Forneça uma interpretação detalhada do desempenho atual do modelo, incluindo:
1. O que cada métrica indica sobre o desempenho do sistema.
2. Áreas de força e fraquezas do modelo, considerando o objetivo de prever a próxima notícia.
3. Recomendações específicas para melhorar o desempenho do modelo, abordando os desafios de cold-start, recência e personalização.
4. Uma conclusão geral sobre o estado atual do sistema e os próximos passos para atingir um desempenho "bom" ou "excelente".
"""

            # Verificar se o prompt está em cache
            cached_interpretation = self._get_cached_interpretation(prompt)
            if cached_interpretation:
                self.interpretation_status["interpretation"] = cached_interpretation
                self.interpretation_status["progress"] = "completed"
                elapsed = time.time() - start_time
                logger.info(f"Interpretação carregada do cache em {elapsed:.2f} segundos")
                return

            # Chamada à API do DeepSeek
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise Exception("Chave da API do DeepSeek não configurada")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "system",
                     "content": "Você é um especialista em interpretação de métricas de sistemas de recomendação."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }

            self.interpretation_status["progress"] = "processing"
            logger.info("Chamando a API do DeepSeek para interpretação de métricas")
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"Erro na API do DeepSeek: {response.text}")
                raise Exception(f"Erro na API do DeepSeek: {response.text}")

            interpretation = response.json().get("choices", [{}])[0].get("message", {}).get("content",
                                                                                            "Erro ao obter interpretação")
            elapsed = time.time() - start_time
            logger.info(f"Interpretação concluída em {elapsed:.2f} segundos")

            # Salvar no cache
            self._cache_interpretation(prompt, interpretation)

            self.interpretation_status["interpretation"] = interpretation
            self.interpretation_status["progress"] = "completed"
        except Exception as e:
            self.interpretation_status["error"] = str(e)
            logger.error(f"Erro ao interpretar métricas: {e}")
        finally:
            self.interpretation_status["running"] = False

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
            predictions = self.model_manager.predict(self.state, request.user_id, request.keywords)
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
        async def get_metrics(force_recalc: bool = False, fetch_only_existing: bool = False,
                              background_tasks: BackgroundTasks = None):
            logger.info("Requisição para obter métricas recebida")
            if hasattr(self.state, 'metrics') and not force_recalc:
                return {"metrics": self.state.metrics}
            if fetch_only_existing:
                return {"metrics": None}
            background_tasks.add_task(self._calculate_metrics_background, force_recalc)
            return {"message": "Cálculo de métricas iniciado em background; acompanhe via /metrics/status"}

        @self.app.get("/metrics/status")
        async def get_metrics_status():
            return self.metrics_status

        @self.app.post("/interpret-metrics")
        async def interpret_metrics(metrics: dict, background_tasks: BackgroundTasks = None):
            logger.info("Requisição para interpretar métricas recebida")
            if self.interpretation_status["running"]:
                raise HTTPException(status_code=429,
                                    detail="Uma interpretação de métricas já está em andamento. Acompanhe via /interpret/status.")

            background_tasks.add_task(self._interpret_metrics_background, metrics)
            return {"message": "Interpretação de métricas iniciada; acompanhe via /interpret/status"}

        @self.app.get("/interpret/status")
        async def get_interpret_status():
            return self.interpretation_status

        @self.app.websocket("/ws/interpret/status")
        async def websocket_interpret_status(websocket: WebSocket):
            async def interpret_status_callback(ws):
                while True:
                    await ws.send_json(self.interpretation_status)
                    await asyncio.sleep(1)

            await self._handle_websocket(websocket, interpret_status_callback, "interpret/status")

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