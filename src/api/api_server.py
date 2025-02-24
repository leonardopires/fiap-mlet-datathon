import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from .state_manager import StateManager
from .data_initializer import DataInitializer
from .model_manager import ModelManager
from .models import TrainRequest, UserRequest, PredictionResponse
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.predictor import Predictor

logger = logging.getLogger(__name__)


class APIServer:
    """
    Configura o servidor web que recebe pedidos (como treinar ou recomendar) e responde.
    """

    def __init__(self):
        """
        Configura o servidor FastAPI.
        """
        # Cria o aplicativo web com informações básicas
        self.app = FastAPI(
            title="Recomendador G1",
            description="API para recomendação de notícias do G1 baseada em histórico de usuários",
            version="1.0.0"
        )
        # Permite que outros programas (como um site) usem esta API
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Cria as ferramentas que o servidor vai usar
        self.state = StateManager()
        self.data_initializer = DataInitializer(DataLoader(), Preprocessor())
        self.model_manager = ModelManager(Trainer(), Predictor)
        self.setup_routes()  # Configura os "endereços" que o servidor responde

    def setup_routes(self):
        """Define os 'endereços' (rotas) que o servidor responde."""

        @self.app.get("/")
        async def read_root():
            """Redireciona quem acessa a raiz para o site em React."""
            start_time = time.time()
            logger.info("Redirecionando para React")
            response = RedirectResponse(url="http://localhost:3000")
            elapsed = time.time() - start_time
            logger.info(f"Redirecionamento concluído em {elapsed:.2f} segundos")
            return response

        @self.app.post("/train")
        async def train_model_endpoint(request: TrainRequest = None):
            """Treina o modelo quando alguém pede."""
            import os  # Importado aqui para evitar dependências desnecessárias fora do endpoint
            from fastapi import HTTPException
            start_time = time.time()
            logger.info("Iniciando treinamento via endpoint")
            # Pega os parâmetros da requisição ou usa padrões
            subsample_frac = request.subsample_frac if request else None
            force_reprocess = request.force_reprocess if request else False

            # Prepara os dados
            self.data_initializer.initialize_data(self.state, subsample_frac, force_reprocess)
            if not os.path.exists('data/validacao.csv'):
                raise HTTPException(status_code=500, detail="data/validacao.csv não encontrado")

            # Gera um arquivo especial para testar o modelo, se não existir
            if not os.path.exists('data/validacao_kaggle.csv'):
                logger.info("Gerando validacao_kaggle.csv")
                current_dir = os.getcwd()
                os.chdir('data')
                result = os.system("python convert_kaggle.py")
                os.chdir(current_dir)
                if result != 0 or not os.path.exists('data/validacao_kaggle.csv'):
                    raise HTTPException(status_code=500, detail="Falha ao gerar validacao_kaggle.csv")

            # Treina o modelo
            self.model_manager.train_model(self.state, 'data/validacao_kaggle.csv')
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
            return {"message": "Modelo treinado com sucesso"}

        @self.app.post("/predict", response_model=PredictionResponse)
        async def get_prediction(request: UserRequest):
            """Dá recomendações para um usuário quando pedido."""
            start_time = time.time()
            logger.info(f"Requisição de predição para {request.user_id}")
            predictions = self.model_manager.predict(self.state, request.user_id)
            elapsed = time.time() - start_time
            logger.info(f"Predição concluída em {elapsed:.2f} segundos")
            return {"user_id": request.user_id, "acessos_futuros": predictions}
