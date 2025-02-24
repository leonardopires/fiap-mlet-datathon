import logging
import os
import pandas as pd
import h5py
import joblib
import time
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from typing import Optional
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class DataInitializer:
    """
    Prepara os dados que o programa vai usar, carregando de arquivos salvos ou processando do zero.
    """

    def __init__(self, data_loader: DataLoader, preprocessor: Preprocessor):
        """
        Configura o inicializador de dados.

        Args:
            data_loader: Uma ferramenta para carregar os dados brutos.
            preprocessor: Uma ferramenta para processar os dados.
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.cache_dir = 'data/cache'  # Onde os dados processados são salvos

    def load_persisted_data(self, state: StateManager) -> bool:
        """
        Tenta carregar os dados já salvos para evitar refazer o trabalho.

        Args:
            state: O lugar onde vamos guardar os dados carregados.

        Returns:
            bool: True se carregou tudo, False se algo deu errado ou falta.
        """
        # Define os caminhos dos arquivos salvos
        interacoes_file = os.path.join(self.cache_dir, 'interacoes.h5')
        noticias_file = os.path.join(self.cache_dir, 'noticias.h5')
        user_profiles_file = os.path.join(self.cache_dir, 'user_profiles_final.h5')
        regressor_file = os.path.join(self.cache_dir, 'regressor.pkl')

        try:
            # Verifica se todos os arquivos existem
            if not all(os.path.exists(f) for f in [interacoes_file, noticias_file, user_profiles_file, regressor_file]):
                logger.info("Arquivos persistentes incompletos.")
                return False

            # Carrega as interações dos usuários
            logger.info(f"Carregando INTERACOES de {interacoes_file}")
            state.INTERACOES = pd.read_hdf(interacoes_file, key='interacoes')

            # Carrega as notícias
            logger.info(f"Carregando NOTICIAS de {noticias_file}")
            state.NOTICIAS = pd.read_hdf(noticias_file, key='noticias')

            # Carrega os perfis dos usuários
            logger.info(f"Carregando USER_PROFILES de {user_profiles_file}")
            with h5py.File(user_profiles_file, 'r') as f:
                embeddings = f['embeddings'][:]  # Números que representam os perfis
                user_ids = f['user_ids'][:].astype(str)  # IDs dos usuários
                state.USER_PROFILES = dict(zip(user_ids, embeddings))  # Junta IDs e perfis

            # Carrega o modelo treinado
            logger.info(f"Carregando REGRESSOR de {regressor_file}")
            state.REGRESSOR = joblib.load(regressor_file)

            logger.info("Dados persistentes carregados com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar dados persistentes: {e}")
            return False

    def initialize_data(self, state: StateManager, subsample_frac: Optional[float] = None,
                        force_reprocess: Optional[bool] = False):
        """
        Prepara os dados, carregando do cache ou processando se necessário.

        Args:
            state: Onde os dados vão ser guardados.
            subsample_frac: Parte dos dados a usar (ex.: 0.1 para 10%).
            force_reprocess: Se True, refaz tudo do zero.
        """
        start_time = time.time()
        logger.info("Iniciando inicialização dos dados")

        processed_flag_path = os.path.join(self.cache_dir, 'processed_flag.txt')
        # Tenta carregar os dados salvos, a menos que seja forçado a reprocessar
        if not force_reprocess and self.load_persisted_data(state):
            logger.info("Dados carregados de cache. Pulando reprocessamento.")
        else:
            # Verifica se temos o arquivo básico necessário
            if not os.path.exists('data/validacao.csv'):
                logger.error("Arquivo data/validacao.csv não encontrado")
                raise FileNotFoundError("data/validacao.csv não encontrado")

            # Carrega os dados brutos de interações
            logger.info("Carregando interações brutas")
            state.INTERACOES = self.data_loader.load_and_concat_files('data/files/treino/treino_parte*.csv')
            logger.info(f"Interações carregadas: {len(state.INTERACOES)} registros")

            # Carrega os dados brutos de notícias
            logger.info("Carregando notícias brutas")
            state.NOTICIAS = self.data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')
            logger.info(f"Notícias carregadas: {len(state.NOTICIAS)} registros")

            # Processa os dados para ficarem prontos para uso
            logger.info("Pré-processando dados")
            state.INTERACOES, state.NOTICIAS, state.USER_PROFILES = self.preprocessor.preprocess(
                state.INTERACOES, state.NOTICIAS, subsample_frac=subsample_frac, force_reprocess=force_reprocess
            )

            # Marca que os dados foram processados
            if not force_reprocess or not os.path.exists(processed_flag_path):
                with open(processed_flag_path, 'w') as f:
                    f.write('Data has been processed')

        elapsed = time.time() - start_time
        logger.info(f"Inicialização concluída em {elapsed:.2f} segundos")
