import logging
import torch
import pandas as pd
from typing import Tuple, Dict
import glob
import os
import time
from .embedding_generator import EmbeddingGenerator
from .cache_manager import CacheManager
from .user_profile_builder import UserProfileBuilder
from .resource_logger import ResourceLogger

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Esta classe organiza todo o processo de preparar os dados para o sistema de recomendação,
    usando outras classes para fazer cada parte do trabalho.
    """

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', batch_size: int = 256):
        """
        Configura o organizador do pré-processamento.

        Args:
            model_name (str): Nome do modelo para criar embeddings.
            batch_size (int): Quantos itens processar de uma vez.
        """
        # Decide se usamos GPU ou CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cria as "ferramentas" que vamos usar
        self.embedding_generator = EmbeddingGenerator(model_name, batch_size)
        self.cache_manager = CacheManager()
        self.user_profile_builder = UserProfileBuilder(self.device)
        self.resource_logger = ResourceLogger()

    def preprocess(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, subsample_frac: float = None,
                   force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Pré-processa os dados, gerando embeddings para notícias e perfis para usuários.

        Args:
            interacoes (pd.DataFrame): Tabela com as interações dos usuários.
            noticias (pd.DataFrame): Tabela com as notícias.
            subsample_frac (float): Se quiser usar só uma parte dos dados (ex.: 0.1 para 10%).
            force_reprocess (bool): Se True, refaz tudo mesmo que já exista algo salvo.

        Returns:
            Tuple: As interações, notícias e perfis dos usuários atualizados.
        """
        total_start_time = time.time()
        logger.info("Iniciando pré-processamento completo dos dados")

        # Se quiser usar só uma parte dos dados, reduz a quantidade aqui
        if subsample_frac is not None and 0 < subsample_frac < 1:
            start_time = time.time()
            logger.info(f"Aplicando subamostragem com fração {subsample_frac}")
            interacoes = interacoes.sample(frac=subsample_frac, random_state=42)  # Pega uma amostra aleatória
            noticias = noticias.sample(frac=subsample_frac, random_state=42)
            elapsed = time.time() - start_time
            logger.info(f"Subamostragem concluída em {elapsed:.2f} segundos")

        # Define onde os embeddings das notícias serão salvos
        embedding_cache = os.path.join(self.cache_manager.cache_dir, 'news_embeddings.h5')
        # Se forçar o reprocessamento, apaga o arquivo antigo
        if force_reprocess and os.path.exists(embedding_cache):
            logger.info(f"Removendo cache de embeddings: {embedding_cache}")
            os.remove(embedding_cache)

        # Carrega ou cria os embeddings das notícias
        if os.path.exists(embedding_cache):
            embeddings = self.cache_manager.load_embeddings(embedding_cache)
        else:
            embeddings = self.embedding_generator.generate_embeddings(noticias['title'].tolist())
            self.cache_manager.save_embeddings(embedding_cache, embeddings)
        noticias['embedding'] = embeddings.tolist()  # Adiciona os embeddings às notícias

        # Cria um "dicionário rápido" para encontrar embeddings das notícias
        start_time = time.time()
        logger.info("Criando lookup de embeddings por page")
        page_to_embedding = {page: torch.tensor(emb, device=self.device, dtype=torch.float32)
                             for page, emb in zip(noticias['page'], noticias['embedding'])}
        elapsed = time.time() - start_time
        logger.info(f"Lookup criado com {len(page_to_embedding)} entradas em {elapsed:.2f} segundos")

        # Apaga arquivos antigos de perfis se forçar o reprocessamento
        if force_reprocess:
            logger.info("Limpando caches antigos de perfis de usuário")
            chunk_files = glob.glob(os.path.join(self.cache_manager.cache_dir, 'user_profiles_*.h5'))
            for f in chunk_files:
                logger.info(f"Removendo: {f}")
                os.remove(f)

        # Cria os perfis dos usuários
        user_profiles = self.user_profile_builder.build_profiles(interacoes, page_to_embedding)
        self.resource_logger.log_resources(f"após batch {len(interacoes)} interações")

        # Salva os perfis finais
        final_cache = os.path.join(self.cache_manager.cache_dir, 'user_profiles_final.h5')
        self.cache_manager.save_user_profiles(final_cache, user_profiles)
        self.resource_logger.log_resources(f"após salvar {len(user_profiles)} perfis")

        # Salva as tabelas de interações e notícias
        interacoes_cache = os.path.join(self.cache_manager.cache_dir, 'interacoes.h5')
        noticias_cache = os.path.join(self.cache_manager.cache_dir, 'noticias.h5')
        self.cache_manager.save_dataframe(interacoes_cache, interacoes, 'interacoes')
        self.cache_manager.save_dataframe(noticias_cache, noticias, 'noticias')

        total_elapsed = time.time() - total_start_time
        logger.info(f"Pré-processamento concluído em {total_elapsed:.2f} segundos")
        return interacoes, noticias, user_profiles