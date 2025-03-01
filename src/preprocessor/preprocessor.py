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
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', batch_size: int = 64):
        """
        Inicializa o Preprocessor sem carregar o EmbeddingGenerator imediatamente.

        Args:
            model_name (str): Nome do modelo Sentence Transformers (padrão: 'paraphrase-multilingual-MiniLM-L12-v2').
            batch_size (int): Tamanho do batch para geração de embeddings (padrão: 256).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        # EmbeddingGenerator será inicializado no preprocess
        self.embedding_generator = None
        self.cache_manager = CacheManager()
        self.user_profile_builder = UserProfileBuilder(self.device)
        self.resource_logger = ResourceLogger()

    def preprocess(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, subsample_frac: float = None,
                   force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        total_start_time = time.time()
        logger.info("Iniciando pré-processamento completo dos dados")

        if subsample_frac is not None and 0 < subsample_frac < 1:
            start_time = time.time()
            logger.info(f"Aplicando subamostragem com fração {subsample_frac}")
            interacoes = interacoes.sample(frac=subsample_frac, random_state=42)
            noticias = noticias.sample(frac=subsample_frac, random_state=42)
            elapsed = time.time() - start_time
            logger.info(f"Subamostragem concluída em {elapsed:.2f} segundos")

        embedding_cache = os.path.join(self.cache_manager.cache_dir, 'news_embeddings.h5')
        if force_reprocess and os.path.exists(embedding_cache):
            logger.info(f"Removendo cache de embeddings: {embedding_cache}")
            os.remove(embedding_cache)

        # Inicializa o EmbeddingGenerator aqui, após qualquer fork potencial do to_hdf
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator(self.model_name, self.batch_size)
            logger.info(f"EmbeddingGenerator inicializado no dispositivo: {self.device}")

        if os.path.exists(embedding_cache):
            embeddings = self.cache_manager.load_embeddings(embedding_cache)
            if len(embeddings) != len(noticias):
                logger.warning(
                    f"Tamanho do cache de embeddings ({len(embeddings)}) não corresponde ao número de notícias ({len(noticias)}); regenerando embeddings")
                embeddings = self.embedding_generator.generate_embeddings(noticias['title'].tolist(), noticias)
                self.cache_manager.save_embeddings(embedding_cache, embeddings)
            else:
                logger.info("Cache de embeddings válido; usando embeddings existentes")
        else:
            logger.info("Nenhum cache de embeddings encontrado; gerando novos embeddings")
            embeddings = self.embedding_generator.generate_embeddings(noticias['title'].tolist(), noticias)
            self.cache_manager.save_embeddings(embedding_cache, embeddings)

        # Converte embeddings para um array NumPy 2D fixo e datas para datetime64
        noticias['embedding'] = embeddings.tolist()
        noticias['issued'] = pd.to_datetime(noticias['issued'])
        noticias['modified'] = pd.to_datetime(noticias['modified'])

        start_time = time.time()
        logger.info("Criando lookup de embeddings por page")
        page_to_embedding = {page: torch.tensor(emb, device=self.device, dtype=torch.float32)
                             for page, emb in zip(noticias['page'], noticias['embedding'])}
        elapsed = time.time() - start_time
        logger.info(f"Lookup criado com {len(page_to_embedding)} entradas em {elapsed:.2f} segundos")

        if force_reprocess:
            logger.info("Limpando caches antigos de perfis de usuário")
            chunk_files = glob.glob(os.path.join(self.cache_manager.cache_dir, 'user_profiles_*.h5'))
            for f in chunk_files:
                logger.info(f"Removendo: {f}")
                os.remove(f)

        user_profiles = self.user_profile_builder.build_profiles(interacoes, page_to_embedding)
        self.resource_logger.log_resources(f"após batch {len(interacoes)} interações")

        final_cache = os.path.join(self.cache_manager.cache_dir, 'user_profiles_final.h5')
        self.cache_manager.save_user_profiles(final_cache, user_profiles)
        self.resource_logger.log_resources(f"após salvar {len(user_profiles)} perfis")

        interacoes_cache = os.path.join(self.cache_manager.cache_dir, 'interacoes.h5')
        noticias_cache = os.path.join(self.cache_manager.cache_dir, 'noticias.h5')
        self.cache_manager.save_dataframe(interacoes_cache, interacoes, 'interacoes')
        self.cache_manager.save_dataframe(noticias_cache, noticias, 'noticias')

        total_elapsed = time.time() - total_start_time
        logger.info(f"Pré-processamento concluído em {total_elapsed:.2f} segundos")
        return interacoes, noticias, user_profiles
