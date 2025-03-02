import logging
import warnings
import os
import h5py
import pandas as pd
from typing import Dict
import numpy as np
import time

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Esta classe cuida de salvar e carregar dados em arquivos para que possamos reutilizá-los mais tarde,
    como uma "memória" para o programa.
    """

    def __init__(self, cache_dir: str = 'data/cache'):
        """
        Configura o gerenciador de cache.

        Args:
            cache_dir (str): Onde os arquivos serão salvos.
        """
        self.cache_dir = cache_dir
        # Cria o diretório se ele não existir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"CacheManager configurado em: {self.cache_dir}")

    def load_embeddings(self, cache_file: str) -> np.ndarray:
        """
        Carrega embeddings salvos em um arquivo.

        Args:
            cache_file (str): Caminho do arquivo onde os embeddings estão salvos.

        Returns:
            np.ndarray: Os embeddings como um array de números.
        """
        start_time = time.time()
        logger.info(f"Carregando embeddings de {cache_file}")
        # Abre o arquivo no formato HDF5 (um tipo de arquivo que guarda muitos dados)
        with h5py.File(cache_file, 'r') as f:
            embeddings = f['embeddings'][:]  # Pega os embeddings salvos
        elapsed = time.time() - start_time
        logger.info(f"Embeddings carregados em {elapsed:.2f} segundos")
        return embeddings

    def save_embeddings(self, cache_file: str, embeddings: np.ndarray) -> None:
        """
        Salva embeddings em um arquivo.

        Args:
            cache_file (str): Onde salvar os embeddings.
            embeddings (np.ndarray): Os embeddings a serem salvos.
        """
        start_time = time.time()
        # Cria um novo arquivo HDF5 e salva os embeddings com compressão para ocupar menos espaço
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('embeddings', data=embeddings, compression="gzip", compression_opts=4)
        elapsed = time.time() - start_time
        logger.info(f"Embeddings salvos em {cache_file} em {elapsed:.2f} segundos")

    def save_user_profiles(self, cache_file: str, user_profiles: Dict[str, np.ndarray]) -> None:
        """
        Salva os perfis dos usuários em um arquivo.

        Args:
            cache_file (str): Onde salvar os perfis.
            user_profiles (Dict): Dicionário com IDs de usuários e seus perfis (embeddings).
        """
        start_time = time.time()
        user_ids = list(user_profiles.keys())  # Lista de IDs dos usuários
        embeddings = np.array(list(user_profiles.values()))  # Lista de perfis como array
        with h5py.File(cache_file, 'w') as f:
            # Salva os embeddings e os IDs separadamente no arquivo
            f.create_dataset('embeddings', data=embeddings, compression="gzip", compression_opts=4)
            dt = h5py.string_dtype(encoding='utf-8')  # Formato especial para textos
            f.create_dataset('user_ids', data=np.array(user_ids, dtype=object), dtype=dt)
            logger.info(f"Salvou {len(user_ids)} perfis como matriz única")
        elapsed = time.time() - start_time
        logger.info(f"Perfis salvos em {cache_file} em {elapsed:.2f} segundos")

    def save_metrics(self, cache_file: str, metrics: Dict[str, float]) -> None:
        """Salva as métricas em um arquivo HDF5."""
        start_time = time.time()
        with h5py.File(cache_file, 'w') as f:
            for key, value in metrics.items():
                f.create_dataset(key, data=value)
        elapsed = time.time() - start_time
        logger.info(f"Métricas salvas em {cache_file} em {elapsed:.2f} segundos")

    def load_metrics(self, cache_file: str) -> Dict[str, float]:
        """Carrega as métricas de um arquivo HDF5."""
        start_time = time.time()
        with h5py.File(cache_file, 'r') as f:
            metrics = {key: float(f[key][()]) for key in f.keys()}
        elapsed = time.time() - start_time
        logger.info(f"Métricas carregadas de {cache_file} em {elapsed:.2f} segundos")
        return metrics

    def save_dataframe(self, cache_file: str, df: pd.DataFrame, key: str) -> None:
        """
        Salva uma tabela (DataFrame) em um arquivo.

        Args:
            cache_file (str): Onde salvar a tabela.
            df (pd.DataFrame): A tabela a ser salva.
            key (str): Um nome para identificar a tabela no arquivo.
        """
        logger.info(f"Salvando DataFrame em {cache_file}")
        start_time = time.time()
        # Salva a tabela sem compressão e em formato 'table' para evitar forks, suprimindo avisos de desempenho
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            df.to_hdf(cache_file, key=key, mode='w', format='fixed')
        elapsed = time.time() - start_time
        logger.info(f"DataFrame salvo com sucesso em {cache_file} em {elapsed:.2f} segundos")

