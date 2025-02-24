import glob
import logging
import pandas as pd
import time

# Configura o logger para mensagens detalhadas
logger = logging.getLogger(__name__)


class DataLoader:
    def load_and_concat_files(self, pattern: str) -> pd.DataFrame:
        """
        Carrega e concatena arquivos CSV que correspondem ao padrão fornecido.

        Args:
            pattern (str): Padrão de caminho dos arquivos (ex.: 'data/files/treino/treino_parte*.csv').

        Returns:
            pd.DataFrame: DataFrame único contendo todos os dados concatenados.

        Raises:
            FileNotFoundError: Se nenhum arquivo for encontrado no padrão especificado.
        """
        load_start = time.time()
        logger.info(f"Iniciando carregamento de arquivos com padrão: {pattern}")

        # Encontra todos os arquivos que correspondem ao padrão
        files = glob.glob(pattern)
        if not files:
            logger.error(f"Nenhum arquivo encontrado com o padrão: {pattern}")
            raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {pattern}")
        logger.info(f"Encontrados {len(files)} arquivos para carregamento")

        # Lê e concatena os arquivos em um único DataFrame
        concat_start = time.time()
        dataframes = []
        for i, file in enumerate(files):
            logger.debug(f"Carregando arquivo {i + 1}/{len(files)}: {file}")
            df = pd.read_csv(file)
            dataframes.append(df)
        concat_df = pd.concat(dataframes, ignore_index=True)
        concat_elapsed = time.time() - concat_start
        total_elapsed = time.time() - load_start
        logger.info(
            f"Arquivos concatenados: {len(concat_df)} registros em {concat_elapsed:.2f} segundos (total: {total_elapsed:.2f} segundos)")
        return concat_df