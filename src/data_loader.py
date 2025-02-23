import glob
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    def load_and_concat_files(self, pattern: str) -> pd.DataFrame:
        logger.info(f"Carregando arquivos de {pattern}...")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {pattern}")
        logger.info(f"Encontrados {len(files)} arquivos.")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
