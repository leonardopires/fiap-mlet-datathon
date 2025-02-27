#!/usr/bin/env python
# coding: utf-8

import os
import zipfile
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_ID = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
OUTPUT_ZIP = "data.zip"
DATA_DIR = "data"

# Verifica se o gdown está instalado, senão instala automaticamente
def install_gdown():
    try:
        import gdown
        logger.info("gdown já está instalado.")
    except ImportError:
        logger.info("Instalando gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

def is_valid_zip(file_path):
    """
    Verifica se o arquivo baixado é realmente um ZIP válido.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False

def download_file():
    """
    Faz o download do arquivo do Google Drive usando `gdown`, que resolve automaticamente os problemas de confirmação.
    """
    logger.info("Baixando o arquivo do Google Drive com gdown...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    import gdown
    gdown.download(url, OUTPUT_ZIP, quiet=False)

    # Verifica se o arquivo baixado é realmente um ZIP
    if not is_valid_zip(OUTPUT_ZIP):
        logger.error("Erro: O arquivo baixado não é um ZIP válido.")
        raise Exception("Download falhou. O arquivo baixado não é um ZIP.")

    logger.info(f"Download concluído com sucesso: {OUTPUT_ZIP}")

def extract_file():
    """
    Extrai o arquivo ZIP baixado.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    logger.info(f"Extraindo {OUTPUT_ZIP} para {DATA_DIR}...")

    try:
        with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            logger.info("Arquivos extraídos:")
            for file in zip_ref.namelist():
                logger.info(f" - {file}")
        logger.info("Extração concluída.")
    except zipfile.BadZipFile:
        logger.error("Erro: O arquivo baixado não é um ZIP válido.")
        raise

def cleanup():
    """
    Remove o arquivo ZIP após a extração.
    """
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
        logger.info(f"Arquivo {OUTPUT_ZIP} removido.")

if __name__ == "__main__":
    try:
        install_gdown()  # Instala gdown caso não esteja presente
        download_file()
        extract_file()
        cleanup()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise