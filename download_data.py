#!/usr/bin/env python
# coding: utf-8

import os
import requests
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_ID = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_ZIP = "data.zip"
DATA_DIR = "data"

def download_file():
    logger.info("Baixando o arquivo do Google Drive...")
    response = requests.get(DOWNLOAD_URL, stream=True)
    if "confirm" in response.url:
        confirm_token = response.cookies.get("download_warning")
        DOWNLOAD_URL_CONFIRM = f"{DOWNLOAD_URL}&confirm={confirm_token}"
        response = requests.get(DOWNLOAD_URL_CONFIRM, stream=True)
    with open(OUTPUT_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Download concluído: {OUTPUT_ZIP}")

def extract_file():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    logger.info(f"Extraindo {OUTPUT_ZIP} para {DATA_DIR}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        logger.info("Arquivos extraídos:")
        for file in zip_ref.namelist():
            logger.info(f" - {file}")
    logger.info("Extração concluída.")

def cleanup():
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
        logger.info(f"Arquivo {OUTPUT_ZIP} removido.")

if __name__ == "__main__":
    try:
        download_file()
        extract_file()
        cleanup()
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise