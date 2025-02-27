#!/usr/bin/env python
# coding: utf-8

import os
import requests
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE_ID = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
OUTPUT_ZIP = "data.zip"
DATA_DIR = "data"

def get_confirm_token(response):
    """
    Retorna o token de confirmação (download_warning) que o Google Drive usa
    para permitir o download de arquivos grandes.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    """
    Salva o conteúdo do 'response' no arquivo de destino, em chunks.
    """
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_file():
    """
    Faz o download do arquivo (ZIP) do Google Drive, levando em conta o 'confirm token'
    caso o Drive exija confirmação para arquivos grandes.
    """
    logger.info("Baixando o arquivo do Google Drive...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # 1) Primeira requisição: pode retornar HTML com aviso sobre verificação de vírus
    response = session.get(URL, params={'id': FILE_ID}, stream=True)
    token = get_confirm_token(response)

    # 2) Se houver token, o Drive pede confirmação explícita
    if token:
        params = {'id': FILE_ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # 3) Salva o conteúdo verdadeiro (ZIP) no destino
    save_response_content(response, OUTPUT_ZIP)
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
