#!/usr/bin/env python
# coding: utf-8

import os
import requests
import zipfile

# URL do Google Drive (usando o ID do arquivo)
FILE_ID = "13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_ZIP = "data.zip"
DATA_DIR = "data"


def download_file():
    """Baixa o arquivo ZIP do Google Drive."""
    print("Baixando o arquivo do Google Drive...")
    response = requests.get(DOWNLOAD_URL, stream=True)

    # Para arquivos grandes no Google Drive, pode ser necessário confirmar o download
    if "confirm" in response.url:
        confirm_token = response.cookies.get("download_warning")
        DOWNLOAD_URL_CONFIRM = f"{DOWNLOAD_URL}&confirm={confirm_token}"
        response = requests.get(DOWNLOAD_URL_CONFIRM, stream=True)

    with open(OUTPUT_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download concluído: {OUTPUT_ZIP}")


def extract_file():
    """Extrai o arquivo ZIP para a pasta 'data'."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Extraindo {OUTPUT_ZIP} para {DATA_DIR}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extração concluída.")


def cleanup():
    """Remove o arquivo ZIP após a extração."""
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
        print(f"Arquivo {OUTPUT_ZIP} removido.")


if __name__ == "__main__":
    try:
        download_file()
        extract_file()
        cleanup()
    except Exception as e:
        print(f"Erro: {e}")
