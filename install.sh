#!/bin/bash

# Script de instalação para linux (Ubuntu)

# Função para verificar se um comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Verifica se o Python 3 está instalado
echo "Verificando Python..."
if ! command_exists python3; then
    echo "Python 3 não encontrado. Por favor, instale o Python 3.9+ antes de continuar."
    echo "No Ubuntu, use: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    echo "Python encontrado: $PYTHON_VERSION"
fi

# 2. Cria o ambiente virtual
VENV_PATH=".venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Criando ambiente virtual em $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "Ambiente virtual já existe em $VENV_PATH."
fi

# 3. Ativa o ambiente virtual
echo "Ativando ambiente virtual..."
source "$VENV_PATH/bin/activate"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Erro ao ativar o ambiente virtual. Verifique o caminho $VENV_PATH."
    exit 1
else
    echo "Ambiente virtual ativado: $VIRTUAL_ENV"
fi

# 4. Instala as dependências
echo "Instalando dependências do requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Erro ao instalar dependências. Verifique o requirements.txt ou sua conexão."
    exit 1
else
    echo "Dependências instaladas com sucesso."
fi

# 5. Baixa e descompacta os dados
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Baixando e descompactando os dados com download_data.py..."
    python download_data.py
    if [ $? -ne 0 ]; then
        echo "Erro ao baixar ou descompactar os dados. Verifique o script download_data.py."
        exit 1
    else
        echo "Dados baixados e descompactados em $DATA_DIR."
    fi
else
    echo "Pasta $DATA_DIR já existe. Pulando o download."
fi

# 6. Gera validacao_kaggle.csv
VALIDACAO_KAGGLE="data/validacao_kaggle.csv"
if [ ! -f "$VALIDACAO_KAGGLE" ]; then
    echo "Gerando $VALIDACAO_KAGGLE com data/convert_kaggle.py..."
    # Muda para o diretório data e executa
    cd "$DATA_DIR"
    python convert_kaggle.py
    EXIT_CODE=$?
    cd ..
    if [ "$EXIT_CODE" -ne 0 ] || [ ! -f "$VALIDACAO_KAGGLE" ]; then
        echo "Erro ao gerar $VALIDACAO_KAGGLE. Verifique data/convert_kaggle.py e se data/validacao.csv existe."
        exit 1
    else
        echo "Arquivo $VALIDACAO_KAGGLE gerado com sucesso."
    fi
fi

echo "Configuração concluída! Para construir e rodar o Docker Compose, execute:"
echo "docker-compose build"
echo "docker-compose up"
echo "Para mais detalhes, consulte o README.md."