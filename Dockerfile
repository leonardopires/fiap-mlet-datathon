FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o arquivo api_server.py e outros arquivos necessários
COPY src/ src/
COPY . .

# Exponha a porta 8000
EXPOSE 8000

# Inicie o aplicativo usando o script Python diretamente com Uvicorn, desativando a configuração padrão de log
CMD ["python", "-c", "from src.api.main import APIServer; server = APIServer(); import uvicorn; uvicorn.run(server.app, host='0.0.0.0', port=8000, log_config=None)"]