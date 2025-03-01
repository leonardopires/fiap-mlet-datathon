# Usando uma imagem base PyTorch com CUDA 12.6, cuDNN 9 e Python 3.10
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Ignore arquivos desnecessários no build
COPY .dockerignore .

# Instale dependências adicionais do sistema (apenas libgomp1 necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie apenas requirements.txt e instale dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta 8000
EXPOSE 8000

# Inicie o aplicativo com Uvicorn
CMD ["python", "-c", "from src.api.main import APIServer; server = APIServer(); import uvicorn; uvicorn.run(server.app, host='0.0.0.0', port=8000, log_config=None)"]