# Recomendador G1 - FIAP MLET Datathon

Este projeto implementa um sistema de recomendação de notícias do G1 usando Sentence Transformers e uma API FastAPI, empacotado com Docker Compose.

## Pré-requisitos

- **Python 3.9+**: Certifique-se de ter o Python instalado.
- **Docker e Docker Compose**: Necessários para rodar o projeto em containers.
- **Ambiente Windows**: Este guia assume que você está no Windows (PowerShell).

## Configuração do Projeto

Siga os passos abaixo para configurar e executar o projeto:

### 1. Clone o Repositório
Clone este repositório para sua máquina local:
```powershell
git clone https://github.com/leonardopires/fiap-mlet-datathon.git
cd fiap-mlet-datathon
```

### 2. Crie e Ative um Ambiente Virtual
Crie um ambiente virtual para isolar as dependências:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale as Dependências
Instale as bibliotecas necessárias listadas em `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 4. Baixe os Dados
Os dados do projeto são muito grandes para o GitHub e estão hospedados no Google Drive. Execute o script `download_data.py` para baixá-los e descompactá-los na pasta `data`:
```powershell
python download_data.py
```
Isso baixará um arquivo ZIP de [https://drive.google.com/file/d/13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr/view](https://drive.google.com/file/d/13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr/view) e extrairá seu conteúdo em `data`.

### 5. (Opcional) Configure o Suporte à GPU
Se você tem uma GPU NVIDIA (como a RTX 2080 SUPER), configure o CUDA para acelerar o treinamento:
1. Instale o [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) e [cuDNN](https://developer.nvidia.com/cudnn).
2. Instale o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):
   ```powershell
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | `
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | `
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   apt-get update
   apt-get install -y nvidia-container-toolkit
   nvidia-ctk runtime configure --runtime=docker
   systemctl restart docker
   ```
3. O `docker-compose.yml` já inclui suporte à GPU; não é necessário ajustar nada.

### 6. Construa e Inicie o Docker Compose
Construa a imagem Docker e inicie o serviço:
```powershell
docker-compose build
docker-compose up
```
- O script `recomendador.py` será executado automaticamente dentro do container.
- Ele gera `submission.csv` e inicia a API em `http://localhost:8000`.

### 7. Teste a API
Com o container rodando, teste a API com um comando curl:
```powershell
curl "http://localhost:8000/predict?user_id=e25fbee3a42d45a2914f9b061df3386b2ded2d8cc1f3d4b901419051126488b9"
```
Você verá uma resposta JSON com recomendações.

### 8. Pare o Serviço
Para parar o container:
```powershell
docker-compose down
```

## Estrutura do Projeto

- `data/`: Contém os arquivos de dados (baixados via script).
- `recomendador.py`: Código principal com pré-processamento, treinamento e API.
- `download_data.py`: Script para baixar e descompactar os dados.
- `Dockerfile`: Configuração da imagem Docker.
- `docker-compose.yml`: Configuração do Docker Compose.
- `requirements.txt`: Dependências do Python.

## Notas

- O primeiro `docker-compose up` pode levar tempo devido ao download e treinamento do modelo.
- Certifique-se de ter espaço suficiente em disco (~5-10 GB) para os dados descompactados.
- Para desenvolvimento, edite `recomendador.py` diretamente; as mudanças são refletidas no container graças aos volumes.

Se precisar de ajuda, abra uma issue no repositório!