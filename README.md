# Recomendador G1 - FIAP MLET Datathon

Este projeto implementa um sistema de recomendação de notícias do G1 usando Sentence Transformers e uma API FastAPI, empacotado com Docker Compose.

## Pré-requisitos

- **Python 3.9+**: Certifique-se de ter o Python instalado.
- **Docker e Docker Compose**: Necessários para rodar o projeto em containers.
- **Ambiente Windows ou Linux**: Este guia suporta ambos (PowerShell para Windows, Bash para Linux).

## Configuração do Projeto

Siga os passos abaixo para configurar e executar o projeto:

### 1. Clone o Repositório
Clone este repositório para sua máquina local:
```powershell
git clone https://github.com/leonardopires/fiap-mlet-datathon.git
cd fiap-mlet-datathon
```

### 2. Execute o Script de Instalação
O script de instalação automatiza a configuração do ambiente. Escolha o script para seu sistema:


#### Windows
```powershell
.\install.ps1
```
Se o comando acime apresentar um erro `running scripts is disabled on this system.` execute antes este comando para habilitar sua permissão.
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

#### Linux
```bash
chmod +x install.sh
./install.sh
```

Isso deverá cria o ambiente virtual, instalar as dependências e baixa os dados.

Se não houverem erros, pule para a etapa Construa e `Inicie o Docker Compose` e siga  instruções exibidas para rodar o Docker Compose.

Se houverem erros, ou você desejar fazer manualmente, siga a partir dos itens opcionais a seguir.

### 2.1. (Alternativa Manual) Baixe os dados
> ⚠ **Atenção:** Se ao rodar o comando anterior, os dados não forem baixados corretamente, você pode ter recebido um erro conforme abaixo:
```bash
zipfile.BadZipFile: File is not a zip file
Erro ao baixar ou descompactar os dados. Verifique o script download_data.py.
```
Se isto ocorrer, ou você quiser fazer manualmente, pode baixá-los pelo link abaixo e descompactar o conteúdo dentro da pasta `data`:
🔗 [Baixar os dados](https://drive.google.com/file/d/13rvnyK5PJADJQgYe-VbdXb7PpLPj7lPr/view)


### 2.2. (Alternativa Manual) Configure o Ambiente
Se preferir configurar manualmente:
1. Crie e ative o ambiente virtual:
   **Windows**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   **Linux**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Baixe os dados:
   ```bash
   python download_data.py
   ```

### 2.3. (Opcional) Configure o Suporte à GPU
Se você tem uma GPU NVIDIA (como a RTX 2080 SUPER), configure o CUDA para acelerar o treinamento. Escolha as instruções para seu sistema operacional:

#### Windows (usando Docker Desktop e WSL 2)
1. **Instale o Docker Desktop**: Baixe e instale o [Docker Desktop para Windows](https://www.docker.com/products/docker-desktop/). Certifique-se de habilitar o WSL 2 durante a instalação.
2. **Configure o WSL 2**: Instale o WSL 2 e uma distribuição Linux (ex.: Ubuntu):
   ```powershell
   wsl --install
   wsl --install -d Ubuntu-20.04
   ```
   Após instalar, defina o WSL 2 como padrão:
   ```powershell
   wsl --set-default-version 2
   ```
3. **Instale os drivers NVIDIA**: Baixe e instale os drivers mais recentes da NVIDIA para sua GPU (ex.: RTX 2080 SUPER) em [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx).
4. **Instale o CUDA Toolkit no WSL 2**: Abra o terminal do Ubuntu no WSL 2 e instale o CUDA:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```
5. **Configure o Docker Desktop para GPU**: No Docker Desktop, vá em Settings > Resources > WSL Integration, habilite a integração com sua distro WSL 2 (ex.: Ubuntu-20.04), e aplique. O `docker-compose.yml` já inclui suporte à GPU.

#### Linux
1. **Instale os drivers NVIDIA**: Instale os drivers para sua GPU via gerenciador de pacotes (ex.: Ubuntu):
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-535
   ```
   Verifique com `nvidia-smi`.
2. **Instale o CUDA Toolkit**: Baixe e instale o CUDA 12.1:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_linux.run
   sudo sh cuda_12.1.0_531.14_linux.run --silent --toolkit
   ```
3. **Instale o NVIDIA Container Toolkit**:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | `
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | `
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
4. O `docker-compose.yml` já inclui suporte à GPU; não é necessário ajustar nada.

### 3. Construa e Inicie o Docker Compose
Construa a imagem Docker e inicie o serviço:
```bash
docker-compose up --build
```
- O script `recomendador.py` será executado automaticamente dentro do container.
- Ele gera `submission.csv` e inicia a API em `http://localhost:8000`.
- O volume `.:/app` monta todos os arquivos do projeto (código, dados, estáticos), e o parâmetro `--reload` no Uvicorn permite que alterações em `recomendador.py` sejam aplicadas sem rebuild; para mudanças em `static/index.html`, basta atualizar a página no navegador.

### 4. Teste a API
Com o container rodando, você pode testar a API de algumas formas:

#### Usando Swagger UI
Acesse a interface Swagger em `http://localhost:8000/docs` para testar a API interativamente:
1. Abra `http://localhost:8000/docs` no navegador.
2. Clique em `POST /predict`.
3. Clique em "Try it out".
4. Insira um `user_id` (ex.: `e25fbee3a42d45a2914f9b061df3386b2ded2d8cc1f3d4b901419051126488b9`) e clique em "Execute".

#### Usando o Microsite
Acesse o microsite em `http://localhost:8000/`, insira um `user_id` e clique em "Obter Recomendações" para ver as predições em uma interface simples.

#### Usando Curl
Teste via terminal:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"user_id": "e25fbee3a42d45a2914f9b061df3386b2ded2d8cc1f3d4b901419051126488b9"}'
```
Você verá uma resposta JSON com recomendações.

#### Depuração
Se a API travar, verifique os logs com:
```bash
docker-compose logs
```
Os logs mostram o progresso do pré-processamento e das requisições, ajudando a identificar problemas.

### 7. Pare o Serviço
Para parar o container:
```bash
docker-compose down
```

## Estrutura do Projeto

- `data/`: Contém os arquivos de dados (baixados via script).
- `recomendador.py`: Código principal com pré-processamento, treinamento e API.
- `download_data.py`: Script para baixar e descompactar os dados.
- `install.ps1`: Script PowerShell para configurar o ambiente automaticamente (Windows).
- `install.sh`: Script Bash para configurar o ambiente automaticamente (Linux).
- `Dockerfile`: Configuração da imagem Docker.
- `docker-compose.yml`: Configuração do Docker Compose.
- `requirements.txt`: Dependências do Python.

## Notas

- O primeiro `docker-compose up` pode levar tempo devido ao download e treinamento do modelo.
- Certifique-se de ter espaço suficiente em disco (~5-10 GB) para os dados descompactados.
- Para desenvolvimento, edite `recomendador.py` diretamente; as mudanças são refletidas no container graças aos volumes.

Se precisar de ajuda, abra uma issue no repositório!