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
Se o comando acima apresentar um erro `running scripts is disabled on this system.` execute antes este comando para habilitar sua permissão.
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

#### Linux
```bash
chmod +x install.sh
./install.sh
```

Isso deverá cria o ambiente virtual, instalar as dependências e baixar os dados.

Se não houverem erros, pule para a etapa `3. Construa e Inicie o Docker Compose`.

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
A. Crie e ative o ambiente virtual:
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
B. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
C. Baixe os dados:
   ```bash
   python download_data.py
   ```

### 2.3. (Opcional) Configure o Suporte à GPU
Se você tem uma GPU NVIDIA (como a RTX 2080 SUPER), configure o CUDA para acelerar o treinamento. Escolha as instruções para seu sistema operacional:

#### Windows (usando Docker Desktop e WSL 2)
A. **Instale o Docker Desktop**: Baixe e instale o [Docker Desktop para Windows](https://www.docker.com/products/docker-desktop/). Certifique-se de habilitar o WSL 2 durante a instalação.
B. **Configure o WSL 2**: Instale o WSL 2 e uma distribuição Linux (ex.: Ubuntu):
   ```powershell
   wsl --install
   wsl --install -d Ubuntu-20.04
   ```
   Após instalar, defina o WSL 2 como padrão:
   ```powershell
   wsl --set-default-version 2
   ```
C. **Instale os drivers NVIDIA**: Baixe e instale os drivers mais recentes da NVIDIA para sua GPU (ex.: RTX 2080 SUPER) em [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx).
D. **Instale o CUDA Toolkit no WSL 2**: Abra o terminal do Ubuntu no WSL 2 e instale o CUDA:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```
3. **Configure o Docker Desktop para GPU**: No Docker Desktop, vá em Settings > Resources > WSL Integration, habilite a integração com sua distro WSL 2 (ex.: Ubuntu-20.04), e aplique. O `docker-compose.yml` já inclui suporte à GPU.

#### Linux
A. **Instale os drivers NVIDIA**: Instale os drivers para sua GPU via gerenciador de pacotes (ex.: Ubuntu):
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-535
   ```
   Verifique com `nvidia-smi`.
B. **Instale o CUDA Toolkit**: Baixe e instale o CUDA 12.1:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_linux.run
   sudo sh cuda_12.1.0_531.14_linux.run --silent --toolkit
   ```
C. **Instale o NVIDIA Container Toolkit**:
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
D. O `docker-compose.yml` já inclui suporte à GPU; não é necessário ajustar nada.

### 3. Construa e Inicie o Docker Compose
Construa a imagem Docker e inicie o serviço:
```bash
docker-compose build --no-cache
docker-compose up
```
- O script `recomendador.py` será executado automaticamente dentro do container.
- Ele gera `submission.csv` e inicia a API em `http://localhost:3000`.
- O volume `.:/app` monta todos os arquivos do projeto (código, dados, estáticos), e o parâmetro `--reload` no Uvicorn permite que alterações em `recomendador.py` sejam aplicadas sem rebuild; para mudanças em `static/index.html`, basta atualizar a página no navegador.

### 4. Teste a API
Com o container rodando, você pode testar a API de algumas formas:

#### Usando Swagger UI
Acesse a interface Swagger em `http://localhost:3000/docs` para testar a API interativamente:
1. Abra `http://localhost:3000/docs` no navegador.
2. Clique em `POST /predict`.
3. Clique em "Try it out".
4. Insira um `user_id` (ex.: `e25fbee3a42d45a2914f9b061df3386b2ded2d8cc1f3d4b901419051126488b9`) e clique em "Execute".

#### Usando o Microsite
Acesse o microsite em `http://localhost:3000/`, insira um `user_id` e clique em "Obter Recomendações" para ver as predições em uma interface simples.

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

## Diagramas

### Arquitetura
```mermaid
graph TD;
    A[Usuário] -->|http://localhost:3000| B[Frontend (React)];
    B -->|HTTP API| C[Backend (FastAPI)];
    subgraph Sistema Recomendador G1
        C --> D[PostgreSQL] |Dados Persistentes|;
        C --> E[GPU (CUDA)] |Treinamento/Predição|;
        C --> F[Cache (HDF5)] |Embeddings/Perfis/Modelo|;
    end
    G[Dados Brutos (CSV)] -->|Carregamento| C;
```

### Fluxo da aplicação
```mermaid
graph TD;
    A[Dados Brutos] --> B[DataLoader];
    B -->|Carrega e Processa| C[Preprocessor];
    C -->|Gera Embeddings| D[EmbeddingGenerator<br>(Sentence Transformers)];
    D -->|Salva Embeddings| E[CacheManager];
    C -->|Cria Perfis| F[UserProfileBuilder];
    F -->|Perfis com Embeddings| G[Trainer];
    G -->|Treina com BCEWithLogitsLoss| H[RecommendationModel<br>(Rede Neural)];
    G -->|Modelo Treinado| I[Predictor];
    J[APIServer] -->|/train<br>(Carrega ou Treina)| G;
    J -->|/predict| I;
    K[App.tsx<br>(Dark Mode UI)] -->|HTTP Requests| J;
    I -->|Top-10 Recomendações<br>(inclui 'date')| K;
```

### Hierarquia de classes

```mermaid
graph TD;
    subgraph Backend
        A[DataLoader];
        B[Preprocessor] --> C[EmbeddingGenerator<br>(Sentence Transformers)];
        B --> D[CacheManager];
        B --> E[UserProfileBuilder] --> F[EngagementCalculator];
        B --> G[ResourceLogger];
        H[Trainer] --> I[RecommendationModel<br>(Rede Neural)];
        J[Predictor];
        K[ModelManager] --> H;
        K --> J;
        L[APIServer] --> M[StateManager];
        L --> N[DataInitializer] --> A;
        L --> N --> B;
        L --> K;
    end

    subgraph Frontend
        O[App<br>(Dark Mode, Sidebar Inferior)];
    end

    L -->|API Calls<br>(/train, /predict)| O;
```

## Notas

- O primeiro `docker-compose up` pode levar tempo devido ao download e treinamento do modelo.
- Certifique-se de ter espaço suficiente em disco (~5-10 GB) para os dados descompactados.
- Para desenvolvimento, edite `recomendador.py` diretamente; as mudanças são refletidas no container graças aos volumes.

Se precisar de ajuda, abra uma issue no repositório!