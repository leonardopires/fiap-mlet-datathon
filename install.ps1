# install.ps1
# Configura o ambiente e instala as dependências

# Função para verificar se um comando está disponível
function Test-CommandExists {
    param ($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# 1. Verifica se o Python está instalado
Write-Host "Verificando Python..."
if (-not (Test-CommandExists "python")) {
    Write-Host "Python não encontrado. Por favor, instale o Python 3.9+ antes de continuar."
    Write-Host "Você pode baixá-lo em: https://www.python.org/downloads/"
    exit 1
} else {
    $pythonVersion = python --version
    Write-Host "Python encontrado: $pythonVersion"
}

# 2. Cria o ambiente virtual
$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Criando ambiente virtual em $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Ambiente virtual já existe em $venvPath."
}

# 3. Ativa o ambiente virtual
Write-Host "Ativando ambiente virtual..."
& "$venvPath\Scripts\Activate.ps1"
if ($null -eq $env:VIRTUAL_ENV) {  # Corrigido: $null no lado esquerdo
    Write-Host "Erro ao ativar o ambiente virtual. Verifique o caminho $venvPath."
    exit 1
} else {
    Write-Host "Ambiente virtual ativado: $env:VIRTUAL_ENV"
}

# 4. Instala as dependências
Write-Host "Instalando dependências do requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro ao instalar dependências. Verifique o requirements.txt ou sua conexão."
    exit 1
} else {
    Write-Host "Dependências instaladas com sucesso."
}

# 5. Baixa e descompacta os dados
$dataDir = "data"
if (-not (Test-Path $dataDir)) {
    Write-Host "Baixando e descompactando os dados com download_data.py..."
    python download_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erro ao baixar ou descompactar os dados. Verifique o script download_data.py."
        exit 1
    } else {
        Write-Host "Dados baixados e descompactados em $dataDir."
    }
} else {
    Write-Host "Pasta $dataDir já existe. Pulando o download."
}

# 6. Gera validacao_kaggle.csv
$validacaoKaggle = "data/validacao_kaggle.csv"
if (-not (Test-Path $validacaoKaggle)) {
    Write-Host "Gerando $validacaoKaggle com data/convert_kaggle.py..."
    # Muda para o diretório data e executa
    Push-Location $dataDir
    python convert_kaggle.py
    $exitCode = $LASTEXITCODE
    Pop-Location
    if (($null -ne $exitCode -and $exitCode -ne 0) -or -not (Test-Path $validacaoKaggle)) {
        Write-Host "Erro ao gerar $validacaoKaggle. Verifique data/convert_kaggle.py e se data/validacao.csv existe."
        exit 1
    } else {
        Write-Host "Arquivo $validacaoKaggle gerado com sucesso."
    }
}

Write-Host "Configuração concluída! Para construir e rodar o Docker Compose, execute:"
Write-Host "docker-compose build"
Write-Host "docker-compose up"
Write-Host "Para mais detalhes, consulte o README.md."