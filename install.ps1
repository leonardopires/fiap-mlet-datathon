# Configura o ambiente e instala as dependencias

# Fun��o para verificar se um comando est� dispon�vel
function Test-CommandExists {
    param ($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# 1. Verifica se o Python est� instalado
Write-Host "Verificando Python..."
if (-not (Test-CommandExists "python")) {
    Write-Host "Python n�o encontrado. Por favor, instale o Python 3.9+ antes de continuar."
    Write-Host "Voc� pode baix�-lo em: https://www.python.org/downloads/"
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
    Write-Host "Ambiente virtual j� existe em $venvPath."
}

# 3. Ativa o ambiente virtual
Write-Host "Ativando ambiente virtual..."
& "$venvPath\Scripts\Activate.ps1"
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "Erro ao ativar o ambiente virtual. Verifique o caminho $venvPath."
    exit 1
} else {
    Write-Host "Ambiente virtual ativado: $env:VIRTUAL_ENV"
}

# 4. Instala as depend�ncias
Write-Host "Instalando depend�ncias do requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Erro ao instalar depend�ncias. Verifique o requirements.txt ou sua conex�o."
    exit 1
} else {
    Write-Host "Depend�ncias instaladas com sucesso."
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
    Write-Host "Pasta $dataDir j� existe. Pulando o download."
}

Write-Host "Configura��o conclu�da! Para construir e rodar o Docker Compose, execute:"
Write-Host "docker-compose build"
Write-Host "docker-compose up"
Write-Host "Para mais detalhes, consulte o README.md."