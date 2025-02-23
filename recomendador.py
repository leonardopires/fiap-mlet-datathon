#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.linear_model import Ridge
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import glob
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração do dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Carrega modelo Sentence Transformers
logger.info("Carregando modelo Sentence Transformers...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)

# Variáveis globais para pré-carregamento
INTERACOES = None
NOTICIAS = None
USER_PROFILES = None
REGRESSOR = None

# Modelo Pydantic para entrada da API
class UserRequest(BaseModel):
    user_id: str

# Funções existentes
def calculate_engagement(clicks, time, scroll):
    return (clicks * 0.3) + (time / 1000 * 0.5) + (scroll * 0.2)

def calculate_recency(timestamp, max_timestamp):
    return np.exp(-(max_timestamp - timestamp) / (7 * 24 * 3600 * 1000))

def load_and_concat_files(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def preprocess_data():
    logger.info("Iniciando pré-processamento dos dados...")
    interacoes = load_and_concat_files('data/files/treino/treino_parte*.csv')
    noticias = load_and_concat_files('data/itens/itens/itens-parte*.csv')
    logger.info("Gerando embeddings para notícias...")
    noticias['embedding'] = model.encode(noticias['title'].tolist(),
                                        convert_to_tensor=True,
                                        device=device,
                                        show_progress_bar=True).cpu().numpy().tolist()
    user_profiles = {}
    logger.info("Processando histórico de interações...")
    for i, row in interacoes.iterrows():
        if i % 1000 == 0:
            logger.info(f"Processados {i} registros de interações...")
        user_id = row['userId']
        hist = row['history'].split(', ')
        clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
        times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
        scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
        timestamps = [int(x) for x in row['timestampHistory'].split(', ')]
        max_ts = max(timestamps)
        embeddings, weights = [], []
        for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
            if h in noticias['page'].values:
                emb = noticias[noticias['page'] == h]['embedding'].values[0]
                eng = calculate_engagement(c, t, s)
                rec = calculate_recency(ts, max_ts)
                embeddings.append(emb)
                weights.append(eng * rec)
        if embeddings:
            user_profiles[user_id] = np.average(embeddings, axis=0, weights=weights)
    logger.info("Pré-processamento concluído.")
    return interacoes, noticias, user_profiles

def train_model(interacoes, noticias, user_profiles, validacao_kaggle_file):
    logger.info(f"Carregando validação de {validacao_kaggle_file}...")
    validacao = pd.read_csv(validacao_kaggle_file)
    X, y = [], []
    logger.info("Preparando dados para treinamento...")
    for i, row in validacao.iterrows():
        if i % 1000 == 0:
            logger.info(f"Processados {i} registros de validação...")
        user_id = row['userId']
        news_id = row['history']
        if user_id in user_profiles and news_id in noticias['page'].values:
            user_emb = user_profiles[user_id]
            news_emb = noticias[noticias['page'] == news_id]['embedding'].values[0]
            X.append(np.concatenate([user_emb, news_emb]))
            y.append(row['relevance'])
    if X:
        logger.info("Treinando modelo Ridge...")
        regressor = Ridge(alpha=1.0)
        regressor.fit(X, y)
        joblib.dump(regressor, 'regressor.pkl')
        logger.info("Modelo treinado e salvo como 'regressor.pkl'")
    else:
        logger.warning("Dados insuficientes para treinamento.")
        regressor = None
    return regressor

def predict(user_id, interacoes, noticias, user_profiles, regressor, k=10):
    logger.info(f"Gerando predições para user_id: {user_id}")
    if user_id not in user_profiles:
        logger.info("Usuário não encontrado, usando recomendações populares...")
        popular = interacoes['history'].str.split(', ').explode().value_counts().head(k).index
        return popular.tolist()
    user_emb = user_profiles[user_id]
    candidates = noticias[~noticias['page'].isin(interacoes[interacoes['userId'] == user_id]['history'].str.split(', ').iloc[0])]
    X = [np.concatenate([user_emb, emb]) for emb in candidates['embedding']]
    scores = regressor.predict(X)
    top_indices = np.argsort(scores)[-k:][::-1]
    predictions = candidates.iloc[top_indices]['page'].tolist()
    logger.info(f"Predições geradas: {predictions}")
    return predictions

# Configuração da API com FastAPI
app = FastAPI(
    title="Recomendador G1",
    description="API para recomendação de notícias do G1 baseada em histórico de usuários.",
    version="1.0.0"
)

# Monta a pasta static para arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rota para o microsite
@app.get("/")
async def read_root():
    file_path = "static/index.html"
    if os.path.exists(file_path):
        logger.info(f"Servindo {file_path}")
        return FileResponse(file_path)
    else:
        logger.error(f"Arquivo {file_path} não encontrado")
        return {"detail": "Arquivo index.html não encontrado"}

@app.post("/predict", response_model=dict)
async def get_prediction(request: UserRequest):
    """Retorna as top 10 recomendações de notícias para um usuário."""
    logger.info(f"Requisição recebida para user_id: {request.user_id}")
    predictions = predict(request.user_id, INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR)
    return {"user_id": request.user_id, "acessos_futuros": predictions}

# Inicialização dos dados globais
if __name__ == "__main__":
    if not os.path.exists('data/validacao_kaggle.csv'):
        logger.info("Gerando validacao_kaggle.csv...")
        os.system("python data/convert_kaggle.py")
    INTERACOES, NOTICIAS, USER_PROFILES = preprocess_data()
    REGRESSOR = train_model(INTERACOES, NOTICIAS, USER_PROFILES, 'data/validacao_kaggle.csv')
    validacao = pd.read_csv('data/validacao.csv')
    submission = []
    logger.info("Gerando submissão para validação...")
    for user_id in validacao['userId'].unique():
        preds = predict(user_id, INTERACOES, NOTICIAS, USER_PROFILES, REGRESSOR)
        for pred in preds:
            submission.append([user_id, pred])
    submission_df = pd.DataFrame(submission, columns=['userId', 'acessos_futuros'])
    submission_df.to_csv('submission.csv', index=False)
    logger.info("Submissão salva em 'submission.csv'")
    uvicorn.run(app, host="0.0.0.0", port=8000)
