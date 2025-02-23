#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.linear_model import Ridge
import joblib
from fastapi import FastAPI
import uvicorn
import os
import glob

# Configuração do dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carrega modelo Sentence Transformers
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)


# Função para calcular score de engajamento
def calculate_engagement(clicks, time, scroll):
    return (clicks * 0.3) + (time / 1000 * 0.5) + (scroll * 0.2)


# Função para calcular peso de recência (decaimento de 7 dias)
def calculate_recency(timestamp, max_timestamp):
    return np.exp(-(max_timestamp - timestamp) / (7 * 24 * 3600 * 1000))


# Função para carregar e consolidar arquivos
def load_and_concat_files(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# Pré-processamento e geração de embeddings
def preprocess_data():
    # Carrega e consolida dados de treino e itens
    interacoes = load_and_concat_files('data/files/treino/treino_parte*.csv')
    noticias = load_and_concat_files('data/itens/itens/itens-parte*.csv')

    # Gera embeddings para notícias
    noticias['embedding'] = model.encode(noticias['title'].tolist(),
                                         convert_to_tensor=True,
                                         device=device,
                                         show_progress_bar=True).cpu().numpy().tolist()

    # Processa histórico de interações
    user_profiles = {}
    for _, row in interacoes.iterrows():
        user_id = row['userId']
        hist = row['history'].split(', ')
        clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
        times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
        scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
        timestamps = [int(x) for x in row['timestampHistory'].split(', ')]

        max_ts = max(timestamps)
        embeddings = []
        weights = []
        for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
            if h in noticias['page'].values:
                emb = noticias[noticias['page'] == h]['embedding'].values[0]
                eng = calculate_engagement(c, t, s)
                rec = calculate_recency(ts, max_ts)
                embeddings.append(emb)
                weights.append(eng * rec)

        if embeddings:
            user_profiles[user_id] = np.average(embeddings, axis=0, weights=weights)

    return interacoes, noticias, user_profiles


# Treinamento do modelo de ranking
def train_model(interacoes, noticias, user_profiles, validacao_kaggle_file):
    validacao = pd.read_csv(validacao_kaggle_file)
    X, y = [], []
    for _, row in validacao.iterrows():
        user_id = row['userId']
        news_id = row['history']
        if user_id in user_profiles and news_id in noticias['page'].values:
            user_emb = user_profiles[user_id]
            news_emb = noticias[noticias['page'] == news_id]['embedding'].values[0]
            X.append(np.concatenate([user_emb, news_emb]))
            y.append(row['relevance'])

    if X:
        regressor = Ridge(alpha=1.0)
        regressor.fit(X, y)
        joblib.dump(regressor, 'regressor.pkl')
        print("Modelo treinado e salvo como 'regressor.pkl'")
    else:
        print("Dados insuficientes para treinamento.")
        regressor = None

    return regressor


# Função de predição
def predict(user_id, interacoes, noticias, user_profiles, regressor, k=10):
    if user_id not in user_profiles:
        # Cold-start: recomenda notícias populares
        popular = interacoes['history'].str.split(', ').explode().value_counts().head(k).index
        return popular.tolist()

    user_emb = user_profiles[user_id]
    candidates = noticias[
        ~noticias['page'].isin(interacoes[interacoes['userId'] == user_id]['history'].str.split(', ').iloc[0])]
    X = [np.concatenate([user_emb, emb]) for emb in candidates['embedding']]
    scores = regressor.predict(X)
    top_indices = np.argsort(scores)[-k:][::-1]
    return candidates.iloc[top_indices]['page'].tolist()


# Configuração da API
app = FastAPI(title="Recomendador G1")


@app.get("/predict")
def get_prediction(user_id: str):
    interacoes, noticias, user_profiles = preprocess_data()
    regressor = joblib.load('regressor.pkl')
    predictions = predict(user_id, interacoes, noticias, user_profiles, regressor)
    return {"user_id": user_id, "acessos_futuros": predictions}


# Função principal
if __name__ == "__main__":
    # Gera validacao_kaggle.csv se não existir
    if not os.path.exists('data/validacao_kaggle.csv'):
        print("Gerando validacao_kaggle.csv...")
        os.system("python data/convert_kaggle.py")

    # Pré-processamento e treinamento
    interacoes, noticias, user_profiles = preprocess_data()
    regressor = train_model(interacoes, noticias, user_profiles, 'data/validacao_kaggle.csv')

    # Gera submissão para validação
    validacao = pd.read_csv('data/validacao.csv')
    submission = []
    for user_id in validacao['userId'].unique():
        preds = predict(user_id, interacoes, noticias, user_profiles, regressor)
        for pred in preds:
            submission.append([user_id, pred])

    submission_df = pd.DataFrame(submission, columns=['userId', 'acessos_futuros'])
    submission_df.to_csv('submission.csv', index=False)
    print("Submissão salva em 'submission.csv'")

    # Inicia a API
    uvicorn.run(app, host="0.0.0.0", port=8000)