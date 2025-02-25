import logging
import pandas as pd
from tqdm import tqdm  # Para barras de progresso
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import os

# Configura o logger para exibir mensagens detalhadas
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ativa otimização para CNNs na GPU
torch.backends.cudnn.benchmark = True

class RecommendationModel(nn.Module):
    def __init__(self, user_embedding_dim, news_embedding_dim, hidden_dim=128):
        super(RecommendationModel, self).__init__()
        self.news_layer = nn.Linear(news_embedding_dim, hidden_dim)
        self.user_layer = nn.Linear(user_embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_emb, news_emb):
        user_out = self.user_layer(user_emb)
        news_out = self.news_layer(news_emb)
        combined = torch.cat((user_out, news_out), dim=1)
        output = self.output_layer(self.relu(combined))
        return self.sigmoid(output)

class Trainer:
    def train(self, interacoes, noticias, user_profiles, validation_file):
        logger.info("Iniciando ajuste do modelo na GPU")
        start_time = time.time()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        
        try:
            validacao = pd.read_csv(validation_file)
        except Exception as e:
            logger.error(f"Erro ao carregar o arquivo de validação: {e}")
            return None
        
        if validacao.empty:
            logger.error("Arquivo de validação está vazio.")
            return None
        
        news_column = 'history'
        if news_column not in validacao.columns:
            logger.error(f"Coluna '{news_column}' não encontrada em {validation_file}. Colunas disponíveis: {validacao.columns}")
            return None
        
        if 'relevance' not in validacao.columns:
            logger.error(f"Coluna 'relevance' não encontrada em {validation_file}. Colunas disponíveis: {validacao.columns}")
            return None
        
        if 'page' not in noticias.columns:
            logger.error("Coluna 'page' não encontrada em notícias. Verifique o dataset.")
            return None
        
        news_encoder = LabelEncoder()
        news_encoder.fit(noticias['page'])
        validacao['news_idx'] = news_encoder.transform(validacao[news_column])
        
        validacao = validacao[validacao['userId'].isin(user_profiles.keys())]
        
        news_embeddings = torch.tensor(np.array(noticias['embedding'].tolist()), dtype=torch.float32).to(device)
        user_embeddings = torch.tensor(np.array(list(user_profiles.values())), dtype=torch.float32).to(device)
        
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_profiles.keys())}
        news_page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        
        X_user_indices = torch.tensor([user_id_to_idx[row['userId']] for _, row in validacao.iterrows()], dtype=torch.long).to(device)
        X_news_indices = torch.tensor([news_page_to_idx[row[news_column]] for _, row in validacao.iterrows()], dtype=torch.long).to(device)
        
        X_user = user_embeddings[X_user_indices]
        X_news = news_embeddings[X_news_indices]
        y = torch.tensor(validacao['relevance'].values, dtype=torch.float32, device=device).unsqueeze(1)
        
        dataset = TensorDataset(X_user, X_news, y)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True if device.type == "cuda" else False)
        
        model = RecommendationModel(X_user.shape[1], X_news.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        logger.info("Iniciando treinamento na GPU")
        scaler = torch.amp.GradScaler()
        
        for epoch in range(50):
            model.train()
            total_loss = 0
            for batch_X_user, batch_X_news, batch_y in dataloader:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch_X_user, batch_X_news)
                    loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * batch_X_user.size(0)
            
            avg_loss = total_loss / len(X_user)
            logger.info(f"Época {epoch + 1}/50, Perda Média: {avg_loss:.4f}")
        
        model = model.cpu()
        return model
    
    def handle_cold_start(self, noticias, keywords=None):
        logger.info("Gerando recomendações cold-start")
        if keywords:
            logger.info(f"Filtrando notícias com palavras-chave: {keywords}")
            mask = noticias['title'].str.contains('|'.join(keywords), case=False, na=False) | \
                   noticias['body'].str.contains('|'.join(keywords), case=False, na=False)
            filtered_news = noticias[mask]
            if len(filtered_news) > 0:
                popular_news = filtered_news.sort_values('issued', ascending=False).head(10)['page'].tolist()
                logger.info(f"Encontradas {len(popular_news)} notícias relevantes para palavras-chave")
                return popular_news
        logger.info("Nenhuma palavra-chave fornecida ou resultados encontrados; usando popularidade")
        return noticias.sort_values('issued', ascending=False).head(10)['page'].tolist()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(
        interacoes=pd.DataFrame(),
        noticias=pd.DataFrame(),
        user_profiles={},
        validation_file="validacao.csv"
    )
