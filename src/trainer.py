import logging
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import os

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
os.makedirs('data/cache', exist_ok=True)

class RecommendationModel(nn.Module):
    def __init__(self, user_embedding_dim, news_embedding_dim, hidden_dim=128):
        super(RecommendationModel, self).__init__()
        self.news_layer = nn.Linear(news_embedding_dim, hidden_dim)
        self.user_layer = nn.Linear(user_embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, user_emb, news_emb):
        user_out = self.user_layer(user_emb)
        news_out = self.news_layer(news_emb)
        combined = torch.cat((user_out, news_out), dim=1)
        output = self.output_layer(self.relu(combined))
        return output

class Trainer:
    def train(self, interacoes, noticias, user_profiles, validation_file):
        logger.info("Iniciando ajuste do modelo na GPU")
        start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        if device.type == "cuda":
            torch.cuda.init()
            logger.info("CUDA inicializado explicitamente")

        logger.info(f"Carregando dados de validação de {validation_file}")
        validacao = pd.read_csv(validation_file)
        elapsed = time.time() - start_time
        logger.info(f"Dados de validação carregados: {len(validacao)} registros em {elapsed:.2f} segundos")

        train_size = int(0.8 * len(validacao))
        train_df = validacao[:train_size]
        val_df = validacao[train_size:]
        logger.info(f"Dados divididos: {len(train_df)} para treino, {len(val_df)} para validação")

        logger.info(f"Preparando dados para treinamento com {len(validacao)} registros")
        start_time = time.time()

        news_column = 'history'
        if news_column not in validacao.columns:
            logger.error(f"Coluna '{news_column}' não encontrada em validacao. Colunas disponíveis: {validacao.columns}")
            raise KeyError(f"Coluna '{news_column}' não encontrada em {validation_file}")

        news_encoder = LabelEncoder()
        news_encoder.fit(noticias['page'])
        validacao['news_idx'] = news_encoder.transform(validacao[news_column])

        validacao = validacao[validacao['userId'].isin(user_profiles.keys())]
        logger.debug(f"Filtradas {len(validacao)} linhas válidas após verificação de usuários")

        logger.debug("Normalizando embeddings de notícias na CPU")
        news_embeddings_np = np.array(noticias['embedding'].tolist())
        news_embeddings_np = news_embeddings_np / np.linalg.norm(news_embeddings_np, axis=1, keepdims=True)
        news_embeddings = torch.tensor(news_embeddings_np, dtype=torch.float32)  # Mantém na CPU
        news_page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        elapsed = time.time() - start_time
        logger.debug(f"Embeddings de notícias carregados: {news_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Normalizando embeddings de usuários na CPU")
        user_ids = list(user_profiles.keys())
        user_embeddings_np = np.array(list(user_profiles.values()))
        user_embeddings_np = user_embeddings_np / np.linalg.norm(user_embeddings_np, axis=1, keepdims=True)
        user_embeddings = torch.tensor(user_embeddings_np, dtype=torch.float32)  # Mantém na CPU
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        elapsed = time.time() - start_time
        logger.debug(f"Embeddings de usuários carregados: {user_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Preparando índices de usuários e notícias com operações vetoriais")
        X_user_indices = torch.tensor(validacao['userId'].map(user_id_to_idx).values, dtype=torch.long)
        X_news_indices = torch.tensor(validacao[news_column].map(news_page_to_idx).values, dtype=torch.long)
        elapsed = time.time() - start_time
        logger.debug(f"Índices preparados: X_user={X_user_indices.shape}, X_news={X_news_indices.shape}. Elapsed: {elapsed:.2f} s")

        X_user = user_embeddings[X_user_indices]
        X_news = news_embeddings[X_news_indices]
        logger.debug(f"Tensores de usuários e notícias: X_user={X_user.shape}, X_news={X_news.shape}")

        logger.debug("Pré-carregando tensor de relevância na CPU")
        y = torch.tensor(validacao['relevance'].values, dtype=torch.float32).unsqueeze(1)
        elapsed = time.time() - start_time
        logger.debug(f"Tensor de relevância preparado: {y.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Criando TensorDataset")
        dataset = TensorDataset(X_user, X_news, y)
        elapsed = time.time() - start_time
        logger.debug(f"Dataset criado. Elapsed: {elapsed:.2f} s")

        logger.debug("Preparando dados de validação uma única vez")
        val_X_user_indices = torch.tensor(val_df['userId'].map(user_id_to_idx).values, dtype=torch.long)
        val_X_news_indices = torch.tensor(val_df[news_column].map(news_page_to_idx).values, dtype=torch.long)
        val_X_user = user_embeddings[val_X_user_indices]
        val_X_news = news_embeddings[val_X_news_indices]
        val_y = torch.tensor(val_df['relevance'].values, dtype=torch.float32).unsqueeze(1)
        elapsed = time.time() - start_time
        logger.debug(f"Dados de validação preparados: {val_X_user.shape}. Elapsed: {elapsed:.2f} s")

        batch_size = 512
        logger.debug(f"Inicializando DataLoader com batch_size={batch_size}")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if device.type == "cuda" else min(8, os.cpu_count() or 1),
            pin_memory=True
        )
        elapsed = time.time() - start_time
        logger.info(f"Dados preparados em {elapsed:.2f} segundos")

        user_embedding_dim = X_user.shape[1]
        news_embedding_dim = X_news.shape[1]
        model = RecommendationModel(user_embedding_dim, news_embedding_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        logger.info("Iniciando treinamento na GPU")
        start_time = time.time()
        num_epochs = 20
        scaler = torch.amp.GradScaler('cuda')
        best_val_loss = float('inf')
        patience = 5
        epochs_no_improve = 0

        for epoch in tqdm(range(num_epochs), desc="Treinando épocas", unit="epoch"):
            try:
                model.train()
                total_loss = 0.0
                num_samples = 0
                for batch_idx, (batch_X_user, batch_X_news, batch_y) in enumerate(
                        tqdm(dataloader, desc=f"Época {epoch + 1}/{num_epochs}", leave=False, unit="batch")):
                    batch_X_user = batch_X_user.to(device)
                    batch_X_news = batch_X_news.to(device)
                    batch_y = batch_y.to(device)

                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_X_user, batch_X_news)
                        loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item() * len(batch_X_user)
                    num_samples += len(batch_X_user)
                avg_loss = total_loss / num_samples
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}")

                model.eval()
                with torch.no_grad():
                    val_X_user = val_X_user.to(device)
                    val_X_news = val_X_news.to(device)
                    val_y = val_y.to(device)
                    val_outputs = model(val_X_user, val_X_news)
                    val_loss = criterion(val_outputs, val_y)
                logger.info(f"Época {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}, Perda Validação: {val_loss.item():.4f}")

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'data/cache/best_model.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info(f"Early stopping na época {epoch + 1} devido a {patience} épocas sem melhoria")
                        break

            except Exception as e:
                logger.error(f"Erro na época {epoch + 1}: {str(e)}")
                torch.save(model.state_dict(), 'data/cache/emergency_checkpoint.pth')
                raise

        elapsed = time.time() - start_time
        logger.info(f"Modelo treinado com sucesso em {elapsed:.2f} segundos")

        model.load_state_dict(torch.load('data/cache/best_model.pth'))
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
        popular_news = noticias.sort_values('issued', ascending=False).head(10)['page'].tolist()
        return popular_news