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
from sentence_transformers import SentenceTransformer  # Importar para gerar embeddings das palavras-chave
from src.preprocessor.resource_logger import ResourceLogger  # Importar ResourceLogger

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
os.makedirs('data/cache', exist_ok=True)


class RecommendationModel(nn.Module):
    def __init__(self, user_embedding_dim, news_embedding_dim, hidden_dim1=256, hidden_dim2=128):
        super(RecommendationModel, self).__init__()
        # Camadas para embeddings de usuário
        self.user_layer1 = nn.Linear(user_embedding_dim, hidden_dim1)
        self.user_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Camadas para embeddings de notícia
        self.news_layer1 = nn.Linear(news_embedding_dim, hidden_dim1)
        self.news_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu = nn.ReLU()
        # Camada de saída
        self.output_layer = nn.Linear(hidden_dim2 * 2, 1)

    def forward(self, user_emb, news_emb):
        # Processar embeddings de usuário
        user_out = self.user_layer1(user_emb)
        user_out = self.relu(user_out)
        user_out = self.user_layer2(user_out)
        user_out = self.relu(user_out)
        # Processar embeddings de notícia
        news_out = self.news_layer1(news_emb)
        news_out = self.relu(news_out)
        news_out = self.news_layer2(news_out)
        news_out = self.relu(news_out)
        # Concatenar e gerar saída
        combined = torch.cat((user_out, news_out), dim=1)
        output = self.output_layer(combined)
        return output


class Trainer:
    def __init__(self):
        # Inicializa o ResourceLogger para monitoramento de recursos
        self.resource_logger = ResourceLogger()

    def train(self, interacoes, noticias, user_profiles, validation_file, subsample_frac=1.0):
        logger.info("Iniciando ajuste do modelo na GPU")
        start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        if device.type == "cuda":
            torch.cuda.init()
            logger.info("CUDA inicializado explicitamente")

        logger.info(f"Carregando dados de validação de {validation_file}")
        validacao = pd.read_csv(validation_file)
        if subsample_frac < 1.0:
            validacao = validacao.sample(frac=subsample_frac, random_state=42)
            logger.info(f"Subamostragem aplicada a validacao: {len(validacao)} registros restantes")
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
            logger.error(
                f"Coluna '{news_column}' não encontrada em validacao. Colunas disponíveis: {validacao.columns}")
            raise KeyError(f"Coluna '{news_column}' não encontrada em {validation_file}")

        # Log antes de processar validacao_history
        logger.info("Criando conjunto único de notícias a partir de validacao['history']")
        validacao_history = validacao[news_column].str.split(', ').explode().unique()
        all_pages = np.unique(np.concatenate((noticias['page'].values, validacao_history)))

        # Treinar o LabelEncoder com todos os identificadores possíveis (para todas as notícias)
        logger.info("Treinando LabelEncoder com todas as notícias")
        news_encoder = LabelEncoder()
        news_encoder.fit(all_pages)

        # Criar um mapeamento apenas para as notícias presentes em noticias['page']
        logger.info("Criando mapeamento de índices para notícias em noticias['page']")
        news_page_to_encoded_idx = {page: idx for idx, page in enumerate(noticias['page'].values)}

        # Transformar os identificadores em validacao['history']
        logger.info("Transformando identificadores em validacao['history']")
        noticias_set = set(noticias['page'].values)
        valid_pages = validacao[news_column].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
        valid_pages = valid_pages[valid_pages.isin(noticias_set)].groupby(level=0).first()
        validacao = validacao.loc[valid_pages.index]
        # Remapear os índices para corresponder ao intervalo de news_embeddings
        validacao['news_idx'] = valid_pages.map(news_page_to_encoded_idx)
        # Verificar se todos os índices são válidos
        validacao = validacao.dropna(subset=['news_idx'])  # Remove linhas com índices ausentes
        validacao['news_idx'] = validacao['news_idx'].astype(int)
        max_news_idx = len(noticias['page']) - 1
        valid_indices = (validacao['news_idx'] >= 0) & (validacao['news_idx'] <= max_news_idx)
        validacao = validacao[valid_indices]
        logger.info(f"Filtradas {len(validacao)} linhas válidas após verificação de notícias disponíveis")

        logger.info("Filtrando usuários presentes em user_profiles")
        validacao = validacao[validacao['userId'].isin(user_profiles.keys())]
        elapsed = time.time() - start_time
        logger.info(f"Filtradas {len(validacao)} linhas válidas após verificação de usuários em {elapsed:.2f} segundos")

        logger.info("Normalizando embeddings de notícias na GPU")
        news_embeddings = torch.tensor(noticias['embedding'].tolist(), dtype=torch.float32).to(device)
        news_embeddings = news_embeddings / torch.norm(news_embeddings, dim=1, keepdim=True)
        news_page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        elapsed = time.time() - start_time
        logger.info(f"Embeddings de notícias carregados: {news_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.info("Normalizando embeddings de usuários na GPU")
        user_ids = list(user_profiles.keys())
        user_embeddings = torch.tensor(np.array(list(user_profiles.values())), dtype=torch.float32).to(device)
        user_embeddings = user_embeddings / torch.norm(user_embeddings, dim=1, keepdim=True)
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        elapsed = time.time() - start_time
        logger.info(f"Embeddings de usuários carregados: {user_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.info("Preparando índices de usuários e notícias na GPU")
        X_user_indices = torch.tensor(validacao['userId'].map(user_id_to_idx).values, dtype=torch.long).to(device)
        X_news_indices = torch.tensor(validacao['news_idx'].values, dtype=torch.long).to(device)
        # Verificar limites dos índices
        logger.info(
            f"Verificando limites dos índices: X_user_indices max={X_user_indices.max().item()}, min={X_user_indices.min().item()}")
        logger.info(
            f"Verificando limites dos índices: X_news_indices max={X_news_indices.max().item()}, min={X_news_indices.min().item()}")
        if X_user_indices.max().item() >= len(user_embeddings):
            raise ValueError(
                f"Índice de usuário fora dos limites: max={X_user_indices.max().item()}, esperado < {len(user_embeddings)}")
        if X_news_indices.max().item() >= len(news_embeddings):
            raise ValueError(
                f"Índice de notícia fora dos limites: max={X_news_indices.max().item()}, esperado < {len(news_embeddings)}")
        elapsed = time.time() - start_time
        logger.info(
            f"Índices preparados: X_user={X_user_indices.shape}, X_news={X_news_indices.shape}. Elapsed: {elapsed:.2f} s")

        X_user = user_embeddings[X_user_indices]
        X_news = news_embeddings[X_news_indices]
        logger.info(f"Tensores de usuários e notícias: X_user={X_user.shape}, X_news={X_news.shape}")

        logger.info("Pré-carregando tensor de relevância na GPU")
        y = torch.tensor(validacao['relevance'].values, dtype=torch.float32).unsqueeze(1).to(device)
        elapsed = time.time() - start_time
        logger.info(f"Tensor de relevância preparado: {y.shape}. Elapsed: {elapsed:.2f} s")

        logger.info("Criando TensorDataset")
        dataset = TensorDataset(X_user, X_news, y)
        elapsed = time.time() - start_time
        logger.info(f"Dataset criado. Elapsed: {elapsed:.2f} s")

        logger.info("Preparando dados de validação uma única vez")
        val_X_user_indices = torch.tensor(val_df['userId'].map(user_id_to_idx).values, dtype=torch.long).to(device)
        val_X_news_indices = torch.tensor(val_df[news_column].map(news_page_to_idx).values, dtype=torch.long).to(device)
        val_X_user = user_embeddings[val_X_user_indices]
        val_X_news = news_embeddings[val_X_news_indices]
        val_y = torch.tensor(val_df['relevance'].values, dtype=torch.float32).unsqueeze(1).to(device)
        elapsed = time.time() - start_time
        logger.info(f"Dados de validação preparados: {val_X_user.shape}. Elapsed: {elapsed:.2f} s")

        batch_size = 512
        logger.info(f"Inicializando DataLoader com batch_size={batch_size}")
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

                    # Log do uso da GPU a cada batch usando ResourceLogger
                    if batch_idx % 10 == 0:  # Log a cada 10 batches para evitar sobrecarga
                        self.resource_logger.log_gpu_usage()

                avg_loss = total_loss / num_samples
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}")

                model.eval()
                with torch.no_grad():
                    val_X_user = val_X_user.to(device)
                    val_X_news = val_X_news.to(device)
                    val_y = val_y.to(device)
                    val_outputs = model(val_X_user, val_X_news)
                    val_loss = criterion(val_outputs, val_y)
                logger.info(
                    f"Época {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}, Perda Validação: {val_loss.item():.4f}")

                # Log do uso da GPU ao final de cada época usando ResourceLogger
                self.resource_logger.log_gpu_usage()

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
            # Usar embeddings para lidar semanticamente com as palavras-chave
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)

            # Gerar embedding para as palavras-chave combinadas
            keywords_text = " ".join(keywords)
            with torch.no_grad():
                keyword_embedding = model.encode(keywords_text, convert_to_tensor=True, device=device)

            # Obter embeddings das notícias (já disponíveis em noticias['embedding'])
            news_embeddings = torch.tensor(noticias['embedding'].tolist(), dtype=torch.float32).to(device)

            # Calcular similaridade de cosseno entre as palavras-chave e as notícias
            keyword_embedding = keyword_embedding / torch.norm(keyword_embedding, dim=-1, keepdim=True)
            news_embeddings = news_embeddings / torch.norm(news_embeddings, dim=-1, keepdim=True)
            similarities = torch.mm(news_embeddings, keyword_embedding.unsqueeze(-1)).squeeze()

            # Obter os índices dos top 10 mais similares
            top_indices = torch.topk(similarities, k=10, largest=True).indices.cpu().numpy()
            popular_news = noticias.iloc[top_indices]['page'].tolist()
            logger.info(f"Encontradas {len(popular_news)} notícias relevantes para palavras-chave")
            return popular_news

        logger.info("Nenhuma palavra-chave fornecida ou resultados encontrados; usando popularidade")
        popular_news = noticias.sort_values('issued', ascending=False).head(10)['page'].tolist()
        return popular_news
