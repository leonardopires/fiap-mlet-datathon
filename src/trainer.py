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
logger = logging.getLogger(__name__)

# Ativa otimização para operações na GPU
torch.backends.cudnn.benchmark = True

# Se a pasta data/cache não existir, cria ela recursivamente
os.makedirs('data/cache', exist_ok=True)


class RecommendationModel(nn.Module):
    """
    Modelo de recomendação neural que combina embeddings de usuários e notícias.
    """

    def __init__(self, user_embedding_dim, news_embedding_dim, hidden_dim=128):
        """
        Inicializa o modelo com camadas densas para processar embeddings.

        Args:
            user_embedding_dim (int): Dimensão dos embeddings de usuário.
            news_embedding_dim (int): Dimensão dos embeddings de notícias.
            hidden_dim (int): Dimensão da camada oculta (padrão: 128).
        """
        super(RecommendationModel, self).__init__()
        self.news_layer = nn.Linear(news_embedding_dim, hidden_dim)  # Reduz embeddings de notícias
        self.user_layer = nn.Linear(user_embedding_dim, hidden_dim)  # Reduz embeddings de usuário
        self.relu = nn.ReLU()  # Função de ativação
        self.output_layer = nn.Linear(hidden_dim * 2, 1)  # Camada de saída para predição

    def forward(self, user_emb, news_emb):
        """
        Faz a passagem direta (forward pass) do modelo.

        Args:
            user_emb (torch.Tensor): Embedding do usuário.
            news_emb (torch.Tensor): Embedding da notícia.

        Returns:
            torch.Tensor: Logits (valores brutos) que serão transformados em probabilidade no BCEWithLogitsLoss.
        """
        user_out = self.user_layer(user_emb)
        news_out = self.news_layer(news_emb)
        combined = torch.cat((user_out, news_out), dim=1)  # Concatena os embeddings processados
        output = self.output_layer(self.relu(combined))
        return output  # Retorna logits, sem Sigmoid!


class Trainer:
    def train(self, interacoes, noticias, user_profiles, validation_file):
        """
        Treina o modelo de recomendação usando GPU com PyTorch.

        Args:
            interacoes (pd.DataFrame): Dados de interações dos usuários.
            noticias (pd.DataFrame): Dados das notícias com embeddings.
            user_profiles (dict): Perfis dos usuários como embeddings.
            validation_file (str): Caminho do arquivo de validação.

        Returns:
            object: Modelo treinado.
        """
        logger.info("Iniciando ajuste do modelo na GPU")
        start_time = time.time()

        # Define o dispositivo (GPU se disponível, senão CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")
        if device.type == "cuda":
            torch.cuda.init()  # Forçar inicialização explícita
            logger.info("CUDA inicializado explicitamente")

        # Carrega os dados de validação
        logger.info(f"Carregando dados de validação de {validation_file}")
        validacao = pd.read_csv(validation_file)
        elapsed = time.time() - start_time
        logger.info(f"Dados de validação carregados: {len(validacao)} registros em {elapsed:.2f} segundos")
        train_size = int(0.8 * len(validacao))
        train_df = validacao[:train_size]
        val_df = validacao[train_size:]
        logger.info(f"Dados divididos: {len(train_df)} para treino, {len(val_df)} para validação")

        # Prepara os dados para treinamento
        logger.info(f"Preparando dados para treinamento com {len(validacao)} registros")
        start_time = time.time()

        # Usa 'history' como a coluna de notícias, conforme validacao_kaggle.csv
        news_column = 'history'
        if news_column not in validacao.columns:
            logger.error(
                f"Coluna '{news_column}' não encontrada em validacao. Colunas disponíveis: {validacao.columns}")
            raise KeyError(f"Coluna '{news_column}' não encontrada em {validation_file}")

        # Codifica os IDs das notícias para índices numéricos
        news_encoder = LabelEncoder()
        news_encoder.fit(noticias['page'])
        validacao['news_idx'] = news_encoder.transform(validacao[news_column])

        # Filtra apenas usuários com perfis existentes na CPU
        elapsed = time.time() - start_time
        logger.debug(f"Verificando usuários com perfis existentes. Elapsed: {elapsed:.2f} s")
        validacao = validacao[validacao['userId'].isin(user_profiles.keys())]
        logger.debug(f"Filtradas {len(validacao)} linhas válidas após verificação de usuários")

        # Pré-normaliza os embeddings na CPU antes de movê-los para a GPU
        logger.debug("Normalizando embeddings de notícias na CPU")
        news_embeddings_np = np.array(noticias['embedding'].tolist())
        news_embeddings_np = news_embeddings_np / np.linalg.norm(news_embeddings_np, axis=1, keepdims=True)
        news_embeddings = torch.tensor(news_embeddings_np, dtype=torch.float32).to(device)
        news_page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        elapsed = time.time() - start_time
        logger.debug(f"Embeddings de notícias carregados: {news_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Normalizando embeddings de usuários na CPU")
        user_ids = list(user_profiles.keys())
        user_embeddings_np = np.array(list(user_profiles.values()))
        user_embeddings_np = user_embeddings_np / np.linalg.norm(user_embeddings_np, axis=1, keepdims=True)
        user_embeddings = torch.tensor(user_embeddings_np, dtype=torch.float32).to(device)
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        elapsed = time.time() - start_time
        logger.debug(f"Embeddings de usuários carregados: {user_embeddings.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Preparando índices de usuários e notícias com barra de progresso")
        X_user_indices = torch.tensor([user_id_to_idx[row['userId']] for _, row in
                                       tqdm(validacao.iterrows(), total=len(validacao), desc="Preparando X_user")],
                                      dtype=torch.long).to(device)
        elapsed = time.time() - start_time
        logger.debug(f"Índices de usuários preparados: {X_user_indices.shape}. Elapsed: {elapsed:.2f} s")

        X_news_indices = torch.tensor([news_page_to_idx[row[news_column]] for _, row in
                                       tqdm(validacao.iterrows(), total=len(validacao), desc="Preparando X_news")],
                                      dtype=torch.long).to(device)
        elapsed = time.time() - start_time
        logger.debug(f"Índices de notícias preparados: {X_news_indices.shape}. Elapsed: {elapsed:.2f} s")

        # Usa índices para selecionar embeddings diretamente na GPU e cria dataset
        X_user = user_embeddings[X_user_indices]  # Já normalizado
        elapsed = time.time() - start_time
        logger.debug(f"Tensores de usuários: {X_user.shape}. Elapsed: {elapsed:.2f} s")

        X_news = news_embeddings[X_news_indices]  # Já normalizado
        elapsed = time.time() - start_time
        logger.debug(f"Tensores de notícias: {X_news.shape}. Elapsed: {elapsed:.2f} s")

        logger.debug("Pré-carregando tensor de relevância na GPU")
        y = torch.tensor(validacao['relevance'].values, dtype=torch.float32, device=device).unsqueeze(1)
        elapsed = time.time() - start_time
        logger.debug(f"Tensor de relevância preparado: {y.shape}. Elapsed: {elapsed:.2f} s")

        # Prepara o dataset e dataloader para treinamento eficiente na GPU
        logger.debug("Criando TensorDataset")
        dataset = TensorDataset(X_user, X_news, y)
        elapsed = time.time() - start_time
        logger.debug(f"Dataset criado. Elapsed: {elapsed:.2f} s")

        # Preparar dados de validação uma única vez
        logger.debug("Preparando dados de validação uma única vez")
        val_X_user_indices = torch.tensor([user_id_to_idx[row['userId']] for _, row in val_df.iterrows()],
                                          dtype=torch.long).to(device)
        val_X_news_indices = torch.tensor([news_page_to_idx[row[news_column]] for _, row in val_df.iterrows()],
                                          dtype=torch.long).to(device)
        val_X_user = user_embeddings[val_X_user_indices]
        val_X_news = news_embeddings[val_X_news_indices]
        val_y = torch.tensor(val_df['relevance'].values, dtype=torch.float32, device=device).unsqueeze(1)
        elapsed = time.time() - start_time
        logger.debug(f"Dados de validação preparados: {val_X_user.shape}. Elapsed: {elapsed:.2f} s")

        # Calcular batch_size com base na VRAM disponível, sem limite superior fixo
        if device.type == "cuda":
            total_memory, used_memory = torch.cuda.mem_get_info()
            free_memory_mb = (total_memory - used_memory) / (1024 * 1024)  # Memória livre em MB
            target_memory_mb = free_memory_mb * 0.9  # Usar 90% da VRAM livre
            overhead_mb = 512  # Overhead de 512 MB para segurança
            # Estimar memória por item
            item_size_mb = (
                                   X_user.element_size() * X_user.nelement() +
                                   X_news.element_size() * X_news.nelement() +
                                   y.element_size() * y.nelement()
                           ) / (1024 * 1024 * len(dataset))
            max_batch_size = max(32, int((target_memory_mb - overhead_mb) / item_size_mb))  # Sem limite superior
            logger.info(
                f"VRAM total: {total_memory / (1024 * 1024 * 1024):.2f} GB, livre: {free_memory_mb / 1024:.2f} GB, "
                f"item size: {item_size_mb:.2f} MB, batch_size: {max_batch_size}"
            )
        else:
            max_batch_size = 512
        batch_size = max_batch_size

        logger.debug(f"Inicializando DataLoader com batch_size={batch_size}")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if device.type == "cuda" else min(8, os.cpu_count() or 1),
            pin_memory=False  # Mantido como False devido a problemas anteriores
        )
        elapsed = time.time() - start_time
        logger.debug(f"DataLoader inicializado. Elapsed: {elapsed:.2f} s")

        elapsed = time.time() - start_time
        logger.info(f"Dados preparados em {elapsed:.2f} segundos")

        # Configura o modelo com as dimensões dos embeddings
        user_embedding_dim = X_user.shape[1]
        news_embedding_dim = X_news.shape[1]
        model = RecommendationModel(user_embedding_dim, news_embedding_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Perda binária com logits, segura para autocast
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Otimizador Adam com regularização L2

        # Treina o modelo na GPU com precisão mista e barra de progresso
        logger.info("Iniciando treinamento na GPU")
        start_time = time.time()
        num_epochs = 100  # Número de épocas (ajustável)
        scaler = torch.amp.GradScaler('cuda')  # Atualizado para nova API do PyTorch

        for epoch in tqdm(range(num_epochs), desc="Treinando épocas", unit="epoch"):
            try:
                model.train()
                total_loss = 0.0
                num_samples = 0
                for batch_idx, (batch_X_user, batch_X_news, batch_y) in enumerate(
                        tqdm(dataloader, desc=f"Época {epoch + 1}/{num_epochs}", leave=False, unit="batch")):
                    # Usa precisão mista para acelerar o treinamento
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_X_user, batch_X_news)  # Forward pass
                        loss = criterion(outputs, batch_y)  # Calcula perda

                    # Backpropagation com escalonamento de gradientes
                    optimizer.zero_grad()  # Limpa gradientes
                    scaler.scale(loss).backward()  # Backpropagation com precisão mista
                    scaler.step(optimizer)  # Atualiza pesos
                    scaler.update()  # Atualiza o scaler para próxima iteração
                    total_loss += loss * len(batch_X_user)  # Acumula perda na GPU
                    num_samples += len(batch_X_user)
                avg_loss = (total_loss / num_samples)  # Sincroniza apenas no final
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}")

                # Validação a cada 10 épocas
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(val_X_user, val_X_news)
                        val_loss = criterion(val_outputs, val_y)
                    logger.info(
                        f"Época {epoch + 1}/{num_epochs}, Perda Treino: {avg_loss:.4f}, Perda Validação: {val_loss.item():.4f}")
                model.train()

            except Exception as e:
                logger.error(f"Erro na época {epoch + 1}: {str(e)}")
                torch.save(model.state_dict(), 'data/cache/emergency_checkpoint.pth')
                raise

        elapsed = time.time() - start_time
        logger.info(f"Modelo treinado com sucesso em {elapsed:.2f} segundos")

        # Move o modelo para CPU para salvamento
        model = model.cpu()
        return model

    def handle_cold_start(self, noticias, keywords=None):
        """
        Gera recomendações cold-start baseadas em popularidade ou palavras-chave.

        Args:
            noticias (pd.DataFrame): Dados das notícias.
            keywords (List[str], opcional): Palavras-chave fornecidas pelo usuário.

        Returns:
            list: Lista de IDs de notícias recomendadas.
        """
        logger.info("Gerando recomendações cold-start")
        if keywords:
            logger.info(f"Filtrando notícias com palavras-chave: {keywords}")
            # Filtra notícias que contêm pelo menos uma palavra-chave no título ou corpo
            mask = noticias['title'].str.contains('|'.join(keywords), case=False, na=False) | \
                   noticias['body'].str.contains('|'.join(keywords), case=False, na=False)
            filtered_news = noticias[mask]
            if len(filtered_news) > 0:
                # Ordena por recência entre as filtradas
                popular_news = filtered_news.sort_values('issued', ascending=False).head(10)['page'].tolist()
                logger.info(f"Encontradas {len(popular_news)} notícias relevantes para palavras-chave")
                return popular_news

        # Fallback: notícias populares se não houver palavras-chave ou resultados
        logger.info("Nenhuma palavra-chave fornecida ou resultados encontrados; usando popularidade")
        popular_news = noticias.sort_values('issued', ascending=False).head(10)['page'].tolist()
        return popular_news
