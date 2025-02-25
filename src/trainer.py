import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import os

# Configura o logger para exibir mensagens detalhadas
logger = logging.getLogger(__name__)


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
        self.sigmoid = nn.Sigmoid()  # Converte saída em probabilidade (0-1)

    def forward(self, user_emb, news_emb):
        """
        Faz a passagem direta (forward pass) do modelo.

        Args:
            user_emb (torch.Tensor): Embedding do usuário.
            news_emb (torch.Tensor): Embedding da notícia.

        Returns:
            torch.Tensor: Probabilidade de interação (0-1).
        """
        user_out = self.user_layer(user_emb)
        news_out = self.news_layer(news_emb)
        combined = torch.cat((user_out, news_out), dim=1)  # Concatena os embeddings processados
        output = self.output_layer(self.relu(combined))
        return self.sigmoid(output)


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

        # Carrega os dados de validação
        logger.info(f"Carregando dados de validação de {validation_file}")
        validacao = pd.read_csv(validation_file)
        elapsed = time.time() - start_time
        logger.info(f"Dados de validação carregados: {len(validacao)} registros em {elapsed:.2f} segundos")

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

        # Filtra apenas usuários com perfis existentes
        validacao = validacao[validacao['userId'].isin(user_profiles.keys())]
        logger.debug(f"Filtradas {len(validacao)} linhas válidas após verificação de usuários")

        # Prepara os tensores para treinamento
        # Converte a lista de embeddings para um único numpy.ndarray antes de criar o tensor
        user_embeddings = np.array([user_profiles[row['userId']] for _, row in validacao.iterrows()])
        X_user = torch.tensor(user_embeddings, dtype=torch.float32).to(device)
        X_news = torch.tensor([noticias[noticias['page'] == row[news_column]]['embedding'].iloc[0]
                               for _, row in validacao.iterrows()],
                              dtype=torch.float32).to(device)
        y = torch.tensor(validacao['relevance'].values, dtype=torch.float32).unsqueeze(1).to(device)

        elapsed = time.time() - start_time
        logger.info(f"Dados preparados em {elapsed:.2f} segundos")

        # Configura o modelo com as dimensões dos embeddings
        user_embedding_dim = X_user.shape[1]
        news_embedding_dim = X_news.shape[1]
        model = RecommendationModel(user_embedding_dim, news_embedding_dim).to(device)
        criterion = nn.BCELoss()  # Perda binária sem pesos de recência
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam

        # Treina o modelo na GPU
        logger.info("Iniciando treinamento na GPU")
        start_time = time.time()
        num_epochs = 50  # Número de épocas (ajustável)
        batch_size = 1024  # Tamanho do batch (ajustável)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for i in range(0, len(X_user), batch_size):
                batch_X_user = X_user[i:i + batch_size]
                batch_X_news = X_news[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                optimizer.zero_grad()  # Limpa gradientes
                outputs = model(batch_X_user, batch_X_news)  # Forward pass
                loss = criterion(outputs, batch_y)  # Calcula perda
                loss.backward()  # Backpropagation
                optimizer.step()  # Atualiza pesos
                total_loss += loss.item() * batch_X_user.size(0)

            avg_loss = total_loss / len(X_user)
            if (epoch + 1) % 10 == 0:  # Log a cada 10 épocas
                logger.info(f"Época {epoch + 1}/{num_epochs}, Perda Média: {avg_loss:.4f}")

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
