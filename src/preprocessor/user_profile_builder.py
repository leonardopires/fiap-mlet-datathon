import logging
import torch
import pandas as pd
import numpy as np
from typing import Dict
import time
from .engagement_calculator import EngagementCalculator

logger = logging.getLogger(__name__)


class UserProfileBuilder:
    """
    Esta classe cria um "perfil" para cada usuário, combinando as notícias que ele viu com o quanto
    ele gostou delas, usando números (embeddings) para representar isso.
    """

    def __init__(self, device: torch.device):
        """
        Configura o construtor de perfis.

        Args:
            device (torch.device): Onde os cálculos serão feitos (GPU ou CPU).
        """
        self.device = device
        # Usa a classe que calcula engajamento e recência
        self.calculator = EngagementCalculator()

    def build_profiles(self, interacoes: pd.DataFrame, page_to_embedding: Dict[str, torch.Tensor]) -> Dict[
        str, np.ndarray]:
        """
        Cria perfis para cada usuário com base nas interações dele.

        Args:
            interacoes (pd.DataFrame): Tabela com as interações dos usuários (histórico, cliques, etc.).
            page_to_embedding (Dict): Um dicionário que mapeia cada notícia (página) para seu embedding.

        Returns:
            Dict[str, np.ndarray]: Um dicionário onde cada usuário tem seu perfil (um array de números).
        """
        start_time = time.time()
        logger.info("Iniciando construção dos perfis de usuário na GPU")
        user_profiles = {}  # Guarda os perfis de cada usuário
        total_interacoes = len(interacoes)
        batch_size = 1000  # Processa 1000 interações por vez para não sobrecarregar a memória

        # Divide as interações em grupos (batches)
        for i in range(0, total_interacoes, batch_size):
            batch_end = min(i + batch_size, total_interacoes)
            batch = interacoes.iloc[i:batch_end]  # Pega o grupo atual
            elapsed_so_far = time.time() - start_time
            logger.info(
                f"Processando batch {i}-{batch_end} de {total_interacoes} interações em {elapsed_so_far:.2f} segundos")

            # Para cada interação no grupo
            for _, row in batch.iterrows():
                user_id = row['userId']  # Pega o ID do usuário
                # Divide as listas de histórico em itens individuais
                hist = row['history'].split(', ')
                clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
                times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
                scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
                timestamps = [int(x) for x in row['timestampHistory'].split(', ')]

                max_ts = max(timestamps)  # Encontra o momento mais recente
                embeddings, weights = [], []  # Lista para armazenar embeddings e pesos

                # Para cada notícia no histórico do usuário
                for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
                    if h in page_to_embedding:  # Se a notícia tem embedding
                        emb = page_to_embedding[h]  # Pega o embedding da notícia
                        eng = self.calculator.calculate_engagement(c, t, s)  # Calcula o engajamento
                        rec = self.calculator.calculate_recency(ts, max_ts)  # Calcula a recência
                        embeddings.append(emb)  # Adiciona o embedding
                        weights.append(eng * rec * 0.5 + rec * 0.5)     # Reduz o peso do engajamento,
                                                                        # aumenta o peso da recência

                # Se temos embeddings, calcula o perfil do usuário
                if embeddings:
                    embeddings_tensor = torch.stack(embeddings)  # Junta todos os embeddings em um "pacote"
                    weights_tensor = torch.tensor(weights, device=self.device,
                                                  dtype=torch.float32)  # Transforma pesos em tensor
                    # Multiplica cada embedding pelo seu peso e soma tudo
                    weighted_sum = torch.sum(embeddings_tensor * weights_tensor.unsqueeze(1), dim=0)
                    total_weight = torch.sum(weights_tensor)
                    # Divide pela soma dos pesos para obter uma média ponderada
                    profile = weighted_sum / total_weight
                    profile = profile / torch.norm(profile)  # Normalizar
                    user_profiles[user_id] = profile.cpu().numpy()

        elapsed = time.time() - start_time
        logger.info(f"Construção dos perfis concluída em {elapsed:.2f} segundos")
        return user_profiles
