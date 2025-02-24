import logging
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import numpy as np
import time
import os  # Adicionado para manipulação de caminhos

# Configura o logger para mensagens detalhadas
logger = logging.getLogger(__name__)

class Trainer:
    def train(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, user_profiles: dict, validacao_file: str) -> Ridge:
        """
        Treina um modelo de regressão Ridge usando dados de validação e perfis de usuário.

        Args:
            interacoes (pd.DataFrame): Dados de interações dos usuários (não usado diretamente, mas mantido para consistência).
            noticias (pd.DataFrame): Dados das notícias com embeddings.
            user_profiles (dict): Perfis de usuário pré-processados (user_id -> embedding).
            validacao_file (str): Caminho do arquivo CSV de validação (ex.: data/validacao_kaggle.csv).

        Returns:
            Ridge: Modelo treinado ou None se falhar.
        """
        train_start_time = time.time()
        logger.info(f"Iniciando treinamento do modelo com arquivo de validação {validacao_file}")

        # Carrega o arquivo de validação
        load_start = time.time()
        logger.info(f"Carregando dados de validação de {validacao_file}")
        validacao = pd.read_csv(validacao_file)
        total_validacao = len(validacao)
        load_elapsed = time.time() - load_start
        logger.info(f"Dados de validação carregados: {total_validacao} registros em {load_elapsed:.2f} segundos")

        # Prepara os dados para treinamento
        prep_start = time.time()
        logger.info(f"Preparando dados para treinamento com {total_validacao} registros")
        # Filtra linhas inválidas (usuários ou notícias ausentes nos dados pré-processados)
        valid_users = validacao['userId'].isin(user_profiles.keys())
        valid_news = validacao['history'].isin(noticias['page'].values)
        valid_rows = valid_users & valid_news
        validacao = validacao[valid_rows]
        logger.debug(f"Filtradas {len(validacao)} linhas válidas após verificação de usuários e notícias")

        # Constrói os dados de entrada (X) e saída (y) para o modelo
        X = np.array([
            np.concatenate([user_profiles[row['userId']], noticias[noticias['page'] == row['history']]['embedding'].values[0]])
            for _, row in validacao.iterrows()
        ])
        y = validacao['relevance'].values
        prep_elapsed = time.time() - prep_start
        logger.info(f"Dados preparados: {len(X)} amostras para treinamento em {prep_elapsed:.2f} segundos")

        # Verifica se há dados suficientes para treinar
        if len(X) > 0:
            model_start = time.time()
            logger.info("Iniciando treinamento do modelo Ridge com sklearn")
            # Inicializa o modelo Ridge com hiperparâmetro alpha=1.0
            regressor = Ridge(alpha=1.0)
            # Treina o modelo com os dados preparados
            regressor.fit(X, y)
            train_elapsed = time.time() - model_start
            logger.info(f"Modelo Ridge treinado em {train_elapsed:.2f} segundos")

            # Salva o modelo treinado no disco, na pasta data/cache
            cache_dir = 'data/cache'
            os.makedirs(cache_dir, exist_ok=True)  # Cria o diretório se não existir
            model_path = os.path.join(cache_dir, 'regressor.pkl')
            save_start = time.time()
            logger.info(f"Salvando modelo treinado em {model_path}")
            joblib.dump(regressor, model_path)
            save_elapsed = time.time() - save_start
            total_elapsed = time.time() - train_start_time
            logger.info(f"Modelo salvo em {save_elapsed:.2f} segundos. Treinamento total concluído em {total_elapsed:.2f} segundos")
            return regressor
        else:
            logger.warning("Dados insuficientes para treinamento após filtragem")
            total_elapsed = time.time() - train_start_time
            logger.info(f"Treinamento abortado em {total_elapsed:.2f} segundos devido a dados insuficientes")
            return None