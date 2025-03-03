import logging
from typing import Optional, List
from fastapi import HTTPException
from src.trainer import Trainer
from src.predictor import Predictor
from .state_manager import StateManager
import time
import joblib
import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.preprocessor.engagement_calculator import EngagementCalculator
from src.preprocessor.cache_manager import CacheManager
import psycopg2
from psycopg2.extras import Json
from datetime import datetime, timedelta
import hashlib  # Para gerar hash das keywords

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, trainer: Trainer, predictor_class: type):
        self.trainer = trainer
        self.predictor_class = predictor_class
        self.cache_dir = 'data/cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_manager = CacheManager()
        self.engagement_calculator = EngagementCalculator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Inicializa a conexão com o PostgreSQL
        self.db_connection = None
        self._initialize_db_connection()

    def _initialize_db_connection(self):
        """Inicializa a conexão com o PostgreSQL."""
        try:
            self.db_connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "postgres"),
                port=int(os.getenv("POSTGRES_PORT", 5432)),
                dbname=os.getenv("POSTGRES_DB", "recomendador_db"),
                user=os.getenv("POSTGRES_USER", "recomendador"),
                password=os.getenv("POSTGRES_PASSWORD", "senha123")
            )
            logger.info("Conexão com PostgreSQL para cache de predições estabelecida")
        except psycopg2.Error as e:
            logger.error(f"Falha ao conectar ao PostgreSQL para cache de predições: {e}")
            raise

    def _calculate_and_cache_metrics(self, interacoes: pd.DataFrame, noticias: pd.DataFrame) -> None:
        """Calcula e armazena recency_weights e global_engagement em cache."""
        # Cache para recency_weights (salva time_diff_days para ajuste posterior)
        recency_cache_file = os.path.join(self.cache_dir, 'recency_base.h5')
        logger.info("Calculando base para pesos de recência")
        start_time = time.time()
        issued_dates = pd.to_datetime(noticias['issued'], errors='coerce').dt.tz_localize(None)
        current_time = pd.Timestamp.now(tz=None).tz_localize(None)
        time_diff_days = np.array([(current_time - date).days if pd.notna(date) else np.nan for date in issued_dates],
                                  dtype=np.float32)
        self.cache_manager.save_array(recency_cache_file, time_diff_days)
        elapsed = time.time() - start_time
        logger.info(f"Base de recência salva em {recency_cache_file} em {elapsed:.2f} segundos")

        # Cache para global_engagement
        engagement_cache_file = os.path.join(self.cache_dir, 'global_engagement.h5')
        logger.info("Calculando engajamento global para notícias")
        start_time = time.time()
        global_engagement = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
        interaction_counts = torch.zeros(len(noticias), dtype=torch.float32).to(self.device)
        page_to_idx = {page: idx for idx, page in enumerate(noticias['page'])}
        logger.debug(f"Mapeamento page_to_idx criado com {len(page_to_idx)} entradas")

        # Pré-processar dados para acumulação na GPU
        indices = []
        engagements = []
        for _, row in tqdm(interacoes.iterrows(), total=len(interacoes),
                           desc="Pré-processando interações para engajamento global", leave=False):
            hist = row['history'].split(', ')
            clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
            times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
            scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
            for h, c, t, s in zip(hist, clicks, times, scrolls):
                if h in page_to_idx:
                    idx = page_to_idx[h]
                    engagement = self.engagement_calculator.calculate_engagement(c, t, s)
                    indices.append(idx)
                    engagements.append(engagement)
        # Acumular na GPU usando scatter_add_
        indices = torch.tensor(indices, dtype=torch.long).to(self.device)
        engagements = torch.tensor(engagements, dtype=torch.float32).to(self.device)
        global_engagement.scatter_add_(0, indices, engagements)
        interaction_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
        logger.info("Engajamento global calculado")
        # Evitar divisão por zero e calcular a média
        logger.info("Normalizando engajamento global")
        global_engagement = torch.where(interaction_counts > 0, global_engagement / interaction_counts,
                                        torch.tensor(0.0, device=self.device))
        logger.info(f"Global engagement normalizado: shape={global_engagement.shape}")
        # Normalizar o engajamento global para o intervalo [0, 1]
        max_global_engagement = torch.max(global_engagement)
        logger.info(f"Máximo de engajamento global: {max_global_engagement}")
        if max_global_engagement > 0:
            global_engagement = global_engagement / max_global_engagement
        self.cache_manager.save_array(engagement_cache_file, global_engagement.cpu().numpy())
        elapsed = time.time() - start_time
        logger.info(f"Engajamento global calculado e salvo em {engagement_cache_file} em {elapsed:.2f} segundos")

    def _generate_keywords_cache_key(self, keywords: Optional[List[str]]) -> str:
        """Gera uma chave de cache para recomendações baseadas em keywords."""
        if keywords is None or not keywords:
            return "default_cold_start"
        # Ordena as palavras-chave para consistência
        sorted_keywords = sorted(keywords)
        # Concatena as palavras-chave e gera um hash MD5
        keywords_str = ",".join(sorted_keywords)
        hash_object = hashlib.md5(keywords_str.encode('utf-8'))
        hash_value = hash_object.hexdigest()
        return hash_value

    def _get_cached_prediction(self, user_id: str) -> Optional[list[dict]]:
        """Verifica se existe uma predição válida no cache para o user_id."""
        try:
            with self.db_connection.cursor() as cursor:
                # Verifica se existe uma predição válida (menos de 1 hora)
                query = """
                    SELECT predictions
                    FROM predictions_cache
                    WHERE user_id = %s
                    AND timestamp >= %s
                """
                expiration_time = datetime.now() - timedelta(hours=1)
                cursor.execute(query, (user_id, expiration_time))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Predição em cache encontrada para user_id {user_id}")
                    return result[0]
                logger.debug(f"Nenhuma predição válida em cache para user_id {user_id}")
                return None
        except psycopg2.Error as e:
            logger.error(f"Erro ao consultar predição em cache: {e}")
            return None

    def _get_cached_keywords_recommendation(self, cache_key: str) -> Optional[list[dict]]:
        """Verifica se existe uma recomendação válida no cache para a chave baseada em keywords."""
        try:
            with self.db_connection.cursor() as cursor:
                # Verifica se existe uma recomendação válida (menos de 1 hora)
                query = """
                    SELECT recommendations
                    FROM keywords_recommendations_cache
                    WHERE cache_key = %s
                    AND timestamp >= %s
                """
                expiration_time = datetime.now() - timedelta(hours=1)
                cursor.execute(query, (cache_key, expiration_time))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Recomendação em cache encontrada para chave {cache_key}")
                    return result[0]
                logger.debug(f"Nenhuma recomendação válida em cache para chave {cache_key}")
                return None
        except psycopg2.Error as e:
            logger.error(f"Erro ao consultar recomendação em cache: {e}")
            return None

    def _save_prediction_to_cache(self, user_id: str, predictions: list[dict]):
        """Salva a predição no cache do PostgreSQL para user_id."""
        try:
            with self.db_connection.cursor() as cursor:
                # Insere ou atualiza a predição
                query = """
                    INSERT INTO predictions_cache (user_id, predictions, timestamp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET predictions = %s, timestamp = %s
                """
                current_time = datetime.now()
                cursor.execute(query, (user_id, Json(predictions), current_time, Json(predictions), current_time))
                self.db_connection.commit()
                logger.debug(f"Predição salva no cache para user_id {user_id}")
        except psycopg2.Error as e:
            logger.error(f"Erro ao salvar predição no cache: {e}")

    def _save_keywords_recommendation_to_cache(self, cache_key: str, recommendations: list[dict]):
        """Salva a recomendação no cache do PostgreSQL para keywords."""
        try:
            with self.db_connection.cursor() as cursor:
                # Insere ou atualiza a recomendação
                query = """
                    INSERT INTO keywords_recommendations_cache (cache_key, recommendations, timestamp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cache_key)
                    DO UPDATE SET recommendations = %s, timestamp = %s
                """
                current_time = datetime.now()
                cursor.execute(query,
                               (cache_key, Json(recommendations), current_time, Json(recommendations), current_time))
                self.db_connection.commit()
                logger.debug(f"Recomendação salva no cache para chave {cache_key}")
        except psycopg2.Error as e:
            logger.error(f"Erro ao salvar recomendação no cache: {e}")

    def train_model(self, state: StateManager, validation_file: str, force_retrain: bool = False) -> None:
        regressor_file = os.path.join(self.cache_dir, 'regressor.pkl')
        if not force_retrain and os.path.exists(regressor_file):
            logger.info("Modelo treinado encontrado em regressor.pkl; carregando modelo existente")
            state.REGRESSOR = joblib.load(regressor_file)
            state.PREDICTOR = self.predictor_class(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES,
                                                   state.REGRESSOR)
            logger.info("Modelo pré-treinado carregado com sucesso; pronto para predições e cálculo de métricas")
            return

        start_time = time.time()
        logger.info("Iniciando treinamento do modelo")
        state.REGRESSOR = self.trainer.train(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES, validation_file)
        if state.REGRESSOR:
            joblib.dump(state.REGRESSOR, regressor_file)
            logger.info(f"Modelo treinado salvo em {regressor_file}")

            # Calcula e salva os caches após o treinamento
            self._calculate_and_cache_metrics(state.INTERACOES, state.NOTICIAS)

            state.PREDICTOR = self.predictor_class(state.INTERACOES, state.NOTICIAS, state.USER_PROFILES,
                                                   state.REGRESSOR)
            elapsed = time.time() - start_time
            logger.info(f"Treinamento concluído em {elapsed:.2f} segundos")
        else:
            raise HTTPException(status_code=500, detail="Falha no treinamento: dados insuficientes")

    def predict(self, state: StateManager, user_id: str, number_of_records=10, keywords: Optional[List[str]] = None) -> \
    list[dict]:
        if state.PREDICTOR is None:
            logger.warning("Modelo não treinado")
            raise HTTPException(status_code=400, detail="Modelo não treinado")

        padrao = r"/noticia/(\d{4}/\d{2}/\d{2})/"
        start_time = time.time()

        # Verifica se o usuário está nos perfis para decidir o tipo de cache a usar
        if user_id in state.USER_PROFILES:
            # Usuário conhecido: usa o user_id como chave de cache
            cached_predictions = self._get_cached_prediction(user_id)
            if cached_predictions:
                logger.info(
                    f"Predições recuperadas do cache para user_id {user_id} em {time.time() - start_time:.2f} segundos")
                return cached_predictions

            # Gera predições personalizadas para usuários conhecidos
            predictions = state.PREDICTOR.predict(user_id, number_of_records)
            # Salva a predição no cache
            self._save_prediction_to_cache(user_id, predictions)
        else:
            # Usuário não encontrado: cold start com base em keywords
            # Gera uma chave de cache baseada nas keywords
            cache_key = self._generate_keywords_cache_key(keywords)
            cached_recommendations = self._get_cached_keywords_recommendation(cache_key)
            if cached_recommendations:
                logger.info(
                    f"Recomendações recuperadas do cache para keywords (chave: {cache_key}) em {time.time() - start_time:.2f} segundos")
                return cached_recommendations

            # Gera recomendações via cold start
            logger.info(f"Usuário {user_id} não encontrado; aplicando cold-start. Palavras-chave: {keywords}")
            popular_news = self.trainer.handle_cold_start(state.NOTICIAS, keywords)
            predictions = [{
                "page": page,
                "title": state.NOTICIAS[state.NOTICIAS['page'] == page]['title'].iloc[0],
                "link": state.NOTICIAS[state.NOTICIAS['page'] == page]['url'].iloc[0],
            } for page in popular_news]
            # Salva a recomendação no cache para keywords
            self._save_keywords_recommendation_to_cache(cache_key, predictions)

        elapsed = time.time() - start_time
        logger.info(f"Predições geradas para {user_id} em {elapsed:.2f} segundos")
        return predictions
