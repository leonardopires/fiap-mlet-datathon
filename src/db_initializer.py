import logging
import psycopg2
from psycopg2.extensions import connection
import time
import os

logger = logging.getLogger(__name__)


class DBInitializer:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection = None

    def connect(self) -> connection:
        """Estabelece conexão com o PostgreSQL."""
        retry_attempts = 5
        retry_delay = 5  # segundos
        for attempt in range(retry_attempts):
            try:
                self.connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password
                )
                logger.info("Conexão com PostgreSQL estabelecida com sucesso")
                return self.connection
            except psycopg2.Error as e:
                logger.warning(f"Falha ao conectar ao PostgreSQL (tentativa {attempt + 1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Não foi possível conectar ao PostgreSQL após várias tentativas")
                    raise

    def initialize_db(self):
        """Cria as tabelas predictions_cache e keywords_recommendations_cache se não existirem."""
        try:
            with self.connection.cursor() as cursor:
                # Tabela para predições baseadas em user_id
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions_cache (
                        user_id TEXT PRIMARY KEY,
                        predictions JSONB NOT NULL,
                        timestamp TIMESTAMP NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_predictions_cache_timestamp ON predictions_cache (timestamp);
                """)
                # Tabela para recomendações baseadas em keywords
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS keywords_recommendations_cache (
                        cache_key TEXT PRIMARY KEY,
                        recommendations JSONB NOT NULL,
                        timestamp TIMESTAMP NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_keywords_recommendations_cache_timestamp ON keywords_recommendations_cache (timestamp);
                """)
                self.connection.commit()
                logger.info("Tabelas predictions_cache e keywords_recommendations_cache criadas ou já existem")
        except psycopg2.Error as e:
            logger.error(f"Erro ao criar tabelas: {e}")
            raise

    def close(self):
        """Fecha a conexão com o PostgreSQL."""
        if self.connection:
            self.connection.close()
            logger.debug("Conexão com PostgreSQL fechada")


def get_db_connection():
    """Retorna uma conexão com o PostgreSQL."""
    db_initializer = DBInitializer(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "recomendador_db"),
        user=os.getenv("POSTGRES_USER", "recomendador"),
        password=os.getenv("POSTGRES_PASSWORD", "senha123")
    )
    return db_initializer.connect()
