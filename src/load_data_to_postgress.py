import pandas as pd
import psycopg2
from src.data_loader import DataLoader

def load_data_to_postgres():
    conn = psycopg2.connect(
        dbname="recomendador_db",
        user="recomendador",
        password="senha123",
        host="postgres",
        port="5432"
    )
    cur = conn.cursor()

    # Cria tabelas
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id UUID PRIMARY KEY,
            history TEXT
        );
        CREATE TABLE IF NOT EXISTS news (
            page TEXT PRIMARY KEY,
            title TEXT
        );
    """)

    # Carrega dados
    data_loader = DataLoader()
    interacoes = data_loader.load_and_concat_files('data/files/treino/treino_parte*.csv')
    noticias = data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')

    # Insere usuários
    for _, row in interacoes.iterrows():
        cur.execute("INSERT INTO users (user_id, history) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (row['userId'], row['history']))

    # Insere notícias
    for _, row in noticias.iterrows():
        cur.execute("INSERT INTO news (page, title) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (row['page'], row['title']))

    conn.commit()
    cur.close()
    conn.close()