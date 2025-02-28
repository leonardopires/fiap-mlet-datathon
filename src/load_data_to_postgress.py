import pandas as pd
import psycopg2
from src.data_loader import DataLoader

def load_data_to_postgres():
    print("Connecting to database...")
    conn = psycopg2.connect(
        dbname="recomendador_db",
        user="recomendador",
        password="senha123",
        host="postgres",
        port="5432"
    )
    print("Connected to PostgreSQL.")

    cur = conn.cursor()

    # Criando tabela
    print("Creating tables...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            page TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            date TIMESTAMP
        );
    """)
    conn.commit()
    print("Table 'news' created!")

    # Carregando dados
    print("Loading data from CSV...")
    data_loader = DataLoader()
    noticias = data_loader.load_and_concat_files('data/itens/itens/itens-parte*.csv')
    print(f"Loaded {len(noticias)} rows from CSV.")

    # Inserindo dados
    print("Inserting news data...")
    for _, row in noticias.iterrows():
        cur.execute(
            "INSERT INTO news (page, title, url, date) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (row['page'], row['title'], row['url'], row.get('issued', None))  # Pegando a data correta
        )

    conn.commit()
    print("Database commit successful!")

    cur.close()
    conn.close()
    print("Database connection closed.")

if __name__ == "__main__":
    load_data_to_postgres()
