import logging
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Optional
import time
import pandas as pd

# Configura o logger para registrar mensagens úteis durante a execução
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Esta classe é responsável por transformar textos (como títulos de notícias) em números que o computador
    pode entender, chamados de "embeddings". Esses números capturam o significado do texto.
    """

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', batch_size: int = 256):
        """
        Configura o gerador de embeddings.

        Args:
            model_name (str): O nome do modelo que vamos usar para criar os embeddings. É como escolher uma "ferramenta" específica.
            batch_size (int): Quantos textos processamos de uma vez. Um número maior usa mais memória, mas é mais rápido.
        """
        # Decide se vamos usar a GPU (placa de vídeo) ou a CPU (processador normal) para fazer os cálculos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Guarda o nome do modelo e o tamanho do lote para usar depois
        self.model_name = model_name
        self.batch_size = batch_size
        # Informa no log qual dispositivo (GPU ou CPU) estamos usando
        logger.info(f"EmbeddingGenerator inicializado no dispositivo: {self.device}")

    def generate_embeddings(self, titles: List[str], noticias_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Transforma uma lista de títulos de notícias em embeddings, opcionalmente combinando com conteúdo.

        Args:
            titles (List[str]): Lista de títulos das notícias.
            noticias_df (Optional[pd.DataFrame]): DataFrame com dados das notícias, incluindo 'body' (opcional).

        Returns:
            np.ndarray: Um array de embeddings representando títulos (e conteúdo, se fornecido).
        """
        start_time = time.time()
        logger.info("Gerando novos embeddings para notícias em batches")

        model = SentenceTransformer(self.model_name).to(self.device)

        # Se noticias_df for fornecido e tiver 'body', combinar títulos com conteúdo
        if noticias_df is not None and 'body' in noticias_df.columns:
            logger.info("Combinando títulos com conteúdo para embeddings")
            texts = [f"{title} {body}" if pd.notna(body) else title
                     for title, body in zip(titles, noticias_df['body'])]
        else:
            logger.info("Usando apenas títulos para embeddings")
            texts = titles

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(f"Preparando {len(texts)} textos para codificação em {total_batches} batches")

        with torch.amp.autocast('cuda'):
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True
            ).cpu().numpy()

        elapsed = time.time() - start_time
        logger.info(f"Embeddings gerados para {len(texts)} textos em {elapsed:.2f} segundos")
        return embeddings
