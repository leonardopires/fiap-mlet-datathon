import logging
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List
import time

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

    def generate_embeddings(self, titles: List[str]) -> np.ndarray:
        """
        Transforma uma lista de títulos de notícias em embeddings (números que representam o significado).

        Args:
            titles (List[str]): Lista de títulos das notícias.

        Returns:
            np.ndarray: Um array (como uma tabela) de números, onde cada linha é o embedding de um título.
        """
        # Marca o tempo inicial para medir quanto demora
        start_time = time.time()
        logger.info("Gerando novos embeddings para notícias em batches")

        # Carrega o modelo que entende texto e o coloca no dispositivo escolhido (GPU ou CPU)
        model = SentenceTransformer(self.model_name).to(self.device)

        # Calcula quantos grupos (batches) precisamos para processar todos os títulos
        total_batches = (len(titles) + self.batch_size - 1) // self.batch_size
        logger.debug(f"Preparando {len(titles)} títulos para codificação em {total_batches} batches")

        # Usa uma técnica para fazer os cálculos mais rápido na GPU, se disponível
        with torch.cuda.amp.autocast():
            # Converte os títulos em embeddings usando o modelo
            embeddings = model.encode(
                titles,  # Os títulos que queremos transformar
                batch_size=self.batch_size,  # Quantos títulos processar de uma vez
                convert_to_tensor=True,  # Retorna os resultados como "tensores" (um formato que a GPU gosta)
                device=self.device,  # Onde fazer os cálculos (GPU ou CPU)
                show_progress_bar=True  # Mostra uma barra de progresso no terminal
            ).cpu().numpy()  # Move os resultados para a CPU e transforma em um array simples

        # Calcula quanto tempo levou e informa no log
        elapsed = time.time() - start_time
        logger.info(f"Embeddings gerados para {len(titles)} títulos em {elapsed:.2f} segundos")
        return embeddings