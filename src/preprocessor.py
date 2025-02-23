import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        logger.info("Carregando modelo Sentence Transformers...")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = batch_size

    def calculate_engagement(self, clicks, time, scroll):
        return (clicks * 0.3) + (time / 1000 * 0.5) + (scroll * 0.2)

    def calculate_recency(self, timestamp, max_timestamp):
        return np.exp(-(max_timestamp - timestamp) / (7 * 24 * 3600 * 1000))

    def preprocess(self, interacoes: pd.DataFrame, noticias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        logger.info("Gerando embeddings para notícias (em batches)...")
        embeddings = []
        for i in range(0, len(noticias), self.batch_size):
            batch = noticias['title'][i:i+self.batch_size].tolist()
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, device=self.device, show_progress_bar=False).cpu().numpy()
            embeddings.extend(batch_embeddings)
            logger.info(f"Processado batch {i//self.batch_size + 1} de {len(noticias)//self.batch_size + 1}")
        noticias['embedding'] = embeddings

        logger.info("Processando histórico de interações...")
        user_profiles = {}
        for i, row in interacoes.iterrows():
            if i % 1000 == 0:
                logger.info(f"Processados {i} registros de interações...")
            user_id = row['userId']
            hist = row['history'].split(', ')
            clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
            times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
            scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
            timestamps = [int(x) for x in row['timestampHistory'].split(', ')]
            max_ts = max(timestamps)
            embeddings, weights = [], []
            for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
                if h in noticias['page'].values:
                    emb = noticias[noticias['page'] == h]['embedding'].values[0]
                    eng = self.calculate_engagement(c, t, s)
                    rec = self.calculate_recency(ts, max_ts)
                    embeddings.append(emb)
                    weights.append(eng * rec)
            if embeddings:
                user_profiles[user_id] = np.average(embeddings, axis=0, weights=weights)
        logger.info("Pré-processamento concluído.")
        return interacoes, noticias, user_profiles
