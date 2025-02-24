import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from typing import Tuple, Dict
import os
import glob
import h5py
import psutil
import time

# Configura o logger para mensagens detalhadas com timestamp
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Preprocessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', batch_size: int = 256):
        """
        Inicializa o pré-processador com configurações do modelo e parâmetros de desempenho.

        Args:
            model_name (str): Nome do modelo Sentence Transformers a ser usado (padrão: paraphrase-multilingual-MiniLM-L12-v2).
            batch_size (int): Tamanho do lote para geração de embeddings na GPU (padrão: 256).
        """
        # Define o dispositivo de computação: GPU (CUDA) se disponível, senão CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Detectando dispositivo disponível: {self.device}")
        # Armazena o nome do modelo para carregamento posterior
        self.model_name = model_name
        self.batch_size = batch_size
        # Define o diretório de cache para salvar arquivos temporários e finais
        self.cache_dir = 'data/cache'
        # Cria o diretório de cache, se não existir, sem levantar erro se já presente
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Diretório de cache configurado em: {self.cache_dir}")

    def calculate_engagement(self, clicks, time, scroll):
        """
        Calcula a pontuação de engajamento para uma interação com base em cliques, tempo e rolagem.

        Args:
            clicks (float): Número de cliques na página.
            time (float): Tempo gasto na página em milissegundos.
            scroll (float): Percentual de rolagem na página (0-100).

        Returns:
            float: Pontuação de engajamento combinada com pesos predefinidos.
        """
        # Calcula engajamento com pesos: 30% cliques, 50% tempo (convertido para segundos), 20% rolagem
        engagement = (clicks * 0.3) + (time / 1000 * 0.5) + (scroll * 0.2)
        return engagement

    def calculate_recency(self, timestamp, max_timestamp):
        """
        Calcula a recência de uma interação usando uma função exponencial decrescente.

        Args:
            timestamp (int): Timestamp da interação em milissegundos.
            max_timestamp (int): Maior timestamp no histórico do usuário em milissegundos.

        Returns:
            float: Valor de recência entre 0 e 1, maior para interações mais recentes.
        """
        # Calcula a diferença de tempo em segundos e aplica uma função exponencial com meia-vida de 7 dias
        time_diff = (max_timestamp - timestamp) / (1000 * 60 * 60 * 24)  # Converte para dias
        recency = np.exp(-time_diff / 7)  # Decaimento exponencial com base em 7 dias
        return recency

    def _load_embedding_chunk(self, args: Tuple[str, int, int]) -> np.ndarray:
        """
        Carrega um chunk específico de embeddings do arquivo HDF5 em paralelo.

        Args:
            args (Tuple[str, int, int]): Caminho do arquivo HDF5, índice inicial e final do chunk.

        Returns:
            np.ndarray: Array NumPy contendo os embeddings do chunk especificado.
        """
        cache_file, start, end = args
        start_time = time.time()
        logger.info(f"Iniciando carregamento do chunk de embeddings de {start} a {end} do arquivo {cache_file}")
        # Abre o arquivo HDF5 em modo leitura
        with h5py.File(cache_file, 'r') as f:
            # Lê o intervalo especificado do dataset 'embeddings'
            chunk = f['embeddings'][start:end]
            logger.debug(f"Extraídos {len(chunk)} embeddings do chunk {start}-{end}")
        elapsed = time.time() - start_time
        logger.info(f"Chunk de embeddings {start}-{end} carregado com sucesso em {elapsed:.2f} segundos")
        return chunk

    def _load_user_profiles_chunk(self, cache_file: str) -> Dict[str, np.ndarray]:
        """
        Carrega um chunk de perfis de usuário de um arquivo HDF5.

        Args:
            cache_file (str): Caminho do arquivo HDF5 contendo os perfis de usuário.

        Returns:
            Dict[str, np.ndarray]: Dicionário mapeando user_id para embeddings.
        """
        start_time = time.time()
        logger.info(f"Iniciando carregamento de perfis de usuário do cache em {cache_file}")
        user_profiles = {}
        with h5py.File(cache_file, 'r') as f:
            if 'embeddings' in f and 'user_ids' in f:  # Nova estrutura
                embeddings = f['embeddings'][:]
                user_ids = f['user_ids'][:].astype(str)
                user_profiles = dict(zip(user_ids, embeddings))
                logger.debug(f"Carregados {len(user_profiles)} perfis na nova estrutura")
            else:  # Estrutura antiga (compatibilidade)
                for user_id in f.keys():
                    user_profiles[user_id] = f[user_id][:]
                    if len(user_profiles) % 1000 == 0:
                        logger.debug(f"Carregados {len(user_profiles)} perfis do chunk até agora (estrutura antiga)")
        elapsed = time.time() - start_time
        logger.info(f"Carregados {len(user_profiles)} perfis de usuário de {cache_file} em {elapsed:.2f} segundos")
        return user_profiles

    def preprocess(self, interacoes: pd.DataFrame, noticias: pd.DataFrame, subsample_frac: float = None,
                   force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Pré-processa interações e notícias, gerando embeddings e perfis de usuário com logs detalhados e medição de tempo.

        Args:
            interacoes (pd.DataFrame): Dados de interações dos usuários (treino_parte*.csv).
            noticias (pd.DataFrame): Dados das notícias (itens-parte*.csv).
            subsample_frac (float, opcional): Fração dos dados a processar (0-1), para testes rápidos.
            force_reprocess (bool): Força reprocessamento completo, ignorando cache existente.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: Interações, notícias com embeddings e perfis de usuário.
        """
        total_start_time = time.time()
        logger.info("Iniciando pré-processamento completo dos dados")

        # Aplica subamostragem se especificada
        if subsample_frac is not None and 0 < subsample_frac < 1:
            subsample_start_time = time.time()
            logger.info(f"Aplicando subamostragem com fração {subsample_frac} nos dados de interações e notícias")
            interacoes = interacoes.sample(frac=subsample_frac, random_state=42)
            noticias = noticias.sample(frac=subsample_frac, random_state=42)
            elapsed = time.time() - subsample_start_time
            logger.info(
                f"Subamostragem concluída: {len(interacoes)} interações e {len(noticias)} notícias em {elapsed:.2f} segundos")

        # Processa embeddings das notícias
        embedding_cache = os.path.join(self.cache_dir, 'news_embeddings.h5')
        if force_reprocess and os.path.exists(embedding_cache):
            logger.info(f"Forçando reprocessamento: removendo cache de embeddings existente em {embedding_cache}")
            os.remove(embedding_cache)

        if os.path.exists(embedding_cache):
            embeddings_start_time = time.time()
            logger.info(f"Carregando embeddings existentes de {embedding_cache}")
            with h5py.File(embedding_cache, 'r') as f:
                embeddings = f['embeddings'][:]
            noticias['embedding'] = embeddings.tolist()
            elapsed = time.time() - embeddings_start_time
            logger.info(f"Embeddings carregados para {len(noticias)} notícias em {elapsed:.2f} segundos")
        else:
            embeddings_start_time = time.time()
            logger.info("Gerando novos embeddings para notícias em batches")
            model = SentenceTransformer(self.model_name).to(self.device)
            noticias_titles = noticias['title'].tolist()
            total_batches = (len(noticias) + self.batch_size - 1) // self.batch_size
            logger.debug(f" Preparando {len(noticias_titles)} títulos para codificação em {total_batches} batches")
            with torch.cuda.amp.autocast():
                embeddings = model.encode(
                    noticias_titles,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=True
                ).cpu().numpy()
            noticias['embedding'] = embeddings.tolist()
            save_start = time.time()
            with h5py.File(embedding_cache, 'w') as f:
                f.create_dataset('embeddings', data=embeddings, compression="gzip", compression_opts=4)
            save_elapsed = time.time() - save_start
            total_elapsed = time.time() - embeddings_start_time
            logger.info(
                f"Embeddings gerados para {len(noticias)} notícias em {total_batches} batches em {total_elapsed:.2f} segundos (salvamento: {save_elapsed:.2f} segundos)")

        # Cria o lookup de embeddings por página como tensores na GPU
        lookup_start_time = time.time()
        logger.info("Criando lookup de embeddings por page para acesso rápido na GPU")
        page_to_embedding = {page: torch.tensor(emb, device=self.device, dtype=torch.float32)
                             for page, emb in zip(noticias['page'], noticias['embedding'])}
        elapsed = time.time() - lookup_start_time
        logger.info(f"Lookup criado com {len(page_to_embedding)} entradas em {elapsed:.2f} segundos")

        # Remove caches antigos se forçar reprocessamento
        if force_reprocess:
            logger.info("Forçando reprocessamento: limpando caches antigos de perfis de usuário")
            chunk_files = glob.glob(os.path.join(self.cache_dir, 'user_profiles_*.h5'))
            for f in chunk_files:
                logger.info(f"Removendo arquivo de cache antigo: {f}")
                os.remove(f)

        # Processa interações diretamente na GPU sem multiprocessing
        interacoes_start_time = time.time()
        logger.info("Iniciando processamento do histórico de interações na GPU")
        user_profiles = {}
        total_interacoes = len(interacoes)
        batch_size = 1000  # Processa em batches para evitar sobrecarga de memória GPU

        for i in range(0, total_interacoes, batch_size):
            batch_end = min(i + batch_size, total_interacoes)
            batch = interacoes.iloc[i:batch_end]
            elapsed_so_far = time.time() - interacoes_start_time
            logger.info(
                f"Processando batch {i}-{batch_end} de {total_interacoes} interações em {elapsed_so_far:.2f} segundos")

            # Processa o batch na GPU
            for idx, row in batch.iterrows():
                user_id = row['userId']
                hist = row['history'].split(', ')
                clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
                times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
                scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
                timestamps = [int(x) for x in row['timestampHistory'].split(', ')]

                max_ts = max(timestamps)
                embeddings, weights = [], []

                for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
                    if h in page_to_embedding:
                        emb = page_to_embedding[h]
                        eng = self.calculate_engagement(c, t, s)
                        rec = self.calculate_recency(ts, max_ts)
                        embeddings.append(emb)
                        weights.append(eng * rec)

                if embeddings:
                    embeddings_tensor = torch.stack(embeddings)
                    weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
                    weighted_sum = torch.sum(embeddings_tensor * weights_tensor.unsqueeze(1), dim=0)
                    total_weight = torch.sum(weights_tensor)
                    user_profiles[user_id] = (weighted_sum / total_weight).cpu().numpy()

            # Log de recursos a cada batch
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(self.cache_dir)
            logger.info(
                f"Estado dos recursos após batch {i}-{batch_end}: "
                f"CPU: {cpu_percent:.1f}%, "
                f"Memória: {mem.percent:.1f}% usada ({mem.available / (1024 ** 3):.2f} GB livre), "
                f"Disco em {self.cache_dir}: {disk.percent:.1f}% usado ({disk.free / (1024 ** 3):.2f} GB livre)"
            )

        interacoes_elapsed = time.time() - interacoes_start_time
        logger.info(f"Processamento das interações concluído em {interacoes_elapsed:.2f} segundos")

        # Salva os perfis finais no cache
        final_cache = os.path.join(self.cache_dir, 'user_profiles_final.h5')
        save_start_time = time.time()
        logger.info(f"Salvando {len(user_profiles)} perfis finais em {final_cache}")

        user_ids = list(user_profiles.keys())
        embeddings = np.array(list(user_profiles.values()))

        with h5py.File(final_cache, 'w') as f:
            f.create_dataset('embeddings', data=embeddings, compression="gzip", compression_opts=4)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('user_ids', data=np.array(user_ids, dtype=object), dtype=dt)
            logger.debug(f"Salvou {len(user_ids)} perfis como matriz única")

            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(self.cache_dir)
            logger.info(
                f"Estado dos recursos após salvar {len(user_ids)} perfis: "
                f"CPU: {cpu_percent:.1f}%, "
                f"Memória: {mem.percent:.1f}% usada ({mem.available / (1024 ** 3):.2f} GB livre), "
                f"Disco em {self.cache_dir}: {disk.percent:.1f}% usado ({disk.free / (1024 ** 3):.2f} GB livre)"
            )

        save_elapsed = time.time() - save_start_time
        logger.info(f"Perfis finais salvos em {save_elapsed:.2f} segundos")

        # Persistir INTERACOES e NOTICIAS
        interacoes_cache = os.path.join(self.cache_dir, 'interacoes.h5')
        noticias_cache = os.path.join(self.cache_dir, 'noticias.h5')
        logger.info(f"Salvando INTERACOES em {interacoes_cache}")
        interacoes.to_hdf(interacoes_cache, key='interacoes', mode='w', complevel=4, complib='blosc')
        logger.info(f"Salvando NOTICIAS em {noticias_cache}")
        noticias.to_hdf(noticias_cache, key='noticias', mode='w', complevel=4, complib='blosc')

        total_elapsed = time.time() - total_start_time
        logger.info(
            f"Pré-processamento concluído: {len(user_profiles)} perfis gerados em {total_elapsed:.2f} segundos (tempo total)")
        return interacoes, noticias, user_profiles