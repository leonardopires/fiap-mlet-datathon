import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from typing import Tuple, Dict
from multiprocessing import Pool
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
        # Armazena o nome do modelo para carregamento posterior nos processos filhos
        self.model_name = model_name
        self.batch_size = batch_size
        # Calcula o número máximo de processos baseado em CPU e memória disponível
        total_cores = os.cpu_count() or 1  # Usa 1 se cpu_count() falhar
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Memória livre em GB
        max_processes = max(2, min(int(total_cores * 0.75), int(available_memory // 1.5)))  # ~1.5 GB por processo
        # Define um número fixo de processos menor para estabilidade
        self.num_processes = min(4, max_processes)
        logger.info(
            f"Configurando paralelismo: {self.num_processes} processos (baseado em {total_cores} núcleos e {available_memory:.2f} GB livres)")
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
        # Abre o arquivo HDF5 em modo leitura
        with h5py.File(cache_file, 'r') as f:
            # Itera sobre todas as chaves (user_ids) no arquivo
            for user_id in f.keys():
                # Carrega o embedding correspondente ao user_id
                user_profiles[user_id] = f[user_id][:]
                if len(user_profiles) % 1000 == 0:
                    logger.debug(f"Carregados {len(user_profiles)} perfis do chunk até agora")
        elapsed = time.time() - start_time
        logger.info(f"Carregados {len(user_profiles)} perfis de usuário de {cache_file} em {elapsed:.2f} segundos")
        return user_profiles

    def _process_interactions_chunk(self, args: Tuple[pd.DataFrame, Dict[str, np.ndarray], int]) -> Dict[
        str, np.ndarray]:
        """
        Processa um chunk de interações para gerar perfis de usuário, com logs detalhados e medição de tempo.

        Args:
            args (Tuple): Contém o chunk do DataFrame de interações, lookup de embeddings por page e índice do chunk.

        Returns:
            Dict[str, np.ndarray]: Perfis de usuário gerados ou carregados do cache (user_id -> embedding médio ponderado).
        """
        chunk, page_to_embedding, chunk_idx = args
        cache_file = os.path.join(self.cache_dir, f'user_profiles_chunk_{chunk_idx}.h5')

        # Verifica se o chunk já está em cache
        if os.path.exists(cache_file):
            logger.info(f"Chunk {chunk_idx} encontrado em cache: {cache_file}. Carregando...")
            return self._load_user_profiles_chunk(cache_file)

        chunk_start_time = time.time()
        logger.info(f"Iniciando processamento do chunk {chunk_idx} com {len(chunk)} interações")

        # Define o dispositivo para o processo atual e carrega o modelo SentenceTransformer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Configurando dispositivo para o chunk {chunk_idx}: {device}")
        model_start_time = time.time()
        model = SentenceTransformer(self.model_name).to(device)
        model_elapsed = time.time() - model_start_time
        logger.info(f"Modelo SentenceTransformer carregado para o chunk {chunk_idx} em {model_elapsed:.2f} segundos")

        user_profiles = {}
        total_chunk = len(chunk)

        # Processa cada linha do chunk para gerar perfis de usuário
        for i, row in chunk.iterrows():
            if i % 50 == 0:
                elapsed_so_far = time.time() - chunk_start_time
                logger.info(
                    f"Processados {i}/{total_chunk} registros no chunk {chunk_idx} em {elapsed_so_far:.2f} segundos")

            user_id = row['userId']
            # Divide as strings de histórico em listas para processamento
            hist = row['history'].split(', ')
            clicks = [float(x) for x in row['numberOfClicksHistory'].split(', ')]
            times = [float(x) for x in row['timeOnPageHistory'].split(', ')]
            scrolls = [float(x) for x in row['scrollPercentageHistory'].split(', ')]
            timestamps = [int(x) for x in row['timestampHistory'].split(', ')]

            # Calcula o timestamp máximo para recência
            max_ts = max(timestamps)
            embeddings, weights = [], []

            # Itera sobre o histórico do usuário para calcular embeddings ponderados
            for h, c, t, s, ts in zip(hist, clicks, times, scrolls, timestamps):
                if h in page_to_embedding:
                    # Obtém o embedding da página a partir do lookup
                    emb = page_to_embedding[h]
                    # Calcula engajamento e recência para o peso
                    eng = self.calculate_engagement(c, t, s)
                    rec = self.calculate_recency(ts, max_ts)
                    embeddings.append(emb)
                    weights.append(eng * rec)

            if embeddings:
                # Calcula a média ponderada dos embeddings para o perfil do usuário
                user_profiles[user_id] = np.average(embeddings, axis=0, weights=weights)
                if len(user_profiles) % 1000 == 0:
                    logger.debug(f"Gerados {len(user_profiles)} perfis no chunk {chunk_idx} até agora")

        # Salva o chunk processado no cache
        save_start_time = time.time()
        logger.info(f"Salvando {len(user_profiles)} perfis do chunk {chunk_idx} em {cache_file}")
        with h5py.File(cache_file, 'w') as f:
            for user_id, embedding in user_profiles.items():
                f.create_dataset(user_id, data=embedding, compression="gzip", compression_opts=4)
        save_elapsed = time.time() - save_start_time
        total_elapsed = time.time() - chunk_start_time
        logger.info(
            f"Chunk {chunk_idx} concluído: {len(user_profiles)} perfis gerados e salvos em {total_elapsed:.2f} segundos (salvamento: {save_elapsed:.2f} segundos)")
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
            logger.info(f"Carregando embeddings existentes de {embedding_cache} em paralelo")
            with h5py.File(embedding_cache, 'r') as f:
                total_embeddings = len(f['embeddings'])
                logger.debug(f"Encontrados {total_embeddings} embeddings no cache")
                chunk_size = max(1, total_embeddings // self.num_processes)
                chunks = [(embedding_cache, i, min(i + chunk_size, total_embeddings))
                          for i in range(0, total_embeddings, chunk_size)]
            logger.info(
                f"Dividido em {len(chunks)} chunks para carregamento paralelo com {self.num_processes} processos")
            with Pool(self.num_processes) as pool:
                results = pool.map(self._load_embedding_chunk, chunks)
            embeddings = np.concatenate(results, axis=0)
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

        # Cria o lookup de embeddings por página
        lookup_start_time = time.time()
        logger.info("Criando lookup de embeddings por page para acesso rápido")
        page_to_embedding = dict(zip(noticias['page'], noticias['embedding']))
        elapsed = time.time() - lookup_start_time
        logger.info(f"Lookup criado com {len(page_to_embedding)} entradas em {elapsed:.2f} segundos")

        # Remove caches antigos se forçar reprocessamento
        if force_reprocess:
            logger.info("Forçando reprocessamento: limpando caches antigos de perfis de usuário")
            chunk_files = glob.glob(os.path.join(self.cache_dir, 'user_profiles_chunk_*.h5'))
            for f in chunk_files:
                logger.info(f"Removendo arquivo de cache antigo: {f}")
                os.remove(f)

        # Processa interações em paralelo
        interacoes_start_time = time.time()
        logger.info("Iniciando processamento paralelo do histórico de interações")
        total_interacoes = len(interacoes)
        chunk_size = max(1, total_interacoes // self.num_processes)
        chunks = [(interacoes.iloc[i:i + chunk_size], page_to_embedding, idx)
                  for idx, i in enumerate(range(0, total_interacoes, chunk_size))]
        logger.info(
            f"Interações divididas em {len(chunks)} chunks para {self.num_processes} processos, com {chunk_size} interações por chunk")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evita conflitos com tokenizers em paralelo
        with Pool(self.num_processes) as pool:
            logger.debug("Iniciando pool de processos para mapear chunks")
            results = pool.map(self._process_interactions_chunk, chunks)
        interacoes_elapsed = time.time() - interacoes_start_time
        logger.info(f"Processamento paralelo das interações concluído em {interacoes_elapsed:.2f} segundos")

        # Combina os resultados dos chunks em um único dicionário
        combine_start_time = time.time()
        logger.info("Combinando resultados dos chunks em um único dicionário de perfis de usuário")
        user_profiles = {}
        for chunk_idx, chunk_result in enumerate(results):
            logger.debug(f"Adicionando resultados do chunk {chunk_idx} ao dicionário principal")
            user_profiles.update(chunk_result)
        elapsed = time.time() - combine_start_time
        logger.info(f"Resultados combinados: {len(user_profiles)} perfis de usuário em {elapsed:.2f} segundos")

        # Salva os perfis finais no cache
        final_cache = os.path.join(self.cache_dir, 'user_profiles_final.h5')
        save_start_time = time.time()
        logger.info(f"Salvando {len(user_profiles)} perfis finais em {final_cache}")
        with h5py.File(final_cache, 'w') as f:
            for user_id, embedding in user_profiles.items():
                f.create_dataset(user_id, data=embedding, compression="gzip", compression_opts=4)
                if len(f) % 10000 == 0:
                    logger.debug(f"Salvando perfil {len(f)}/{len(user_profiles)} no arquivo final")
        save_elapsed = time.time() - save_start_time
        total_elapsed = time.time() - total_start_time
        logger.info(f"Perfis finais salvos em {save_elapsed:.2f} segundos")
        logger.info(
            f"Pré-processamento concluído: {len(user_profiles)} perfis gerados em {total_elapsed:.2f} segundos (tempo total)")
        return interacoes, noticias, user_profiles