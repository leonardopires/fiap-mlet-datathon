import logging
import psutil
import pynvml

logger = logging.getLogger(__name__)


class ResourceLogger:
    """
    Esta classe verifica e registra como o computador está sendo usado (CPU, GPU, memória, disco),
    para sabermos se está tudo funcionando bem.
    """

    def __init__(self, cache_dir: str = 'data/cache'):
        """
        Configura o logger de recursos.

        Args:
            cache_dir (str): Onde os arquivos estão sendo salvos, para verificar o uso do disco.
        """
        self.cache_dir = cache_dir
        # Tenta inicializar o monitoramento da GPU (placa de vídeo)
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
        except pynvml.NVMLError:
            self.gpu_available = False
            logger.warning("NVIDIA GPU não detectada ou driver NVML não instalado.")

    def get_gpu_usage(self) -> str:
        """
        Verifica quanto da GPU está sendo usado.

        Returns:
            str: Uma mensagem dizendo o uso da GPU ou "N/A" se não houver GPU.
        """
        if not self.gpu_available:
            return "GPU: N/A"
        try:
            # Conta quantas GPUs estão disponíveis
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # Pega informações da primeira GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # Uso de memória da GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # Percentual de uso
                return f"GPU: {util.gpu}% usada, Memória GPU: {mem_info.used / 1024 ** 2:.2f}/{mem_info.total / 1024 ** 2:.2f} MB"
            return "GPU: N/A"
        except pynvml.NVMLError as e:
            return f"GPU: Erro ao acessar ({str(e)})"

    def log_resources(self, context: str) -> None:
        """
        Registra o uso de recursos do computador.

        Args:
            context (str): Uma mensagem explicando em que ponto estamos (ex.: "após processar batch").
        """
        # Mede o uso da CPU (processador)
        cpu_percent = psutil.cpu_percent(interval=1)
        # Verifica a memória do computador
        mem = psutil.virtual_memory()
        # Verifica o uso do disco onde os arquivos estão sendo salvos
        disk = psutil.disk_usage(self.cache_dir)
        # Junta todas as informações em uma mensagem
        logger.info(
            f"Estado dos recursos {context}: "
            f"{self.get_gpu_usage()}, "  # Uso da GPU vem primeiro
            f"CPU: {cpu_percent:.1f}%, "  # Percentual de uso do processador
            f"Memória: {mem.percent:.1f}% usada ({mem.available / (1024 ** 3):.2f} GB livre), "  # Uso da memória em GB
            f"Disco em {self.cache_dir}: {disk.percent:.1f}% usado ({disk.free / (1024 ** 3):.2f} GB livre)"
            # Uso do disco
        )