
  const STATUS_MAP = new Map([
      ["starting", "Inicializando..."],
      ["preprocessing", "Pré-processando dados..."],
      ["training", "Treinando modelo..."],
      ["completed", "Concluído"],
      ["error", "Erro"],
      ["converting kaggle data", "Convertendo dataset Kaggle..."],
      ["calculating", "Calculando métricas..."],
      ["predicting", "Realizando predições..."],
    ]
  );

  function mapStatus(status: string) {
    return STATUS_MAP.get(status) || status;
  }

  export { mapStatus };