// frontend/src/utils/metricsConfig.ts
export interface MetricRange {
  range: [number, number];
  category: 'ruim' | 'aceitável' | 'bom' | 'excelente';
}

export interface MetricDefinition {
  label: string;
  field: string;
  ranges: MetricRange[];
  significantDigits: number;
  transform?: (value: number) => string;
  tooltip: string;
  highIsBad: boolean;
}

export const metricsConfig: MetricDefinition[] = [
  {
    label: "Precisão top 10 (P@10)",
    field: "precision_at_k",
    ranges: [
      { range: [0, 0.05], category: 'ruim' },
      { range: [0.05, 0.1], category: 'aceitável' },
      { range: [0.1, 0.2], category: 'bom' },
      { range: [0.2, 1], category: 'excelente' },
    ],
    significantDigits: 2,
    transform: (v: number) => `${(v * 100).toFixed(2)}%`,
    tooltip: "Percentual de itens relevantes entre os 10 primeiros recomendados.",
    highIsBad: true,
  },
  {
    label: "Recall top 10 (R@10)",
    field: "recall_at_k",
    ranges: [
      { range: [0, 0.05], category: 'ruim' },
      { range: [0.05, 0.08], category: 'aceitável' },
      { range: [0.08, 0.15], category: 'bom' },
      { range: [0.15, 1], category: 'excelente' },
    ],
    significantDigits: 2,
    transform: (v: number) => `${(v * 100).toFixed(2)}%`,
    tooltip: "Percentual de itens relevantes recuperados entre os 10 primeiros em relação ao total de itens relevantes.",
    highIsBad: true,
  },
  {
    label: "Mean Reciprocal Rank (MRR)",
    field: "mrr",
    ranges: [
      { range: [0, 0.1], category: 'ruim' },
      { range: [0.1, 0.2], category: 'aceitável' },
      { range: [0.2, 0.5], category: 'bom' },
      { range: [0.5, 1], category: 'excelente' },
    ],
    significantDigits: 3,
    tooltip: "Média da posição inversa do primeiro item relevante (entre 0 e 1).",
    highIsBad: false,
  },
  {
    label: "Intra-List Similarity (ILS)",
    field: "intra_list_similarity",
    ranges: [
      { range: [0.4, 1], category: 'ruim' },
      { range: [0.3, 0.4], category: 'aceitável' },
      { range: [0.15, 0.3], category: 'bom' },
      { range: [0, 0.15], category: 'excelente' },
    ],
    significantDigits: 4,
    tooltip: "Similaridade média entre os itens recomendados, indicando diversidade (0 a 1).",
    highIsBad: true,
  },
  {
    label: "Cobertura de Catálogo (CC)",
    field: "catalog_coverage",
    ranges: [
      { range: [0, 0.01], category: 'ruim' },
      { range: [0.01, 0.02], category: 'aceitável' },
      { range: [0.02, 0.05], category: 'bom' },
      { range: [0.05, 1], category: 'excelente' },
    ],
    significantDigits: 2,
    transform: (v: number) => `${(v * 100).toFixed(2)}%`,
    tooltip: "Percentual de itens do catálogo que o sistema consegue recomendar.",
    highIsBad: false,
  },
];