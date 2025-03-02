/** @jsxImportSource @emotion/react */
import React, { useState, useEffect, useRef } from 'react';
import { css } from '@emotion/react';
import {
  Grid,
  Typography,
  Box,
  Tooltip,
  IconButton,
  useTheme,
  Button,
  Collapse,
} from '@mui/material';
import axios from 'axios';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useSnackbar } from '../contexts/SnackbarContext';
import { metricsConfig, MetricDefinition } from '../utils/metricsConfig';

interface MetricsDisplayProps {
  metrics: { [key: string]: number };
}

interface InterpretationStatus {
  running: boolean;
  progress: string;
  error: string | null;
  interpretation: string | null;
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
  const theme = useTheme();
  const { showSnackbar, updateSnackbar } = useSnackbar();
  const [showInterpretation, setShowInterpretation] = useState<boolean>(false);
  const [interpretation, setInterpretation] = useState<string | null>(null);
  const [interpretationStatus, setInterpretationStatus] = useState<InterpretationStatus>({
    running: false,
    progress: 'idle',
    error: null,
    interpretation: null,
  });
  const [interpretationId, setInterpretationId] = useState<string | null>(null);
  const statusWsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/interpret/status');
    statusWsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket de status de interpretação conectado');
    };

    ws.onmessage = (event) => {
      const status: InterpretationStatus = JSON.parse(event.data);
      console.log('Status de interpretação recebido:', status); // Log para depuração
      setInterpretationStatus(status);

      if (status.running) {
        if (!interpretationId) {
          console.log('Iniciando mensagem de carregamento no Snackbar');
          showSnackbar(`Interpretação em andamento: ${status.progress}`, 'info', true);
          const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
          setInterpretationId(id);
        } else {
          console.log('Atualizando mensagem de carregamento no Snackbar:', interpretationId);
          updateSnackbar(interpretationId, `Interpretação em andamento: ${status.progress}`, 'info', true);
        }
      } else if (status.progress === 'completed') {
        if (interpretationId) {
          console.log('Interpretação concluída. Atualizando Snackbar e exibindo interpretação.');
          updateSnackbar(interpretationId, 'Interpretação carregada com sucesso!', 'success', false);
          if (status.interpretation) {
            console.log('Definindo interpretação:', status.interpretation);
            setInterpretation(status.interpretation);
            setShowInterpretation(true);
          } else {
            console.warn('Interpretação concluída, mas interpretation é nulo');
            updateSnackbar(interpretationId, 'Erro: Nenhuma interpretação disponível.', 'error', false);
          }
        }
      } else if (status.error) {
        if (interpretationId) {
          console.log('Erro na interpretação:', status.error);
          updateSnackbar(interpretationId, status.error, 'error', false);
        }
      }
    };

    ws.onerror = (error) => console.error('Erro no WebSocket de status de interpretação:', error);

    ws.onclose = () => {
      console.log('WebSocket de status de interpretação desconectado. Tentando reconectar...');
      setTimeout(() => (statusWsRef.current = new WebSocket('ws://localhost:8000/ws/interpret/status')), 1000);
    };

    return () => {
      console.log('Fechando WebSocket de status de interpretação');
      ws.close();
    };
  }, [showSnackbar, updateSnackbar, interpretationId]);

  const limitsStyle = css`
    font-size: 0.6rem;
    color: ${theme.palette.text.secondary};
    margin-top: 2px;
  `;

  const infoIconStyle = css`
    font-size: 1rem;
    color: ${theme.palette.text.secondary};
    margin-left: 4px;
    vertical-align: middle;
  `;

  const legendStyle = css`
    display: flex;
    justify-content: flex-start;
    margin-top: 10px;
    flex-wrap: wrap;
    gap: 10px;
  `;

  const legendItemStyle = css`
    display: flex;
    align-items: center;
    margin-right: 15px;
  `;

  const colorBoxStyle = (color: string) => css`
    width: 16px;
    height: 16px;
    background-color: ${color};
    margin-right: 5px;
    border: 1px solid ${theme.palette.divider};
    border-radius: 3px;
  `;

  const interpretationStyle = css`
    background: ${theme.palette.background.paper};
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    border: 1px solid ${theme.palette.divider};
  `;

  const getCategoryAndColor = (value: number, metric: MetricDefinition) => {
    if (value === undefined) return { category: 'N/A', color: theme.palette.text.primary };

    const range = metric.ranges.find(
      (r) => value >= r.range[0] && value <= r.range[1]
    );
    if (!range) return { category: 'N/A', color: theme.palette.text.primary };

    const { category } = range;
    let color: string;
    switch (category) {
      case 'ruim':
        color = theme.palette.error.main; // Vermelho
        break;
      case 'aceitável':
        color = theme.palette.warning.main; // Amarelo
        break;
      case 'bom':
        color = theme.palette.success.main; // Verde
        break;
      case 'excelente':
        color = theme.palette.info.main; // Azul
        break;
      default:
        color = theme.palette.text.primary;
    }
    return { category, color };
  };

  // Função para iniciar a interpretação
  const fetchInterpretation = async () => {
    try {
      console.log('Iniciando requisição para interpretar métricas');
      await axios.post('http://localhost:8000/interpret-metrics', metrics);
      console.log('Requisição para interpretar métricas enviada com sucesso');
      // O status será atualizado via WebSocket
    } catch (err) {
      console.error('Erro ao iniciar interpretação:', err);
      if (interpretationId) {
        updateSnackbar(interpretationId, 'Erro ao iniciar interpretação. Tente novamente mais tarde.', 'error', false);
      }
    }
  };

  return (
    <Box>
      {/* Interpretação */}
      <Box css={css`margin-bottom: 20px;`}>
        <Button
          variant="outlined"
          onClick={fetchInterpretation}
          endIcon={<ExpandMoreIcon />}
          disabled={interpretationStatus.running}
          css={css`margin-bottom: 10px;`}
        >
          {showInterpretation ? 'Ocultar Interpretação' : 'Obter Interpretação'}
        </Button>
        <Collapse in={showInterpretation}>
          <Box css={interpretationStyle}>
            <Typography variant="body2" component="div">
              {interpretation ? (
                interpretation.split('\n').map((line, index) => (
                  <React.Fragment key={index}>
                    {line}
                    <br />
                  </React.Fragment>
                ))
              ) : (
                'Nenhuma interpretação disponível.'
              )}
            </Typography>
          </Box>
        </Collapse>
      </Box>

      {/* Métricas */}
      <Grid container spacing={2} css={css`margin-top: 2px;`}>
        {metricsConfig.map((metric: MetricDefinition) => {
          const value = metrics[metric.field];
          const displayValue =
            value !== undefined && metric.transform
              ? metric.transform(value)
              : value !== undefined
              ? value.toFixed(metric.significantDigits)
              : "N/A";
          const displayLimits = metric.transform
            ? `Min: ${(metric.ranges[0].range[0] * 100).toFixed(0)}%, Max: ${(metric.ranges[metric.ranges.length - 1].range[1] * 100).toFixed(0)}%`
            : `Min: ${metric.ranges[0].range[0].toFixed(2)}, Max: ${metric.ranges[metric.ranges.length - 1].range[1].toFixed(2)}`;
          const { color } = getCategoryAndColor(value, metric);
          return (
            <React.Fragment key={metric.field}>
              <Grid item xs={6}>
                <Box display="flex" alignItems="center">
                  <Typography color="textPrimary" sx={{ fontSize: "0.7rem" }}>
                    {metric.label}
                  </Typography>
                  <Tooltip title={metric.tooltip} arrow>
                    <IconButton size="small">
                      <InfoIcon css={infoIconStyle} />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography css={limitsStyle}>
                  {displayLimits}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography css={css`color: ${color}; font-weight: 500;`}>
                  {displayValue}
                </Typography>
              </Grid>
            </React.Fragment>
          );
        })}
      </Grid>

      {/* Legenda de cores */}
      <Box css={legendStyle}>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.error.main)} />
          <Typography variant="caption">Ruim</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.warning.main)} />
          <Typography variant="caption">Aceitável</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.success.main)} />
          <Typography variant="caption">Bom</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.info.main)} />
          <Typography variant="caption">Excelente</Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default MetricsDisplay;