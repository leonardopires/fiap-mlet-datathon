/** @jsxImportSource @emotion/react */
import React, { useState, useEffect, useRef } from 'react';
import { css } from '@emotion/react';
import {
  FormControlLabel,
  Checkbox,
  FormGroup,
  useTheme,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Collapse,
  IconButton,
  Box,
} from '@mui/material';
import axios from 'axios';
import ButtonComponent from '../atoms/Button';
import FormField from '../molecules/FormField';
import DescriptionIcon from '@mui/icons-material/Description';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { useSnackbar } from '../contexts/SnackbarContext';
import Alert from "../atoms/Alert";
import { mapStatus } from "../utils/status_mapping";
import MetricsDisplay from '../organisms/MetricsDisplay';

interface Status {
  running: boolean;
  progress: string;
  error: string | null;
}

interface ManagementProps {
  trainingStatus: Status;
  metricsStatus: Status;
  setTrainingStatus: (status: Status) => void;
  setMetricsStatus: (status: Status) => void;
}

const Management: React.FC<ManagementProps> = ({
  trainingStatus,
  metricsStatus,
  setTrainingStatus,
  setMetricsStatus,
}) => {
  const theme = useTheme();
  const { showSnackbar, updateSnackbar } = useSnackbar();
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRecalc, setForceRecalc] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [isModelTrained, setIsModelTrained] = useState<boolean>(false);
  const [showDocumentation, setShowDocumentation] = useState<boolean>(false);
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [metricsId, setMetricsId] = useState<string | null>(null);
  const [displayedErrors, setDisplayedErrors] = useState<{ training: string | null; metrics: string | null }>({
    training: null,
    metrics: null,
  });
  const hasInitialized = useRef(false); // Ref para evitar múltiplas chamadas

  useEffect(() => {
    if (hasInitialized.current) return; // Impede múltiplas execuções
    hasInitialized.current = true;

    const initialize = async () => {
      try {
        // Verificar status do modelo
        const trainResponse = await axios.get('http://localhost:8000/train/status');
        if (trainResponse.data.progress === 'completed') {
          setIsModelTrained(true);
        }

        // Buscar métricas existentes, sem iniciar cálculo
        showSnackbar('Carregando métricas iniciais...', 'info', true);
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
        setMetricsId(id);

        const metricsResponse = await axios.get('http://localhost:8000/metrics', {
          params: { force_recalc: false, fetch_only_existing: true }
        });
        if (metricsResponse.data.metrics) {
          setMetrics(metricsResponse.data.metrics);
          updateSnackbar(id, 'Métricas iniciais carregadas!', 'success', false);
        } else {
          updateSnackbar(id, 'Nenhuma métrica inicial disponível.', 'info', false);
        }
      } catch (error) {
        console.error('Erro ao inicializar:', error);
        updateSnackbar(metricsId || '', 'Erro ao carregar métricas iniciais.', 'error', false);
      }
    };
    initialize();
  }, []); // Dependências vazias para executar apenas uma vez

  const startTraining = async () => {
    try {
      // Resetar o controle de erros exibidos para treinamento
      setDisplayedErrors((prev) => ({ ...prev, training: null }));

      showSnackbar('Iniciando treinamento...', 'info', true);
      const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
      setTrainingId(id);

      const payload: { subsample_frac?: number; force_reprocess?: boolean; force_retrain?: boolean } = {};
      if (subsampleFrac) {
        const frac = parseFloat(subsampleFrac);
        if (frac > 0 && frac <= 1) payload.subsample_frac = frac;
        else {
          updateSnackbar(id, 'Erro: subsample_frac deve estar entre 0 e 1.', 'error', false);
          return;
        }
      }
      payload.force_retrain = forceRetrain;
      await axios.post('http://localhost:8000/train', payload);
      if (isModelTrained && !forceRetrain) {
        updateSnackbar(id, 'Modelo já treinado; dados existentes foram utilizados.', 'info', false);
      } else {
        updateSnackbar(id, 'Treinamento iniciado com sucesso!', 'success', false);
      }
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      updateSnackbar(trainingId || '', 'Erro ao iniciar treinamento.', 'error', false);
    }
  };

  const fetchMetrics = async () => {
    try {
      // Resetar o controle de erros exibidos para métricas
      setDisplayedErrors((prev) => ({ ...prev, metrics: null }));

      showSnackbar('Carregando métricas...', 'info', true);
      const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
      setMetricsId(id);

      const response = await axios.get('http://localhost:8000/metrics', { params: { force_recalc: forceRecalc } });
      setMetrics(response.data.metrics || null);
      updateSnackbar(id, 'Métricas carregadas com sucesso!', 'success', false);
    } catch (error) {
      console.error('Erro ao obter métricas:', error);
      updateSnackbar(metricsId || '', 'Erro ao carregar métricas.', 'error', false);
    }
  };

  const containerStyle = css`
    padding: 20px;
  `;

  const preStyle = css`
    background: ${theme.palette.background.paper};
    padding: 10px;
    border-radius: 5px;
    color: ${theme.palette.text.primary};
  `;

  const checkboxStyle = css`
    color: ${theme.palette.text.primary};
    .MuiCheckbox-root {
      color: ${theme.palette.text.primary};
    }
  `;

  React.useEffect(() => {
    if (trainingStatus.running) {
      if (!trainingId) {
        showSnackbar(`Treinamento em andamento: ${mapStatus(trainingStatus.progress)}`, 'info', true);
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
        setTrainingId(id);
      } else {
        updateSnackbar(trainingId, `Treinamento em andamento: ${mapStatus(trainingStatus.progress)}`, 'info', true);
      }
    }
    if (trainingStatus.error && trainingStatus.error !== displayedErrors.training) {
      if (trainingId) {
        updateSnackbar(trainingId, `Erro no treinamento: ${mapStatus(trainingStatus.error)}`, 'error', false);
        setDisplayedErrors((prev) => ({ ...prev, training: trainingStatus.error }));
      }
    }
    if (metricsStatus.running) {
      if (!metricsId) {
        showSnackbar(`Cálculo de métricas em andamento: ${mapStatus(metricsStatus.progress)}`, 'info', true);
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
        setMetricsId(id);
      } else {
        updateSnackbar(metricsId, `Cálculo de métricas em andamento: ${mapStatus(metricsStatus.progress)}`, 'info', true);
      }
    }
    if (metricsStatus.error && metricsStatus.error !== displayedErrors.metrics) {
      if (metricsId) {
        updateSnackbar(metricsId, `Erro ao calcular métricas: ${mapStatus(metricsStatus.error)}`, 'error', false);
        setDisplayedErrors((prev) => ({ ...prev, metrics: metricsStatus.error }));
      }
    }
  }, [trainingStatus, metricsStatus, showSnackbar, updateSnackbar, trainingId, metricsId, displayedErrors]);

  return (
    <div css={containerStyle}>
      <h1 css={css`color: ${theme.palette.text.primary}; font-size: 2.8rem; margin-bottom: 20px;`}>Gerenciamento</h1>
      <div>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Treinamento do Modelo" />
              <CardContent>
                {isModelTrained && !forceRetrain && (
                  <Alert variant="info" css={css`margin-bottom: 15px;`}>
                    Modelo já treinado e pronto para uso. Clique em "Iniciar Treinamento" para usar os dados existentes
                    ou marque "Forçar Novo Treinamento" para re-treinar.
                  </Alert>
                )}
                <FormGroup css={css`margin-bottom: 15px;`}>
                  <FormField
                    id="subsampleFrac"
                    label="Fração de Subamostragem (0 a 1, opcional)"
                    value={subsampleFrac}
                    onChange={(e) => setSubsampleFrac(e.target.value)}
                    placeholder="Ex.: 0.1 para 10% dos dados"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={forceRetrain} onChange={(e) => setForceRetrain(e.target.checked)} />}
                    label={
                      <span css={checkboxStyle}>
                        <DescriptionIcon css={css`margin-right: 8px; vertical-align: middle;`} /> Forçar Novo Treinamento
                      </span>
                    }
                  />
                  <ButtonComponent
                    variant="primary"
                    onClick={startTraining}
                    icon={<PlayArrowIcon />}
                    disabled={trainingStatus.running}
                  >
                    {trainingStatus.running ? 'Treinando...' : 'Iniciar Treinamento'}
                  </ButtonComponent>
                </FormGroup>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Métricas de Qualidade" />
              <CardContent>
                {/* Exibir métricas primeiro, se existirem */}
                {metrics && (
                  <div css={css`margin-bottom: 50px;`}>
                    {metrics.error ? (
                      <Typography css={css`color: ${theme.palette.error.main};`}>{metrics.error}</Typography>
                    ) : (
                      <MetricsDisplay metrics={metrics} />
                    )}
                  </div>
                )}
                {/* Controles para obter métricas */}
                <FormGroup css={css`margin-bottom: 15px;`}>
                  <FormControlLabel
                    control={<Checkbox checked={forceRecalc} onChange={(e) => setForceRecalc(e.target.checked)} />}
                    label={
                      <Typography css={checkboxStyle}>
                        <DescriptionIcon css={css`margin-right: 8px; vertical-align: middle;`} /> Forçar Recálculo de Métricas
                      </Typography>
                    }
                  />
                  <ButtonComponent
                    variant="info"
                    onClick={fetchMetrics}
                    icon={<PlayArrowIcon />}
                    disabled={metricsStatus.running}
                    css={css`margin-left: 10px;`}
                  >
                    {metricsStatus.running ? 'Calculando...' : 'Obter Métricas'}
                  </ButtonComponent>
                </FormGroup>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardHeader
                title={
                  <Box display="flex" alignItems="center">
                    <Typography variant="h6">Documentação da API</Typography>
                    <IconButton onClick={() => setShowDocumentation(!showDocumentation)}>
                      {showDocumentation ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                }
                css={css`padding-bottom: ${showDocumentation ? '16px' : '0'};`}
              />
              <Collapse in={showDocumentation}>
                <CardContent>
                  {showDocumentation && (
                    <iframe
                      src="http://localhost:8000/docs"
                      title="Swagger UI"
                      css={css`
                        width: 100%;
                        height: 600px;
                        border: none;
                        border-radius: 5px;
                        background: #ffffff;
                      `}
                    />
                  )}
                </CardContent>
              </Collapse>
            </Card>
          </Grid>
        </Grid>
      </div>
    </div>
  );
};

export default Management;