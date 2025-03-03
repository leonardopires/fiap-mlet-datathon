/** @jsxImportSource @emotion/react */
import React, {useState, useEffect} from 'react';
import {css} from '@emotion/react';
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
} from '@mui/material';
import axios from 'axios';
import Button from '../atoms/Button';
import FormField from '../molecules/FormField';
import DescriptionIcon from '@mui/icons-material/Description';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {useSnackbar} from '../contexts/SnackbarContext';
import Alert from "../atoms/Alert";
import {mapStatus} from "../utils/status_mapping";
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
  const {showSnackbar} = useSnackbar();
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRecalc, setForceRecalc] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [isModelTrained, setIsModelTrained] = useState<boolean>(false);

  const updateMetrics = async () => {
    const metricsResponse = await axios.get(`http://${window.location.hostname}:8000/metrics`, {
      params: {force_recalc: false, fetch_only_existing: true}
    });
    if (metricsResponse.data.metrics) {
      setMetrics(metricsResponse.data.metrics);
    }
  };

  useEffect(() => {
    // Verifica se o modelo já está treinado e busca métricas existentes ao carregar o componente
    const initialize = async () => {
      try {
        // Verificar status do modelo
        const trainResponse = await axios.get(`http://${window.location.hostname}:8000/train/status`);
        if (trainResponse.data.progress === 'completed') {
          setIsModelTrained(true);
        }

        // Buscar métricas existentes, sem iniciar cálculo
        await updateMetrics();
      } catch (error) {
        console.error('Erro ao inicializar:', error);
      }
    };
    initialize();
  }, [metricsStatus.running, trainingStatus.running]);

  function clearErrors() {
      setMetricsStatus({...metricsStatus, error: null});
      setTrainingStatus({...trainingStatus, error: null});
  }

  const startTraining = async () => {
    try {
      clearErrors();

      const payload: { subsample_frac?: number; force_reprocess?: boolean; force_retrain?: boolean } = {};
      if (subsampleFrac) {
        const frac = parseFloat(subsampleFrac);
        if (frac > 0 && frac <= 1) payload.subsample_frac = frac;
        else {
          showSnackbar('Erro: subsample_frac deve estar entre 0 e 1.', 'error');
          return;
        }
      }
      payload.force_retrain = forceRetrain;
      await axios.post(`http://${window.location.hostname}:8000/train`, payload);
      if (isModelTrained && !forceRetrain) {
        showSnackbar('Modelo já treinado; dados existentes foram utilizados.', 'info');
      } else {
        showSnackbar('Treinamento iniciado com sucesso!', 'success');
      }
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      showSnackbar('Erro ao iniciar treinamento.', 'error');
    }
  };

  const fetchMetrics = async () => {
    try {
      clearErrors();

      const response = await axios.get(`http://${window.location.hostname}:8000/metrics`, {params: {force_recalc: forceRecalc}});
      setMetrics(response.data.metrics || null);
      showSnackbar('Métricas carregadas com sucesso!', 'success');
    } catch (error) {
      console.error('Erro ao obter métricas:', error);
      setMetrics({error: 'Erro ao carregar métricas do servidor.'});
      showSnackbar('Erro ao carregar métricas.', 'error');
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
      showSnackbar(`Treinamento em andamento: ${mapStatus(trainingStatus.progress)}`, 'info');
    } else if (trainingStatus.error) {
      showSnackbar(`Erro no treinamento: ${mapStatus(trainingStatus.error)}`, 'error');
    }

    if (metricsStatus.running) {
      showSnackbar(`Cálculo de métricas em andamento: ${mapStatus(metricsStatus.progress)}`, 'info');
    } else if (metricsStatus.error && !trainingStatus.running) {
      showSnackbar(`Erro ao calcular métricas: ${mapStatus(metricsStatus.error)}`, 'error');
    }
  }, [trainingStatus, metricsStatus, showSnackbar]);

  return (
    <div css={containerStyle}>
      <h1 css={css`color: ${theme.palette.text.primary};
          font-size: 2.8rem;
          margin-bottom: 20px;`}>Gerenciamento</h1>
      <div>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Treinamento do Modelo"/>
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
                    control={<Checkbox checked={forceRetrain} onChange={(e) => setForceRetrain(e.target.checked)}/>}
                    label={
                      <span css={checkboxStyle}>
                        <DescriptionIcon css={css`margin-right: 8px;
                            vertical-align: middle;`}/> Forçar Novo Treinamento
                      </span>
                    }
                  />
                  <Button
                    variant="primary"
                    onClick={startTraining}
                    icon={<PlayArrowIcon/>}
                    disabled={trainingStatus.running}
                  >
                    {trainingStatus.running ? 'Treinando...' : 'Iniciar Treinamento'}
                  </Button>
                </FormGroup>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Métricas de Qualidade"/>
              <CardContent>
                {/* Exibir métricas primeiro, se existirem */}
                {metrics && (
                  <div css={css`margin-bottom: 50px;`}>
                    {metrics.error ? (
                      <Typography css={css`color: ${theme.palette.error.main};`}>{metrics.error}</Typography>
                    ) : (
                      <MetricsDisplay metrics={metrics}/>
                    )}
                  </div>
                )}
                {/* Controles para obter métricas */}
                <FormGroup css={css`margin-bottom: 15px;`}>
                  <FormControlLabel
                    control={<Checkbox checked={forceRecalc} onChange={(e) => setForceRecalc(e.target.checked)}/>}
                    label={
                      <Typography css={checkboxStyle}>
                        <DescriptionIcon css={css`margin-right: 8px;
                            vertical-align: middle;`}/> Forçar Recálculo de Métricas
                      </Typography>
                    }
                  />
                  <Button
                    variant="info"
                    onClick={fetchMetrics}
                    icon={<PlayArrowIcon/>}
                    disabled={metricsStatus.running}
                    css={css`margin-left: 10px;`}
                  >
                    {metricsStatus.running ? 'Calculando...' : 'Obter Métricas'}
                  </Button>
                </FormGroup>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Documentação da API"/>
              <CardContent>
                <iframe
                  src={`http://${window.location.hostname}:8000/docs`}
                  title="Swagger UI"
                  css={css`
                      width: 100%;
                      height: 600px;
                      border: none;
                      border-radius: 5px;
                      background: #ffffff;
                  `}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </div>
    </div>
  );
};

export default Management;