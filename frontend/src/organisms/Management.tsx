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

  useEffect(() => {
    // Verifica se o modelo já está treinado ao carregar o componente
    const checkModelStatus = async () => {
      try {
        const response = await axios.get('http://localhost:8000/train/status');
        if (response.data.progress === 'completed') {
          setIsModelTrained(true);
        }
      } catch (error) {
        console.error('Erro ao verificar status do modelo:', error);
      }
    };
    checkModelStatus();
  }, []);

  const startTraining = async () => {
    try {
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
      await axios.post('http://localhost:8000/train', payload);
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
      const response = await axios.get('http://localhost:8000/metrics', {params: {force_recalc: forceRecalc}});
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

  // Notificações de status em tempo real (apenas quando mudam)
  React.useEffect(() => {
    if (trainingStatus.running) {
      showSnackbar(`Treinamento em andamento: ${trainingStatus.progress}`, 'info');
    }
    if (trainingStatus.error) {
      showSnackbar(`Erro no treinamento: ${trainingStatus.error}`, 'error');
    }
    if (metricsStatus.running) {
      showSnackbar(`Cálculo de métricas em andamento: ${metricsStatus.progress}`, 'info');
    }
    if (metricsStatus.error) {
      showSnackbar(`Erro ao calcular métricas: ${metricsStatus.error}`, 'error');
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
                {metrics && (
                  <div css={css`margin-top: 20px;`}>
                    <h3 css={css`color: ${theme.palette.text.primary};
                        font-size: 1.8rem;`}>Métricas de Qualidade</h3>
                    {metrics.error ? (
                      <Typography css={css`color: ${theme.palette.error.main};`}>{metrics.error}</Typography>
                    ) : (
                      <pre css={preStyle}>
                        Precisão@10: {metrics.precision_at_k?.toFixed(4)}<br/>
                        Recall@10: {metrics.recall_at_k?.toFixed(4)}<br/>
                        MRR: {metrics.mrr?.toFixed(4)}<br/>
                        ILS: {metrics.intra_list_similarity?.toFixed(4)}<br/>
                        Cobertura: {(metrics.catalog_coverage * 100)?.toFixed(2)}%
                      </pre>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Documentação da API"/>
              <CardContent>
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
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </div>
    </div>
  );
};

export default Management;