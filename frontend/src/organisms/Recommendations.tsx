/** @jsxImportSource @emotion/react */
import React, {useState, useEffect, useRef} from 'react';
import {css} from '@emotion/react';
import axios from 'axios';
import Button from '../atoms/Button';
import FormField from '../molecules/FormField';
import RecommendationCard from '../molecules/RecommendationCard';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {Box, useTheme, CircularProgress, Typography, Grid} from '@mui/material';
import {useSnackbar} from '../contexts/SnackbarContext'; // Importe o hook

interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string;
}

interface PredictionStatus {
  running: boolean;
  progress: string;
  error: string | null;
}

const Recommendations: React.FC = () => {
  const theme = useTheme();
  const {showSnackbar} = useSnackbar(); // Use o hook
  const [userId, setUserId] = useState('');
  const [keywords, setKeywords] = useState('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionStatus, setPredictionStatus] = useState<PredictionStatus>({
    running: false,
    progress: 'idle',
    error: null
  });
  const statusWsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/predict/status`);
    statusWsRef.current = ws;
    ws.onopen = () => console.log('WebSocket de status de predição conectado');
    ws.onmessage = (event) => {
      const status = JSON.parse(event.data);
      setPredictionStatus(status);
      if (status.progress === 'completed') {
        showSnackbar('Recomendações geradas com sucesso!', 'success');
      } else if (status.error) {
        showSnackbar(status.error, 'error');
        setIsPredicting(false);
      }
    };
    ws.onerror = (error) => console.error('Erro no WebSocket de status de predição:', error);
    ws.onclose = () => {
      console.log('WebSocket de status de predição desconectado. Tentando reconectar...');
      setTimeout(() => (statusWsRef.current = new WebSocket(`ws://${window.location.hostname}:8000/ws/predict/status`)), 1000);
    };
    return () => ws.close();
  }, [showSnackbar]);

  const fetchRecommendations = async () => {
    try {
      setIsPredicting(true);
      setPredictionStatus({running: true, progress: 'starting', error: null});

      const payload: { user_id: string; keywords?: string[] } = {user_id: userId};
      if (keywords) {
        payload.keywords = keywords.split(',').map(kw => kw.trim());
      }
      const response = await axios.post(`http://${window.location.hostname}:8000/predict_foreground`, payload);
      const urlSet = new Set();
      let acessosFuturos = (response.data.acessos_futuros || []).filter((rec: Recommendation) => {
        const exists = urlSet.has(rec.link);
        urlSet.add(rec.link);
        return !exists;
      });
      // limita o tamanho do array a 10
      acessosFuturos = acessosFuturos.slice(0, Math.min(10, acessosFuturos.length));
      setRecommendations(acessosFuturos);
      console.log('Resposta da API:', response.data);
      setPredictionStatus({running: false, progress: 'completed', error: null});
    } catch (error: any) {
      console.error('Erro ao obter recomendações:', error);
      const errorMsg = error.response?.status === 400 ? '⚠️ O modelo ainda não foi treinado.' : 'Erro ao carregar recomendações.';
      showSnackbar(errorMsg, 'error'); // Substitui setErrorMessage
      setPredictionStatus({running: false, progress: 'failed', error: errorMsg});
      setRecommendations([]);
    } finally {
      setIsPredicting(false);
    }
  };

  const logInteraction = async (page: string) => {
    try {
      const interaction = {
        user_id: userId,
        page,
        clicks: 1,
        time_on_page: 10000,
        scroll_percentage: 50,
        timestamp: Date.now(),
      };
      await axios.post(`http://${window.location.hostname}:8000/log_interaction`, interaction);
      showSnackbar('Interação registrada com sucesso!', 'success'); // Notificação de sucesso
    } catch (error) {
      console.error('Erro ao registrar interação:', error);
      showSnackbar('Erro ao registrar interação.', 'error');
    }
  };

  const containerStyle = css`
      padding: 20px;
  `;

  return (
    <div css={containerStyle}>
      <h1 css={css`color: ${theme.palette.text.primary};
          font-size: 2.8rem;
          margin-bottom: 20px;`}>Recomendações</h1>
      <div>
        <Box>
          <FormField
            id="userId"
            label="ID do Usuário (UUID)"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Digite o UUID do usuário"
          />
        </Box>
        <Box>
          <FormField
            id="keywords"
            label="Palavras-Chave (separadas por vírgula, opcional)"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="Ex.: esportes, tecnologia"
            helpText="Insira palavras-chave para personalizar recomendações iniciais."
          />
        </Box>
        <Box css={css`display: flex;
            align-items: center;
            gap: 10px;`}>
          <Button
            variant="primary"
            onClick={fetchRecommendations}
            icon={<PlayArrowIcon/>}
            disabled={isPredicting || predictionStatus.running}
          >
            Obter Recomendações
          </Button>
          {isPredicting && <CircularProgress size={24}/>}
        </Box>
        {(predictionStatus.running || predictionStatus.error) && (
          <Typography css={css`color: ${theme.palette.text.secondary};
              margin-top: 10px;`}>
            Status da Predição: {predictionStatus.progress}
            {predictionStatus.error && ` - Erro: ${predictionStatus.error}`}
          </Typography>
        )}
      </div>
      <div css={css`margin-top: 20px;`}>
        <h3 css={css`color: ${theme.palette.text.primary};
            font-size: 1.8rem;`}>Recomendações</h3>
        {recommendations && recommendations.length > 0 ? (
          <Grid container spacing={2}>
            {recommendations.map((rec) => (
              <Grid item xs={12} lg={6} key={rec.page}>
                <RecommendationCard recommendation={rec} onLinkClick={logInteraction}/>
              </Grid>
            ))}
          </Grid>
        ) : (
          <p css={css`color: ${theme.palette.text.secondary};`}>Nenhuma recomendação carregada ainda.</p>
        )}
      </div>
    </div>
  );
};

export default Recommendations;