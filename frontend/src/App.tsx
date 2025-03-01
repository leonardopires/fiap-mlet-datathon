import React, { useState, useEffect, useRef } from 'react';
import { Container, Form, Button, Card, Alert, Nav } from 'react-bootstrap';
import axios from 'axios';
import DescriptionIcon from '@mui/icons-material/Description';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import SettingsIcon from '@mui/icons-material/Settings';

interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string;
}

interface Status {
  running: boolean;
  progress: string;
  error: string | null;
}

const App: React.FC = () => {
  const [activeKey, setActiveKey] = useState<string>('recommendations');
  const [userId, setUserId] = useState<string>('');
  const [keywords, setKeywords] = useState<string>('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRecalc, setForceRecalc] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<any>(null);
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [logsVisible, setLogsVisible] = useState<boolean>(true);
  const [trainingStatus, setTrainingStatus] = useState<Status>({ running: false, progress: "idle", error: null });
  const [metricsStatus, setMetricsStatus] = useState<Status>({ running: false, progress: "idle", error: null });

  const logsRef = useRef<HTMLDivElement>(null);
  const logsWsRef = useRef<WebSocket | null>(null);
  const statusWsRef = useRef<WebSocket | null>(null);

  interface PredictPayload {
    user_id: string;
    keywords?: string[];
  }

  const fetchRecommendations = async () => {
    try {
      const payload: PredictPayload = { user_id: userId };
      if (keywords) {
        payload.keywords = keywords.split(',').map(kw => kw.trim());
      }
      const response = await axios.post('http://localhost:8000/predict', payload);
      setRecommendations(response.data.acessos_futuros);
      setErrorMessage('');
      console.log('Resposta da API:', response.data);
    } catch (error: any) {
      console.error('Erro ao obter recomendações:', error);
      setErrorMessage(error.response?.status === 400 ? '⚠️ O modelo ainda não foi treinado.' : 'Erro ao carregar recomendações.');
      setRecommendations([]);
    }
  };

  const startTraining = async () => {
    try {
      const payload: { subsample_frac?: number; force_reprocess?: boolean; force_retrain?: boolean } = {};
      if (subsampleFrac) {
        const frac = parseFloat(subsampleFrac);
        if (frac > 0 && frac <= 1) payload.subsample_frac = frac;
        else { alert('Erro: subsample_frac deve estar entre 0 e 1.'); return; }
      }
      payload.force_retrain = forceRetrain;
      const response = await axios.post('http://localhost:8000/train', payload);
      console.log('Treinamento iniciado:', response.data);
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      alert('Erro ao iniciar treinamento.');
    }
  };

  const logInteraction = async (page: string) => {
    try {
      const interaction = { user_id: userId, page, clicks: 1, time_on_page: 10000, scroll_percentage: 50, timestamp: Date.now() };
      await axios.post('http://localhost:8000/log_interaction', interaction);
      console.log('Interação registrada com sucesso');
    } catch (error) {
      console.error('Erro ao registrar interação:', error);
    }
  };

  const connectLogsWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/logs');
    logsWsRef.current = ws;
    ws.onopen = () => console.log('WebSocket de logs conectado');
    ws.onmessage = (event) => {
      setLogs((prevLogs) => [...prevLogs, event.data].slice(-2000));
    };
    ws.onerror = (error) => console.error('Erro no WebSocket de logs:', error);
    ws.onclose = () => {
      console.log('WebSocket de logs desconectado. Tentando reconectar...');
      setTimeout(connectLogsWebSocket, 1000);
    };
  };

  const connectStatusWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/status');
    statusWsRef.current = ws;
    ws.onopen = () => console.log('WebSocket de status conectado');
    ws.onmessage = (event) => {
      const status = JSON.parse(event.data);
      setTrainingStatus(status.training);
      setMetricsStatus(status.metrics);
      if (status.metrics.progress === "completed" && !forceRecalc) {
        fetchMetrics(); // Atualiza métricas quando concluído
      }
    };
    ws.onerror = (error) => console.error('Erro no WebSocket de status:', error);
    ws.onclose = () => {
      console.log('WebSocket de status desconectado. Tentando reconectar...');
      setTimeout(connectStatusWebSocket, 1000);
    };
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:8000/metrics', { params: { force_recalc: forceRecalc } });
      setMetrics(response.data.metrics || null);
      console.log('Métricas recebidas:', response.data);
    } catch (error: any) {
      console.error('Erro ao obter métricas:', error);
      setMetrics({ error: 'Erro ao carregar métricas do servidor.' });
    }
  };

  useEffect(() => {
    connectLogsWebSocket();
    connectStatusWebSocket();
    return () => {
      if (logsWsRef.current) logsWsRef.current.close();
      if (statusWsRef.current) statusWsRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  function extractDate(rec: Recommendation): string {
    const pattern = /\/noticia\/(\d{4}\/\d{2}\/\d{2})\//;
    const match = rec.link.match(pattern);
    if (match && match[1]) {
      const [year, month, day] = match[1].split('/');
      return `${day}/${month}/${year}`;
    }
    return 'Data não disponível';
  }

  return (
    <div className="d-flex vh-100 app-container bg-black">
      <div className="bg-dark sidebar-left" style={{ width: '60px', background: '#163747', paddingTop: '10px' }}>
        <Nav variant="pills" className="flex-column">
          <Nav.Item className="mb-2">
            <Nav.Link
              eventKey="recommendations"
              onClick={() => setActiveKey('recommendations')}
              className="text-white icon-nav d-flex align-items-center justify-content-center"
              data-label="Recomendações"
              style={{
                transition: 'background-color 0.3s ease, color 0.3s ease',
                transitionDelay: '0.2s',
                backgroundColor: activeKey === 'recommendations' ? '#2a4d62' : 'transparent',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#2a4d62'; e.currentTarget.style.color = '#FFFFFF'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = activeKey === 'recommendations' ? '#2a4d62' : 'transparent'; e.currentTarget.style.color = '#FFFFFF'; }}
            >
              <NewspaperIcon />
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="mb-2">
            <Nav.Link
              eventKey="management"
              onClick={() => setActiveKey('management')}
              className="text-white icon-nav d-flex align-items-center justify-content-center"
              data-label="Gerenciamento"
              style={{
                transition: 'background-color 0.3s ease, color 0.3s ease',
                transitionDelay: '0.2s',
                backgroundColor: activeKey === 'management' ? '#2a4d62' : 'transparent',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#2a4d62'; e.currentTarget.style.color = '#FFFFFF'; }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = activeKey === 'management' ? '#2a4d62' : 'transparent'; e.currentTarget.style.color = '#FFFFFF'; }}
            >
              <SettingsIcon />
            </Nav.Link>
          </Nav.Item>
        </Nav>
      </div>

      <Container fluid className="flex-grow-1 p-4 main-content" style={{ background: 'none' }}>
        <div style={{ marginTop: '10px', textAlign: 'right' }}>
          <a href="https://www.fiap.com.br/" target="_blank" rel="noopener noreferrer">
            <img src="https://postech.fiap.com.br/svg/fiap-plus-alura.svg" alt="FIAP + Alura" className="logo-img" style={{ width: '150px', height: 'auto' }} />
          </a>
          <div className="project-info mb-5">
            <h2 className="text-white mb-0" style={{ fontSize: '1.5rem', fontWeight: 600, margin: 0 }}>
              ML TECH DATATHON - Fase Final - Engenharia em Machine Learning - 2025
            </h2>
            <p className="text-white mb-0" style={{ fontSize: '1rem', fontWeight: 500 }}>
              Entrega para a etapa final do curso Datathon 2025
            </p>
            <p className="text-white" style={{ fontSize: '0.9rem', fontWeight: 400 }}>
              Membros do Grupo:
              <br />- Leonardo T Pires: <a href="https://github.com/leonardopires" target="_blank" rel="noopener noreferrer" className="text-white">RM355401</a>
              <br />- Felipe de Paula G.: <a href="https://github.com/Felipe-DePaula" target="_blank" rel="noopener noreferrer" className="text-white">RM355402</a>
              <br />- Jorge Guilherme D. W: <a href="https://github.com/JorgeWald" target="_blank" rel="noopener noreferrer" className="text-white">RM355849</a>
            </p>
          </div>
        </div>

        <h1 className="text-white mb-4 title">{activeKey === 'recommendations' ? 'Recomendações' : 'Gerenciamento'}</h1>

        {activeKey === 'recommendations' && (
          <div>
            <Form>
              <Form.Group controlId="userId" className="mb-3">
                <Form.Label className="text-white form-label d-flex align-items-center">
                  <DescriptionIcon className="me-2" /> ID do Usuário (UUID)
                </Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Digite o UUID do usuário"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="form-control"
                  style={{ maxWidth: '400px', transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                  onClick={() => logInteraction(userId)}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#444444'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
                />
              </Form.Group>
              <Form.Group controlId="keywords" className="mb-3">
                <Form.Label className="text-white form-label d-flex align-items-center">
                  <DescriptionIcon className="me-2" /> Palavras-Chave (separadas por vírgula, opcional)
                </Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Ex.: esportes, tecnologia"
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  className="form-control"
                  style={{ maxWidth: '400px', transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                  onClick={() => logInteraction(keywords)}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#444444'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
                />
                <Form.Text className="text-white form-text">
                  Insira palavras-chave para personalizar recomendações iniciais.
                </Form.Text>
              </Form.Group>
              <Button
                variant="primary"
                onClick={fetchRecommendations}
                className="btn-primary mt-2 d-flex align-items-center"
                style={{ transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#0056b3'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
              >
                <PlayArrowIcon className="me-2" /> Obter Recomendações
              </Button>
            </Form>

            {errorMessage && (
              <Alert variant="danger" className="mt-3 alert-danger">
                {errorMessage}
              </Alert>
            )}

            <div className="mt-4">
              <h3 className="text-white subtitle">Recomendações</h3>
              {recommendations.length > 0 ? (
                <div className="row">
                  {recommendations.map((rec, index) => (
                    <div className="col-md-4 mb-3" key={rec.page}>
                      <Card className="card border-dark shadow-dark bg-dark">
                        <Card.Body>
                          <Card.Title className="text-white card-title">{rec.title}</Card.Title>
                          <Card.Text className="text-white card-text">
                            <strong>ID:</strong> {rec.page}<br />
                            <strong>Data:</strong> {extractDate(rec)}<br />
                            <strong>Link:</strong> {rec.link !== 'N/A' ? (
                              <a
                                href={rec.link}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={() => logInteraction(rec.page)}
                                className="text-white"
                                style={{ transition: 'color 0.3s ease', transitionDelay: '0.2s' }}
                                onMouseEnter={(e) => e.currentTarget.style.color = '#66b0ff'}
                                onMouseLeave={(e) => e.currentTarget.style.color = '#FFFFFF'}
                              >
                                {rec.link}
                              </a>
                            ) : 'Não disponível'}
                          </Card.Text>
                        </Card.Body>
                      </Card>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-white text-no-recommendations">Nenhuma recomendação carregada ainda.</p>
              )}
            </div>
          </div>
        )}

        {activeKey === 'management' && (
          <div>
            <Form>
              <Form.Group controlId="subsampleFrac" className="mb-3">
                <Form.Label className="text-white form-label d-flex align-items-center">
                  <DescriptionIcon className="me-2" /> Fração de Subamostragem (0 a 1, opcional)
                </Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  placeholder="Ex.: 0.1 para 10% dos dados"
                  value={subsampleFrac}
                  onChange={(e) => setSubsampleFrac(e.target.value)}
                  className="form-control"
                  style={{ maxWidth: '400px', transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                  onClick={() => logInteraction(subsampleFrac)}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#444444'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
                />
              </Form.Group>
              <Form.Group controlId="forceRetrain" className="mb-3">
                <Form.Check
                  type="checkbox"
                  label={<span className="text-white d-flex align-items-center"><DescriptionIcon className="me-2" /> Forçar Novo Treinamento</span>}
                  checked={forceRetrain}
                  onChange={(e) => setForceRetrain(e.target.checked)}
                  className="text-white"
                  onClick={() => logInteraction('forceRetrain')}
                />
              </Form.Group>
              <Form.Group controlId="forceRecalc" className="mb-3">
                <Form.Check
                  type="checkbox"
                  label={<span className="text-white d-flex align-items-center"><DescriptionIcon className="me-2" /> Forçar Recálculo de Métricas</span>}
                  checked={forceRecalc}
                  onChange={(e) => setForceRecalc(e.target.checked)}
                  className="text-white"
                />
              </Form.Group>
              <Button
                variant="primary"
                onClick={startTraining}
                className="btn-primary mt-2 d-flex align-items-center"
                disabled={trainingStatus.running}
                style={{ transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#0056b3'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
              >
                <PlayArrowIcon className="me-2" /> {trainingStatus.running ? 'Treinando...' : 'Iniciar Treinamento'}
              </Button>
              <Button
                variant="info"
                onClick={fetchMetrics}
                className="btn-info mt-2 ms-2 d-flex align-items-center"
                disabled={metricsStatus.running}
                style={{ transition: 'background-color 0.3s ease, border-color 0.3s ease', transitionDelay: '0.2s' }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#145523'; e.currentTarget.style.borderColor = '#FFFFFF'; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ''; e.currentTarget.style.borderColor = ''; }}
              >
                <PlayArrowIcon className="me-2" /> {metricsStatus.running ? 'Calculando...' : 'Obter Métricas'}
              </Button>
              {trainingStatus.running && (
                <Alert variant="info" className="mt-3">
                  Treinamento em andamento: {trainingStatus.progress}
                </Alert>
              )}
              {trainingStatus.error && (
                <Alert variant="danger" className="mt-3">
                  Erro no treinamento: {trainingStatus.error}
                </Alert>
              )}
              {metricsStatus.running && (
                <Alert variant="info" className="mt-3">
                  Cálculo de métricas em andamento: {metricsStatus.progress}
                </Alert>
              )}
              {metricsStatus.error && (
                <Alert variant="danger" className="mt-3">
                  Erro ao calcular métricas: {metricsStatus.error}
                </Alert>
              )}
              {metrics && (
                <div className="mt-4">
                  <h3 className="text-white subtitle">Métricas de Qualidade</h3>
                  {metrics.error ? (
                    <Alert variant="danger">{metrics.error}</Alert>
                  ) : (
                    <pre className="text-white" style={{ background: '#2d2d2d', padding: '10px', borderRadius: '5px' }}>
                      Precisão@10: {metrics.precision_at_k?.toFixed(4)}<br />
                      Recall@10: {metrics.recall_at_k?.toFixed(4)}<br />
                      MRR: {metrics.mrr?.toFixed(4)}<br />
                      ILS: {metrics.intra_list_similarity?.toFixed(4)}<br />
                      Cobertura: {(metrics.catalog_coverage * 100)?.toFixed(2)}%
                    </pre>
                  )}
                </div>
              )}
              <div className="mt-4">
                <iframe
                  src="http://localhost:8000/docs"
                  title="Swagger UI"
                  className="swagger-iframe"
                  style={{ width: '100%', height: '600px', border: 'none', borderRadius: '5px', background: '#FFFFFF' }}
                />
              </div>
            </Form>
          </div>
        )}
      </Container>

      {logsVisible && (
        <div className="bg-dark sidebar-bottom" style={{ position: 'fixed', bottom: 0, left: 0, right: 0, height: '20%', zIndex: 1000, boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.5)', background: '#000000' }}>
          <div className="d-flex justify-content-between align-items-center">
            <h5 className="mb-2 text-white logs-title">Logs do Servidor</h5>
            <Button variant="secondary" size="sm" onClick={() => setLogsVisible(false)} style={{ marginRight: '10px' }}>
              Ocultar
            </Button>
          </div>
          {logs.length > 0 ? (
            <div ref={logsRef} className="logs-content" style={{ maxHeight: '80%', overflowY: 'auto', borderTop: '1px solid #444', padding: '10px' }}>
              {logs.map((log, index) => (
                <p key={index} className="log-entry text-white" style={{ margin: '0', whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>{log}</p>
              ))}
            </div>
          ) : (
            <p className="text-white no-logs">Nenhum log disponível.</p>
          )}
        </div>
      )}
      {!logsVisible && (
        <Button variant="secondary" onClick={() => setLogsVisible(true)} style={{ position: 'fixed', bottom: '10px', right: '10px', zIndex: 1000 }}>
          Mostrar Logs
        </Button>
      )}
    </div>
  );
};

export default App;