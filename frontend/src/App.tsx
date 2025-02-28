import React, { useState, useEffect, useRef } from 'react';
import { Container, Form, Button, Card, Alert, Nav } from 'react-bootstrap';
import axios from 'axios';
import NewspaperIcon from '@mui/icons-material/Newspaper'; // Ícone para Recomendações
import SettingsIcon from '@mui/icons-material/Settings'; // Ícone para Gerenciamento
import TerminalIcon from '@mui/icons-material/Terminal'; // Ícone para Logs (opcional)

interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string;
}

const App: React.FC = () => {
  const [userId, setUserId] = useState<string>('');
  const [keywords, setKeywords] = useState<string>('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [trainStatus, setTrainStatus] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [activeKey, setActiveKey] = useState<string>('recommendations');

  const logsRef = useRef<HTMLDivElement>(null);

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
    } catch (error: any) {
      console.error('Erro ao obter recomendações:', error);
      if (error.response && error.response.status === 400) {
        setErrorMessage('⚠️ O modelo ainda não foi treinado. Tente novamente após o treinamento.');
      } else {
        setErrorMessage('Erro ao carregar recomendações.');
      }
      setRecommendations([]);
    }
  };

  const startTraining = async () => {
    try {
      const payload: { subsample_frac?: number; force_reprocess?: boolean; force_retrain?: boolean } = {};
      if (subsampleFrac) {
        const frac = parseFloat(subsampleFrac);
        if (frac > 0 && frac <= 1) {
          payload.subsample_frac = frac;
        } else {
          setTrainStatus('Erro: subsample_frac deve estar entre 0 e 1.');
          return;
        }
      }
      payload.force_retrain = forceRetrain;
      const response = await axios.post('http://localhost:8000/train', payload);
      setTrainStatus(response.data.message);
      fetchLogs();
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      setTrainStatus('Erro ao iniciar treinamento.');
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
        timestamp: Date.now()
      };
      const response = await axios.post('http://localhost:8000/log_interaction', interaction);
      console.log(response.data.message);
    } catch (error) {
      console.error('Erro ao registrar interação:', error);
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await axios.get('http://localhost:8000/logs');
      console.log('Resposta completa dos logs:', response.data.logs);
      const allLogs = response.data.logs || [];
      const recentLogs = allLogs.slice(-2000);
      setLogs(recentLogs);
    } catch (error: unknown) {
      console.error('Erro ao obter logs:', error instanceof Error ? error.message : 'Erro desconhecido');
      setLogs(['Erro ao carregar logs do servidor.']);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="d-flex vh-100 app-container">
      {/* Sidebar de ícones no canto esquerdo */}
      <div className="bg-primary sidebar-left">
        <Nav variant="pills" className="flex-column">
          <Nav.Item className="mb-2">
            <Nav.Link
              eventKey="recommendations"
              onClick={() => setActiveKey('recommendations')}
              className="text-white icon-nav"
            >
              <NewspaperIcon />
            </Nav.Link>
          </Nav.Item>
          <Nav.Item className="mb-2">
            <Nav.Link
              eventKey="management"
              onClick={() => setActiveKey('management')}
              className="text-white icon-nav"
            >
              <SettingsIcon />
            </Nav.Link>
          </Nav.Item>
        </Nav>
      </div>

      {/* Conteúdo principal */}
      <Container fluid className="flex-grow-1 p-4 main-content">
        <h1 className="text-primary mb-4 title">Recomendador G1</h1>

        {activeKey === 'recommendations' && (
          <div>
            <Form>
              <Form.Group controlId="userId" className="mb-3">
                <Form.Label className="text-dark form-label">ID do Usuário (UUID)</Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Digite o UUID do usuário"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="form-control"
                />
              </Form.Group>
              <Form.Group controlId="keywords" className="mb-3">
                <Form.Label className="text-dark form-label">Palavras-Chave (separadas por vírgula, opcional)</Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Ex.: esportes, tecnologia"
                  value={keywords}
                  onChange={(e) => setKeywords(e.target.value)}
                  className="form-control"
                />
                <Form.Text className="text-muted form-text">
                  Insira palavras-chave para personalizar recomendações iniciais.
                </Form.Text>
              </Form.Group>
              <Button variant="primary" onClick={fetchRecommendations} className="btn-primary mt-2">
                Obter Recomendações
              </Button>
            </Form>

            {errorMessage && (
              <Alert variant="danger" className="mt-3 alert-danger">
                {errorMessage}
              </Alert>
            )}

            <div className="mt-4">
              <h3 className="text-primary subtitle">Recomendações</h3>
              {recommendations.length > 0 ? (
                <div className="row">
                  {recommendations.map((rec, index) => (
                    <div className="col-md-4 mb-3" key={rec.page}>
                      <Card className="card border-primary shadow-sm">
                        <Card.Body>
                          <Card.Title className="text-primary card-title">{rec.title}</Card.Title>
                          <Card.Text className="text-dark card-text">
                            <strong>ID:</strong> {rec.page}<br />
                            <strong>Data:</strong> {rec.date ? new Date(rec.date).toLocaleDateString() : 'Data não disponível'}<br />
                            <strong>Link:</strong> {rec.link !== 'N/A' ? (
                              <a href={rec.link} target="_blank" rel="noopener noreferrer" onClick={() => logInteraction(rec.page)} className="text-primary">
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
                <p className="text-muted text-no-recommendations">Nenhuma recomendação carregada ainda.</p>
              )}
            </div>
          </div>
        )}

        {activeKey === 'management' && (
          <div>
            <Form>
              <Form.Group controlId="subsampleFrac" className="mb-3">
                <Form.Label className="text-dark form-label">Fração de Subamostragem (0 a 1, opcional)</Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  placeholder="Ex.: 0.1 para 10% dos dados"
                  value={subsampleFrac}
                  onChange={(e) => setSubsampleFrac(e.target.value)}
                  className="form-control"
                />
              </Form.Group>
              <Form.Group controlId="forceRetrain" className="mb-3">
                <Form.Check
                  type="checkbox"
                  label="Forçar Novo Treinamento"
                  checked={forceRetrain}
                  onChange={(e) => setForceRetrain(e.target.checked)}
                  className="text-dark"
                />
              </Form.Group>
              <Button variant="primary" onClick={startTraining} className="btn-primary mt-2">
                Iniciar Treinamento
              </Button>
              <Button variant="info" href="http://localhost:8000/docs" target="_blank" className="btn-info mt-2 ms-2">
                Abrir Swagger UI
              </Button>
            </Form>
          </div>
        )}
      </Container>

      {/* Sidebar de logs na parte direita com auto-scroll */}
      <div className="bg-light sidebar-right" style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: '30%' }}>
        <h5 className="mb-2 text-primary logs-title">Logs do Servidor</h5>
        {logs.length > 0 ? (
          <div ref={logsRef} className="logs-content" style={{ maxHeight: '90vh', overflowY: 'auto', borderLeft: '1px solid #ccc', padding: '10px' }}>
            {logs.map((log, index) => (
              <p key={index} className="log-entry" style={{ margin: '0', whiteSpace: 'pre-wrap', fontSize: '0.9rem', color: '#333' }}>{log}</p>
            ))}
          </div>
        ) : (
          <p className="text-muted no-logs">Nenhum log disponível.</p>
        )}
        <Button variant="primary" onClick={fetchLogs} size="sm" className="btn-primary mt-2">
          Atualizar Logs
        </Button>
      </div>
    </div>
  );
};

export default App;