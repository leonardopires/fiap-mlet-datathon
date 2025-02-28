import React, { useState, useEffect, useRef } from 'react';
import { Container, Form, Button, Card, Alert, Nav } from 'react-bootstrap';
import axios from 'axios';
import DescriptionIcon from '@mui/icons-material/Description'; // Para labels de ID e Keywords
import PlayArrowIcon from '@mui/icons-material/PlayArrow'; // Corrigido para o botão "Iniciar Treinamento" e "Obter Recomendações"
import NewspaperIcon from '@mui/icons-material/Newspaper'; // Recomendações
import SettingsIcon from '@mui/icons-material/Settings'; // Gerenciamento

interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string;
}

const App: React.FC = () => {
  const [activeKey, setActiveKey] = useState<string>('recommendations'); // Adiciona o estado para controlar a navegação
  const [userId, setUserId] = useState<string>('');
  const [keywords, setKeywords] = useState<string>('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>('');

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
      console.log('Resposta da API:', response.data);
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
          alert('Erro: subsample_frac deve estar entre 0 e 1.');
          return;
        }
      }
      payload.force_retrain = forceRetrain;
      await axios.post('http://localhost:8000/train', payload);
      fetchLogs();
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      alert('Erro ao iniciar treinamento.');
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
      await axios.post('http://localhost:8000/log_interaction', interaction);
      console.log('Interação registrada com sucesso');
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
    <div className="d-flex vh-100 app-container" style={{ background: 'linear-gradient(to bottom, #1a1a1a, #2d2d2d)', minHeight: '100vh' }}>
      {/* Logo FIAP + Alura no canto superior esquerdo */}
      <div className="fiap-logo" style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 1001 }}>
        <a href="https://www.fiap.com.br/" target="_blank" rel="noopener noreferrer">
          <img src="https://postech.fiap.com.br/svg/fiap-plus-alura.svg" alt="FIAP + Alura" className="logo-img" style={{ width: '150px', height: 'auto' }} />
        </a>
      </div>

      {/* Sidebar de ícones no canto esquerdo */}
      <div className="bg-dark sidebar-left" style={{ width: '60px', background: '#1a1a1a' }}>
        <Nav variant="pills" className="flex-column">
          <Nav.Item className="mb-2">
            <Nav.Link
              eventKey="recommendations"
              onClick={() => setActiveKey('recommendations')}
              className="text-white icon-nav d-flex align-items-center justify-content-center"
              data-label="Recomendações"
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
            >
              <SettingsIcon />
            </Nav.Link>
          </Nav.Item>
        </Nav>
      </div>

      {/* Conteúdo principal com referência ao trabalho */}
      <Container fluid className="flex-grow-1 p-4 main-content" style={{ background: 'none' }}>
        {/* Seção de referência ao trabalho */}
        <div className="work-reference mb-5">
          <h2 className="text-white mb-3" style={{ fontSize: '1.8rem', fontWeight: 600 }}>
            ML TECH DATATHON - Fase Final - Engenharia em Machine Learning - 2025
          </h2>
          <p className="text-white mb-2" style={{ fontSize: '1.2rem', fontWeight: 500 }}>
            Entrega para a etapa final do curso Datathon 2025
          </p>
          <p className="text-white" style={{ fontSize: '1rem', fontWeight: 400 }}>
            Membros do Grupo:
            <br />- Leonardo T Pires: RM355401
            <br />- Felipe de Paula G.: RM355402
            <br />- Jorge Guilherme D. W: RM355849
          </p>
        </div>

        <h1 className="text-white mb-4 title">Recomendador G1</h1>

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
                  onClick={() => logInteraction(userId)}
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
                  onClick={() => logInteraction(keywords)}
                />
                <Form.Text className="text-muted form-text">
                  Insira palavras-chave para personalizar recomendações iniciais.
                </Form.Text>
              </Form.Group>
              <Button variant="primary" onClick={fetchRecommendations} className="btn-primary mt-2 d-flex align-items-center">
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
                      <Card className="card border-dark shadow-dark" style={{ background: '#2d2d2d' }}>
                        <Card.Body>
                          <Card.Title className="text-white card-title">{rec.title}</Card.Title>
                          <Card.Text className="text-light card-text">
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
                  onClick={() => logInteraction(subsampleFrac)}
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
              <Button variant="primary" onClick={startTraining} className="btn-primary mt-2 d-flex align-items-center">
                <PlayArrowIcon className="me-2" /> Iniciar Treinamento
              </Button>
              {/* Iframe para o Swagger UI na aba Gerenciamento */}
              <div className="mt-4">
                <iframe
                  src="http://localhost:8000/docs"
                  title="Swagger UI"
                  className="swagger-iframe"
                  style={{ width: '100%', height: '600px', border: 'none', borderRadius: '5px' }}
                />
              </div>
            </Form>
          </div>
        )}
      </Container>

      {/* Sidebar de logs no canto inferior com dark mode */}
      <div className="bg-dark sidebar-bottom" style={{ position: 'fixed', bottom: 0, left: 0, right: 0, height: '20%', zIndex: 1000, boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.3)', background: '#1a1a1a' }}>
        <h5 className="mb-2 text-light logs-title">Logs do Servidor</h5>
        {logs.length > 0 ? (
          <div ref={logsRef} className="logs-content" style={{ maxHeight: '80%', overflowY: 'auto', borderTop: '1px solid #444', padding: '10px' }}>
            {logs.map((log, index) => (
              <p key={index} className="log-entry text-light" style={{ margin: '0', whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>{log}</p>
            ))}
          </div>
        ) : (
          <p className="text-muted no-logs">Nenhum log disponível.</p>
        )}
      </div>
    </div>
  );
};

export default App;