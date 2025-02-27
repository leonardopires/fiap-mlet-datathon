import React, { useState } from 'react';
import { Container, Tabs, Tab, Form, Button, Card, Alert } from 'react-bootstrap';
import axios from 'axios';

interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string; // Adicionamos a data
}

const App: React.FC = () => {
  const [userId, setUserId] = useState<string>('');
  const [keywords, setKeywords] = useState<string>(''); // Estado para palavras-chave
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [subsampleFrac, setSubsampleFrac] = useState<string>('');
  const [forceRetrain, setForceRetrain] = useState<boolean>(false);
  const [trainStatus, setTrainStatus] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>(""); // Novo estado para erro

  // Define o tipo do payload explicitamente
  interface PredictPayload {
    user_id: string;
    keywords?: string[]; // Propriedade opcional para palavras-chave
  }

  const fetchRecommendations = async () => {
    try {
      const payload: PredictPayload = { user_id: userId }; // Tipo explícito
      if (keywords) {
        payload.keywords = keywords.split(',').map(kw => kw.trim()); // Agora TypeScript reconhece 'keywords'
      }

      const response = await axios.post('http://localhost:8000/predict', payload);
      setRecommendations(response.data.acessos_futuros);
      setErrorMessage(""); // Se a requisição for bem-sucedida, remove o erro

    } catch (error: any) {
      console.error('Erro ao obter recomendações:', error);

      // Se for erro 400, exibir mensagem personalizada
      if (error.response && error.response.status === 400) {
        setErrorMessage("⚠️ O modelo ainda não foi treinado. Tente novamente após o treinamento.");
      } else {
        setErrorMessage("Erro ao carregar recomendações.");
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
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      setTrainStatus('Erro ao iniciar treinamento.');
    }
  };

  const logInteraction = async (page: string) => {
    try {
      const interaction = {
        user_id: userId,
        page: page,
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
      const allLogs = response.data.logs;
      const recentLogs = allLogs.slice(-100);
      setLogs(recentLogs);
    } catch (error) {
      console.error('Erro ao obter logs:', error);
      setLogs(['Erro ao carregar logs do servidor.']);
    }
  };

  return (
    <Container className="mt-4">
      <h1>Recomendador G1</h1>
      <Tabs defaultActiveKey="recommendations" id="main-tabs" className="mb-3">
        <Tab eventKey="recommendations" title="Recomendações">
          <Form>
            <Form.Group controlId="userId" className="mb-3">
              <Form.Label>ID do Usuário (UUID)</Form.Label>
              <Form.Control
                type="text"
                placeholder="Digite o UUID do usuário"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
              />
            </Form.Group>
            <Form.Group controlId="keywords" className="mb-3">
              <Form.Label>Palavras-Chave (separadas por vírgula, opcional)</Form.Label>
              <Form.Control
                type="text"
                placeholder="Ex.: esportes, tecnologia"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
              />
              <Form.Text className="text-muted">
                Insira palavras-chave para personalizar recomendações iniciais.
              </Form.Text>
            </Form.Group>
            <Button variant="primary" onClick={fetchRecommendations}>
              Obter Recomendações
            </Button>
          </Form>

          {/* Exibir alerta caso haja erro */}
          {errorMessage && (
            <Alert variant="danger" className="mt-3">
              {errorMessage}
            </Alert>
          )}

          <div className="mt-4">
            <h3>Recomendações</h3>
            {recommendations.length > 0 ? (
              <div className="row">
                {recommendations.map((rec, index) => (
                  <div className="col-md-4 mb-3" key={rec.page}>
                    <Card>
                      <Card.Body>
                        <Card.Title>{rec.title}</Card.Title>
                          <Card.Text>
                            <strong>ID:</strong> {rec.page}<br />
                            <strong>Data:</strong> {rec.date ? new Date(rec.date).toLocaleDateString() : 'Data não disponível'}<br />
                            <strong>Link:</strong> {rec.link !== 'N/A' ? (
                              <a href={rec.link} target="_blank" rel="noopener noreferrer" onClick={() => logInteraction(rec.page)}>
                                {rec.link}
                              </a>
                            ) : (
                              'Não disponível'
                            )}
                          </Card.Text>
                      </Card.Body>
                    </Card>
                  </div>
                ))}
              </div>
            ) : (
              <p>Nenhuma recomendação carregada ainda.</p>
            )}
          </div>
        </Tab>

        <Tab eventKey="management" title="Gerenciamento">
          <Form>
            <Form.Group controlId="subsampleFrac" className="mb-3">
              <Form.Label>Fração de Subamostragem (0 a 1, opcional)</Form.Label>
              <Form.Control
                type="number"
                step="0.1"
                min="0"
                max="1"
                placeholder="Ex.: 0.1 para 10% dos dados"
                value={subsampleFrac}
                onChange={(e) => setSubsampleFrac(e.target.value)}
              />
            </Form.Group>
            <Form.Group controlId="forceRetrain" className="mb-3">
              <Form.Check
                type="checkbox"
                label="Forçar Novo Treinamento"
                checked={forceRetrain}
                onChange={(e) => setForceRetrain(e.target.checked)}
              />
            </Form.Group>
            <Button variant="success" onClick={startTraining} className="mb-3">
              Iniciar Treinamento
            </Button>
            <Button variant="info" href="http://localhost:8000/docs" target="_blank" className="mb-3 ms-2">
              Abrir Swagger UI
            </Button>
          </Form>
        </Tab>
      </Tabs>
    </Container>
  );
};

export default App;
