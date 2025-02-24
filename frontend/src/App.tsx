import React, { useState } from 'react';
import { Container, Tabs, Tab, Form, Button, Card, Alert } from 'react-bootstrap';
import axios from 'axios';

const App: React.FC = () => {
  const [userId, setUserId] = useState<string>('');
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [subsampleFrac, setSubsampleFrac] = useState<string>(''); // Novo estado para subsample_frac
  const [trainStatus, setTrainStatus] = useState<string>(''); // Estado para feedback do treinamento

  // Função para obter recomendações
  const fetchRecommendations = async () => {
    try {
      const response = await axios.post('http://localhost:8000/predict', { user_id: userId });
      setRecommendations(response.data.acessos_futuros);
    } catch (error) {
      console.error('Erro ao obter recomendações:', error);
      setRecommendations(['Erro ao carregar recomendações']);
    }
  };

  // Função para iniciar o treinamento com subsample_frac opcional
  const startTraining = async () => {
    try {
      const payload: { subsample_frac?: number; force_reprocess?: boolean } = {};
      if (subsampleFrac) {
        const frac = parseFloat(subsampleFrac);
        if (frac > 0 && frac <= 1) {
          payload.subsample_frac = frac;
        } else {
          setTrainStatus('Erro: subsample_frac deve estar entre 0 e 1.');
          return;
        }
      }
      // Adicione force_reprocess se quiser suportar isso na UI também
      const response = await axios.post('http://localhost:8000/train', payload);
      setTrainStatus(response.data.message);
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      setTrainStatus('Erro ao iniciar treinamento.');
    }
  };

  // Função fictícia para obter logs (atualize com endpoint real se disponível)
  const fetchLogs = () => {
    setLogs(['Log 1: Iniciando...', 'Log 2: Treinamento em andamento...']);
  };

  return (
    <Container className="mt-4">
      <h1>Recomendador G1</h1>
      <Tabs defaultActiveKey="recommendations" id="main-tabs" className="mb-3">
        {/* Tab de Recomendações */}
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
            <Button variant="primary" onClick={fetchRecommendations}>
              Obter Recomendações
            </Button>
          </Form>
          <div className="mt-4">
            <h3>Recomendações</h3>
            {recommendations.length > 0 ? (
              <div className="row">
                {recommendations.map((rec, index) => (
                  <div className="col-md-4 mb-3" key={index}>
                    <Card>
                      <Card.Body>
                        <Card.Title>Notícia {index + 1}</Card.Title>
                        <Card.Text>{rec}</Card.Text>
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

        {/* Tab de Gerenciamento */}
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
            <Button variant="success" onClick={startTraining} className="mb-3">
              Iniciar Treinamento
            </Button>
            <Button variant="info" href="http://localhost:8000/docs" target="_blank" className="mb-3 ms-2">
              Abrir Swagger UI
            </Button>
            {trainStatus && (
              <Alert variant={trainStatus.includes('Erro') ? 'danger' : 'success'} className="mt-3">
                {trainStatus}
              </Alert>
            )}
          </Form>
          <h3>Logs do Servidor</h3>
          <Button variant="secondary" onClick={fetchLogs} className="mb-3">
            Atualizar Logs
          </Button>
          <pre style={{ maxHeight: '400px', overflowY: 'auto', backgroundColor: '#f8f9fa', padding: '10px' }}>
            {logs.length > 0 ? logs.join('\n') : 'Nenhum log disponível ainda.'}
          </pre>
        </Tab>
      </Tabs>
    </Container>
  );
};

export default App;