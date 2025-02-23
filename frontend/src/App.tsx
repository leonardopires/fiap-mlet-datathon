import React, { useState } from 'react';
import { Container, Tabs, Tab, Form, Button, Card } from 'react-bootstrap';
import axios from 'axios';

const App: React.FC = () => {
  const [userId, setUserId] = useState<string>('');
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

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

  // Função para iniciar o treinamento
  const startTraining = async () => {
    try {
      const response = await axios.post('http://localhost:8000/train');
      alert(response.data.message);
    } catch (error) {
      console.error('Erro ao iniciar treinamento:', error);
      alert('Erro ao iniciar treinamento');
    }
  };

  // Stub pra criaçào da função pra obter os logs.
  const fetchLogs = () => {
    setLogs(['Log 1: Iniciando...', 'Log 2: Treinamento em andamento...']); // Placeholder
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
          <Button variant="success" onClick={startTraining} className="mb-3">
            Iniciar Treinamento
          </Button>
          <Button variant="info" href="http://localhost:8000/docs" target="_blank" className="mb-3 ms-2">
            Abrir Swagger UI
          </Button>
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