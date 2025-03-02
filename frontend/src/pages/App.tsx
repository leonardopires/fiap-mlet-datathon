/** @jsxImportSource @emotion/react */
import React, { useState, useEffect, useRef } from 'react';
import MainLayout from '../templates/MainLayout';
import Recommendations from '../organisms/Recommendations';
import Management from '../organisms/Management';

interface Status {
  running: boolean;
  progress: string;
  error: string | null;
}

const App: React.FC = () => {
  const [activeKey, setActiveKey] = useState<string>('recommendations');
  const [trainingStatus, setTrainingStatus] = useState<Status>({ running: false, progress: 'idle', error: null });
  const [metricsStatus, setMetricsStatus] = useState<Status>({ running: false, progress: 'idle', error: null });
  const statusWsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/status`);
    statusWsRef.current = ws;
    ws.onopen = () => console.log('WebSocket de status conectado');
    ws.onmessage = (event) => {
      const status = JSON.parse(event.data);
      console.log("Status recebido: ", status)
      setTrainingStatus(status.training);
      setMetricsStatus(status.metrics);
    };
    ws.onerror = (error) => console.error('Erro no WebSocket de status:', error);
    ws.onclose = () => {
      console.log('WebSocket de status desconectado. Tentando reconectar...');
      setTimeout(() => (statusWsRef.current = new WebSocket(`ws://${window.location.hostname}:8000/ws/status`)), 1000);
    };
    return () => ws.close();
  }, []);

  return (
    <MainLayout activeKey={activeKey} setActiveKey={setActiveKey}>
      {activeKey === 'recommendations' && <Recommendations />}
      {activeKey === 'management' && (
        <Management
          trainingStatus={trainingStatus}
          metricsStatus={metricsStatus}
          setTrainingStatus={setTrainingStatus}
          setMetricsStatus={setMetricsStatus}
        />
      )}
    </MainLayout>
  );
};

export default App;