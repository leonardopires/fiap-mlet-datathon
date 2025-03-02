/** @jsxImportSource @emotion/react */
import React, {useRef, useEffect, useState} from 'react';
import {css} from '@emotion/react';
import {Drawer, Box, Typography, IconButton, useTheme} from '@mui/material';
import ExpandMore from '@mui/icons-material/ExpandMore';

interface LogPanelProps {
  visible: boolean;
  setVisible: (visible: boolean) => void;
}

const LogPanel: React.FC<LogPanelProps> = ({visible, setVisible}) => {
  const theme = useTheme();
  const [logs, setLogs] = useState<string[]>([]);
  const logsRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/logs`);
    wsRef.current = ws;
    ws.onopen = () => console.log('WebSocket de logs conectado');
    ws.onmessage = (event) => setLogs((prev) => [...prev, event.data].slice(-2000));
    ws.onclose = () => setTimeout(() => (wsRef.current = new WebSocket(`ws://${window.location.hostname}:8000/ws/logs`)), 1000);
    return () => ws.close();
  }, []);

  // Função para verificar se o usuário está no final do conteúdo
  const checkIfAtBottom = () => {
    if (!logsRef.current) return false;

    let {scrollTop, scrollHeight, clientHeight} = logsRef.current;
    scrollTop = scrollTop ?? 0;
    scrollHeight = scrollHeight ?? 0;
    clientHeight = clientHeight ?? 0;

    // Consideramos "no final" se o usuário estiver a menos de 50px do fim
    return scrollHeight - scrollTop - clientHeight < 50;
  };

  // Rolar para o final quando novos logs chegarem, mas só se estiver no final
  useEffect(() => {
    if (isAtBottom && logsRef.current) {
      logsRef.current.scrollTo({
        top: logsRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [logs, isAtBottom]);

  // Atualizar o estado `isAtBottom` quando o usuário rolar


  // Adicionar ouvinte de rolagem ao montar o componente
  useEffect(() => {
    const handleScroll = () => {
      setIsAtBottom(checkIfAtBottom());
    };

    const logsElement = logsRef.current;
    if (logsElement) {
      logsElement.addEventListener('scroll', handleScroll);
      return () => logsElement.removeEventListener('scroll', handleScroll);
    }
  }, []);

  const panelStyle = css`
      width: auto; /* Garante que o Drawer não se expanda para toda a largura */

      & .MuiDrawer-paper {
          bottom: ${visible ? '0' : '-200px'};
          max-height: 200px;
          height: 200px;
          margin-left: 100px;
          margin-right: 40px;
          background: ${theme.palette.background.paper};
          z-index: 1000;
          box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
          transition: bottom 0.4s ease;
          padding: 10px;
      }
  `;

  const logsContentStyle = css`
      max-height: 80%;
      overflow-y: auto;
      border-top: 1px solid ${theme.palette.divider};
  `;

  return (
    <Drawer variant="permanent" anchor="bottom" css={panelStyle}>
      {visible && (
        <>
          <Box css={css`display: flex;
              justify-content: space-between;
              align-items: center;`}>
            <Typography css={css`color: ${theme.palette.text.primary};
                font-weight: 600;
                font-size: 1.4rem;`}>
              Logs do Servidor
            </Typography>
            <IconButton onClick={() => setVisible(false)} css={css`color: ${theme.palette.text.primary};`}>
              <ExpandMore/>
            </IconButton>
          </Box>
          <Box ref={logsRef} css={logsContentStyle}>
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <Typography key={index} css={css`color: ${theme.palette.text.secondary};
                    margin: 0;
                    white-space: pre-wrap;
                    font-size: 0.9rem;`}>
                  {log}
                </Typography>
              ))
            ) : (
              <Typography css={css`color: ${theme.palette.text.disabled};`}>Nenhum log disponível.</Typography>
            )}
          </Box>
        </>
      )}
    </Drawer>
  );
};

export default LogPanel;