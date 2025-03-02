/** @jsxImportSource @emotion/react */
import React, { ReactNode, useState } from 'react';
import { css } from '@emotion/react';
import { Box, IconButton, Typography, useTheme } from '@mui/material';
import ExpandLess from '@mui/icons-material/ExpandLess';
import Sidebar from '../organisms/Sidebar';
import LogPanel from '../organisms/LogPanel';
import Authors from '../molecules/Authors';
import { SnackbarProvider } from '../contexts/SnackbarContext';
import {authors} from "../models/Author"; // Importe o provedor

interface MainLayoutProps {
  activeKey: string;
  setActiveKey: (key: string) => void;
  children: ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ activeKey, setActiveKey, children }) => {
  const theme = useTheme();
  const [logsVisible, setLogsVisible] = useState(true);

  const containerStyle = css`
    display: flex;
    height: 100vh;
    background: ${theme.palette.background.default};
    overflow: hidden;
  `;

  const contentStyle = css`
    flex-grow: 1;
    padding: 30px;
    margin-left: 60px;
    overflow-y: auto;
    margin-bottom: ${logsVisible ? '210px' : '0'};
  `;


  return (
    <SnackbarProvider>
      <Box css={containerStyle}>
        <Sidebar activeKey={activeKey} setActiveKey={setActiveKey} />
        <Box css={contentStyle}>
          <Box css={css`margin-top: 10px; text-align: right;`}>
            <a href="https://www.fiap.com.br/" target="_blank" rel="noopener noreferrer">
              <img
                src="https://postech.fiap.com.br/svg/fiap-plus-alura.svg"
                alt="FIAP + Alura"
                css={css`width: 150px; height: auto;`}
              />
            </a>
            <Box css={css`margin-bottom: 5px;`}>
              <Typography variant="h5" css={css`color: ${theme.palette.text.primary}; font-size: 0.9rem;`}>
                ML TECH DATATHON - Fase Final - Engenharia em Machine Learning - 2025
              </Typography>
              <Authors authors={authors} />
            </Box>
          </Box>
          {children}
        </Box>
        <LogPanel visible={logsVisible} setVisible={setLogsVisible} />
        {!logsVisible && (
          <IconButton
            onClick={() => setLogsVisible(true)}
            css={css`
              position: fixed;
              bottom: 10px;
              right: 10px;
              z-index: 1000;
              color: ${theme.palette.text.primary};
              background: ${theme.palette.secondary.main};
            `}
          >
            <ExpandLess />
          </IconButton>
        )}
      </Box>
    </SnackbarProvider>
  );
};

export default MainLayout;