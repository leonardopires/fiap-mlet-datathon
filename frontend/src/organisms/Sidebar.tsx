/** @jsxImportSource @emotion/react */
import React from 'react';
import { css } from '@emotion/react';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, useTheme } from '@mui/material';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import SettingsIcon from '@mui/icons-material/Settings';

interface SidebarProps {
  activeKey: string;
  setActiveKey: (key: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeKey, setActiveKey }) => {
  const theme = useTheme();

  const sidebarStyle = css`
      flex-shrink: 0;

      & .MuiDrawer-paper {
          min-height: 100vh;
          background: ${theme.palette.background.paper};
          padding-top: 20px;
          position: fixed;
          top: 0;
          left: 0;
          z-index: 999;
          display: flex;
          flex-direction: column;
          align-items: center;
          transition: transform 0.4s ease;
          color: ${theme.palette.text.primary};
          border: none;
          overflow-y: hidden;

          &:hover {
              transform: translateX(5px);
          }
      }
  `;

  return (
    <Drawer variant="permanent" css={sidebarStyle}>
      <List>
        <ListItem button selected={activeKey === 'recommendations'} onClick={() => setActiveKey('recommendations')}>
          <ListItemIcon css={css`color: ${theme.palette.text.primary};`}>
            <NewspaperIcon />
          </ListItemIcon>
          <ListItemText primary="Recomendações" css={css`display: none`} />
        </ListItem>
        <ListItem button selected={activeKey === 'management'} onClick={() => setActiveKey('management')}>
          <ListItemIcon css={css`color: ${theme.palette.text.primary};`}>
            <SettingsIcon />
          </ListItemIcon>
          <ListItemText primary="Gerenciamento" css={css`display: none`} />
        </ListItem>
      </List>
    </Drawer>
  );
};

export default Sidebar;