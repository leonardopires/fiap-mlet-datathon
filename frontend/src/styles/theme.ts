import {createTheme} from "@mui/material";

export const theme = {
  colors: { primary: '#007bff', dark: '#1a1a1a' },
  transitions: { default: '0.3s ease' },
};

// Criar um tema escuro personalizado
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#007bff', // Azul primário
      dark: '#0056b3',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#6c757d', // Cinza secundário
      dark: '#5a6268',
      contrastText: '#ffffff',
    },
    info: {
      main: '#17a2b8', // Azul info
      dark: '#117a8b',
      contrastText: '#ffffff',
    },
    error: {
      main: '#dc3545', // Vermelho para erros
      dark: '#c82333',
      contrastText: '#ffffff',
    },
    background: {
      default: '#000000', // Fundo preto puro
      paper: '#323234', // Fundo dos elementos como Drawer
    },
    text: {
      primary: '#ffffff', // Texto principal branco
      secondary: '#b0b0b0', // Texto secundário cinza claro
      disabled: '#7f8c8d', // Texto desativado
    },
  },
  typography: {
    fontFamily: "'Poppins', sans-serif",
    h1: { fontSize: '1.5rem', fontWeight: 600 },
    h3: { fontSize: '1.4rem', fontWeight: 600 },
    h5: { fontSize: '1.3rem', fontWeight: 600 },
    body1: { fontSize: '1rem' },
    body2: { fontSize: '0.9rem' },
  },
});