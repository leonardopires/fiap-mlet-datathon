/** @jsxImportSource @emotion/react */
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Snackbar, Alert as MuiAlert, AlertColor, CircularProgress, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface SnackbarMessage {
  id: string; // Identificador único para cada mensagem
  message: string;
  severity: AlertColor;
  loading?: boolean; // Indica se a mensagem está em estado de carregamento
}

interface SnackbarContextType {
  showSnackbar: (message: string, severity?: AlertColor, loading?: boolean) => void;
  updateSnackbar: (id: string, message: string, severity?: AlertColor, loading?: boolean) => void;
}

const SnackbarContext = createContext<SnackbarContextType | undefined>(undefined);

export const useSnackbar = () => {
  const context = useContext(SnackbarContext);
  if (!context) {
    throw new Error('useSnackbar must be used within a SnackbarProvider');
  }
  return context;
};

interface SnackbarProviderProps {
  children: ReactNode;
}

export const SnackbarProvider: React.FC<SnackbarProviderProps> = ({ children }) => {
  const [snackbars, setSnackbars] = useState<SnackbarMessage[]>([]);

  // Função para gerar um ID único para cada mensagem
  const generateId = () => Date.now().toString() + Math.random().toString(36).substr(2, 9);

  const showSnackbar = (message: string, severity: AlertColor = 'info', loading: boolean = false) => {
    // Evitar duplicatas: verificar se já existe uma mensagem idêntica
    const exists = snackbars.some(
      (snack) => snack.message === message && snack.severity === severity && snack.loading === loading
    );
    if (exists) return;

    const id = generateId();
    setSnackbars((prev) => [...prev, { id, message, severity, loading }]);
  };

  // Função para atualizar uma mensagem existente (ex.: mudar de loading para sucesso/erro)
  const updateSnackbar = (id: string, message: string, severity?: AlertColor, loading: boolean = false) => {
    setSnackbars((prev) =>
      prev.map((snack) =>
        snack.id === id
          ? { ...snack, message, severity: severity || snack.severity, loading }
          : snack
      )
    );
  };

  const handleClose = (id: string) => {
    setSnackbars((prev) => prev.filter((snack) => snack.id !== id));
  };

  return (
    <SnackbarContext.Provider value={{ showSnackbar, updateSnackbar }}>
      {children}
      {snackbars.map((snack) => (
        <Snackbar
          key={snack.id}
          open={true}
          autoHideDuration={snack.loading ? null : 6000} // Não fechar automaticamente se estiver carregando
          onClose={() => handleClose(snack.id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          sx={{ marginTop: `${snackbars.indexOf(snack) * 60}px` }} // Empilhar mensagens
        >
          <MuiAlert
            severity={snack.severity}
            variant={'filled'}
            sx={{ width: '100%' }}
            action={
              <>
                {snack.loading && <CircularProgress size={20} sx={{ marginRight: 2 }} />}
                {!snack.loading && (
                  <IconButton
                    size="small"
                    color="inherit"
                    onClick={() => handleClose(snack.id)}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                )}
              </>
            }
          >
            {snack.message}
          </MuiAlert>
        </Snackbar>
      ))}
    </SnackbarContext.Provider>
  );
};