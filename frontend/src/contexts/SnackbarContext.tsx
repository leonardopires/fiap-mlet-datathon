/** @jsxImportSource @emotion/react */
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Snackbar, Alert as MuiAlert, AlertColor } from '@mui/material';

interface SnackbarMessage {
  message: string;
  severity: AlertColor;
}

interface SnackbarContextType {
  showSnackbar: (message: string, severity?: AlertColor) => void;
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
  const [snackbar, setSnackbar] = useState<SnackbarMessage | null>(null);
  const [open, setOpen] = useState(false);

  const showSnackbar = (message: string, severity: AlertColor = 'info') => {
    setSnackbar({ message, severity });
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    // Atrasar a limpeza do estado para garantir que a transição termine
    setTimeout(() => setSnackbar(null), 200);
  };

  // Limpeza ao desmontar o componente
  useEffect(() => {
    return () => {
      setOpen(false);
      setSnackbar(null);
    };
  }, []);

  return (
    <SnackbarContext.Provider value={{ showSnackbar }}>
      {children}
      <Snackbar
        open={open}
        autoHideDuration={6000}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        TransitionProps={{ onExited: handleClose }} // Garante que o estado seja limpo após a transição
      >
        {snackbar ? (
          <MuiAlert
            severity={snackbar.severity}
            variant={'filled'}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </MuiAlert>
        ) : undefined}
      </Snackbar>
    </SnackbarContext.Provider>
  );
};