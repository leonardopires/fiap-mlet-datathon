/** @jsxImportSource @emotion/react */
import React from 'react';
import { css } from '@emotion/react';
import { Alert as MuiAlert, AlertProps as MuiAlertProps, useTheme } from '@mui/material';

interface AlertProps extends Omit<MuiAlertProps, 'variant'> {
  variant: 'danger' | 'info';
}

const Alert: React.FC<AlertProps> = ({ variant, children, ...rest }) => {
  const theme = useTheme();

  const baseStyle = css`
    border-radius: 10px;
    padding: 15px;
    font-size: 1rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    &:hover {
      transform: scale(1.02);
    }
  `;

  const variantStyles = {
    danger: css`
      background: ${theme.palette.error.main};
      color: ${theme.palette.error.contrastText};
      box-shadow: 0 2px 10px rgba(220, 53, 69, 0.3);
      &:hover {
        background: ${theme.palette.error.dark};
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
      }
    `,
    info: css`
      background: ${theme.palette.info.main};
      color: ${theme.palette.info.contrastText};
      box-shadow: 0 2px 10px rgba(23, 162, 184, 0.3);
      &:hover {
        background: ${theme.palette.info.dark};
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
      }
    `,
  };

  return (
    <MuiAlert
      severity={variant === 'danger' ? 'error' : 'info'}
      css={[baseStyle, variantStyles[variant]]}
      {...rest}
    >
      {children}
    </MuiAlert>
  );
};

export default Alert;