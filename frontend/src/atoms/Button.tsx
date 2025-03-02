/** @jsxImportSource @emotion/react */
import React, { ReactNode } from 'react';
import { css } from '@emotion/react';
import { Button as MuiButton, ButtonProps as MuiButtonProps, useTheme } from '@mui/material';

interface ButtonProps extends Omit<MuiButtonProps, 'variant'> {
  variant: 'primary' | 'info' | 'secondary';
  icon?: ReactNode;
}

const Button: React.FC<ButtonProps> = ({ variant, onClick, children, icon, disabled, ...rest }) => {
  const theme = useTheme();

  const baseStyle = css`
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    text-transform: none;
    transition: background 0.4s ease, transform 0.3s ease;
    &:hover {
      transform: scale(1.05);
    }
    &:active {
      transform: scale(0.98);
    }
  `;

  const variantStyles = {
    primary: css`
      background: ${theme.palette.primary.main};
      color: ${theme.palette.primary.contrastText};
      &:hover {
        background: ${theme.palette.primary.dark};
      }
    `,
    info: css`
      background: ${theme.palette.info.main};
      color: ${theme.palette.info.contrastText};
      &:hover {
        background: ${theme.palette.info.dark};
      }
    `,
    secondary: css`
      background: ${theme.palette.secondary.main};
      color: ${theme.palette.secondary.contrastText};
      &:hover {
        background: ${theme.palette.secondary.dark};
      }
    `,
  };

  return (
    <MuiButton
      onClick={onClick}
      disabled={disabled}
      css={[baseStyle, variantStyles[variant]]}
      {...rest}
    >
      {icon && <span css={css`margin-right: 8px;`}>{icon}</span>}
      {children}
    </MuiButton>
  );
};

export default Button;