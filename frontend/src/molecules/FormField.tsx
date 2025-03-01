/** @jsxImportSource @emotion/react */
import React from 'react';
import { css } from '@emotion/react';
import { TextField, FormHelperText, useTheme, Grid } from '@mui/material';

interface FormFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
  helpText?: string;
  xs?: number; // Adicionando prop para largura em telas pequenas (equivalente a col-xs-*)
  sm?: number; // Opcional: largura em telas pequenas-médias
  md?: number; // Opcional: largura em telas médias
}

const FormField: React.FC<FormFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  helpText,
  xs = 12, // Padrão: ocupa toda a largura em telas pequenas
  sm,
  md,
}) => {
  const theme = useTheme();

  const labelStyle = css`
    color: ${theme.palette.text.primary};
    font-size: 1.1rem;
    font-weight: 500;
    margin-bottom: 5px;
  `;

  const inputStyle = css`
    width: 100%; /* Ajuste para preencher o container do Grid */
    input {
      border-color: ${theme.palette.primary.main};
      border-radius: 8px;
      padding: 10px 12px;
      background: ${theme.palette.background.paper};
      color: ${theme.palette.text.primary};
      transition: border-color 0.4s ease, transform 0.3s ease;
      &:focus {
        border-color: ${theme.palette.primary.dark};
        transform: scale(1.02);
      }
    }
  `;

  return (
    <Grid item xs={xs} sm={sm} md={md} css={css`margin-bottom: 15px;`}>
      <label htmlFor={id} css={labelStyle}>
        {label}
      </label>
      <TextField
        id={id}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        variant="outlined"
        css={inputStyle}
        fullWidth
      />
      {helpText && (
        <FormHelperText css={css`color: ${theme.palette.text.secondary};`}>
          {helpText}
        </FormHelperText>
      )}
    </Grid>
  );
};

export default FormField;