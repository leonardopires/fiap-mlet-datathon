/** @jsxImportSource @emotion/react */
import React from 'react';
import { css } from '@emotion/react';
import {
  Grid,
  Typography,
  Box,
  Tooltip,
  IconButton,
  useTheme,
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import { metricsConfig, MetricDefinition } from '../utils/metricsConfig';

interface MetricsDisplayProps {
  metrics: { [key: string]: number };
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
  const theme = useTheme();

  const limitsStyle = css`
    font-size: 0.6rem;
    color: ${theme.palette.text.secondary};
    margin-top: 2px;
  `;

  const infoIconStyle = css`
    font-size: 1rem;
    color: ${theme.palette.text.secondary};
    margin-left: 4px;
    vertical-align: middle;
  `;

  const legendStyle = css`
    display: flex;
    justify-content: flex-start;
    margin-top: 10px;
    flex-wrap: wrap;
    gap: 10px;
  `;

  const legendItemStyle = css`
    display: flex;
    align-items: center;
    margin-right: 15px;
  `;

  const colorBoxStyle = (color: string) => css`
    width: 16px;
    height: 16px;
    background-color: ${color};
    margin-right: 5px;
    border: 1px solid ${theme.palette.divider};
    border-radius: 3px;
  `;

  const getCategoryAndColor = (value: number, metric: MetricDefinition) => {
    if (value === undefined) return { category: 'N/A', color: theme.palette.text.primary };

    const range = metric.ranges.find(
      (r) => value >= r.range[0] && value <= r.range[1]
    );
    if (!range) return { category: 'N/A', color: theme.palette.text.primary };

    const { category } = range;
    let color: string;
    switch (category) {
      case 'ruim':
        color = theme.palette.error.main; // Vermelho
        break;
      case 'aceitável':
        color = theme.palette.warning.main; // Amarelo
        break;
      case 'bom':
        color = theme.palette.success.main; // Verde
        break;
      case 'excelente':
        color = theme.palette.info.main; // Azul
        break;
      default:
        color = theme.palette.text.primary;
    }
    return { category, color };
  };

  return (
    <Box>
      {/* Métricas */}
      <Grid container spacing={2} css={css`margin-top: 2px;`}>
        {metricsConfig.map((metric: MetricDefinition) => {
          const value = metrics[metric.field];
          const displayValue =
            value !== undefined && metric.transform
              ? metric.transform(value)
              : value !== undefined
              ? value.toFixed(metric.significantDigits)
              : "N/A";
          const displayLimits = metric.transform
            ? `Min: ${(metric.ranges[0].range[0] * 100).toFixed(0)}%, Max: ${(metric.ranges[metric.ranges.length - 1].range[1] * 100).toFixed(0)}%`
            : `Min: ${metric.ranges[0].range[0].toFixed(2)}, Max: ${metric.ranges[metric.ranges.length - 1].range[1].toFixed(2)}`;
          const { color } = getCategoryAndColor(value, metric);
          return (
            <React.Fragment key={metric.field}>
              <Grid item xs={6}>
                <Box display="flex" alignItems="center">
                  <Typography color="textPrimary" sx={{ fontSize: "0.7rem" }}>
                    {metric.label}
                  </Typography>
                  <Tooltip title={metric.tooltip} arrow>
                    <IconButton size="small">
                      <InfoIcon css={infoIconStyle} />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography css={limitsStyle}>
                  {displayLimits}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography css={css`color: ${color}; font-weight: 500;`}>
                  {displayValue}
                </Typography>
              </Grid>
            </React.Fragment>
          );
        })}
      </Grid>

      {/* Legenda de cores */}
      <Box css={legendStyle}>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.error.main)} />
          <Typography variant="caption">Ruim</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.warning.main)} />
          <Typography variant="caption">Aceitável</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.success.main)} />
          <Typography variant="caption">Bom</Typography>
        </Box>
        <Box css={legendItemStyle}>
          <Box css={colorBoxStyle(theme.palette.info.main)} />
          <Typography variant="caption">Excelente</Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default MetricsDisplay;