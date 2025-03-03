/** @jsxImportSource @emotion/react */
import React from 'react';
import { css } from '@emotion/react';
import { Card, CardContent, Typography, useTheme } from '@mui/material';
import {extractDate, Recommendation} from "../utils/api";

interface RecommendationCardProps {
  recommendation: Recommendation;
  onLinkClick?: (page: string) => void;
}


const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, onLinkClick }) => {
  const theme = useTheme();

  const cardStyle = css`
    border-color: ${theme.palette.primary.main};
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    background: ${theme.palette.background.paper};
    transition: transform 0.4s ease, box-shadow 0.4s ease;
    &:hover {
      transform: translateY(-5px);
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }
  `;

  const titleStyle = css`
    color: ${theme.palette.text.primary};
    font-weight: 600;
    font-size: 1.2rem;
  `;

  const textStyle = css`
    font-size: 1rem;
    color: ${theme.palette.text.secondary};
  `;

  const linkStyle = css`
    color: ${theme.palette.text.primary};
    transition: color 0.3s ease;
    &:hover {
      color: ${theme.palette.primary.main};
    }
  `;


  return (
    <Card css={cardStyle}>
      <CardContent>
        <Typography css={titleStyle}>{recommendation.title}</Typography>
        <Typography css={textStyle}>
          <strong>ID:</strong> {recommendation.page}<br />
          <strong>Data:</strong> {extractDate(recommendation)}<br />
          <strong>Link:</strong>{' '}
          {recommendation.link !== 'N/A' ? (
            <a
              href={recommendation.link}
              target="_blank"
              rel="noopener noreferrer"
              css={linkStyle}
              onClick={() => onLinkClick?.(recommendation.page)}
            >
              {recommendation.link}
            </a>
          ) : (
            'Não disponível'
          )}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default RecommendationCard;