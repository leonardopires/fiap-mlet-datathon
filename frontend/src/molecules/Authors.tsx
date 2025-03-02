/** @jsxImportSource @emotion/react */

import React from "react";
import {Avatar, Chip, useTheme} from "@mui/material";
import {Author} from "../models/Author";
import {css} from "@emotion/react";

interface AuthorsProps {
  authors: Author[];
}

const Authors: React.FC<AuthorsProps> = ({authors}) => {
  const theme = useTheme();
  return (
    <div css={css`margin-top: 10px; text-align: right;`}>
      {authors.map((author) => (
        <Chip
          component="a"
          href={`https://github.com/${author.github}`}
          clickable
          label={
            <>
              {author.name} <Chip
              label={author.rm}
              size="small"
              color={"secondary"}
              sx={{ fontSize: "0.7rem", marginLeft: "2px", backgroundColor: theme.palette.background.default }}
            />
            </>
          }
          avatar={<Avatar
            src={`https://github.com/${author.github}.png`}
            title={`${author.name} - ${author.rm}`}
          />}
          target="_blank"
          sx={{color: theme.palette.text.primary, fontSize: "0.7rem"}}
        />
      ))}
    </div>
  );
};

export default Authors;