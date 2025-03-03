export interface Recommendation {
  page: string;
  title: string;
  link: string;
  date?: string;
}


export const extractDate = (rec: Recommendation): string => {
  const pattern = /\/noticia\/(\d{4}\/\d{2}\/\d{2})\//;
  const match = rec.link.match(pattern);
  let item = match && match[1].split('/');
  return item ? `${item[0]}/${item[1]}/${item[2]}` : '';
};


