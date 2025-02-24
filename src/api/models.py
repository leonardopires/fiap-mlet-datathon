from pydantic import BaseModel
from typing import Optional

class UserRequest(BaseModel):
    """
    Define como deve ser uma requisição para pedir recomendações para um usuário.
    """
    user_id: str  # O ID do usuário que queremos recomendar notícias


class TrainRequest(BaseModel):
    """
    Define como deve ser uma requisição para treinar o modelo.
    """
    subsample_frac: Optional[float] = None  # Parte dos dados a usar (opcional, ex.: 0.1 para 10%)
    force_reprocess: Optional[bool] = False  # Se True, refaz tudo do zero


class PredictionResponse(BaseModel):
    """
    Define como será a resposta com as recomendações.
    """
    user_id: str  # O ID do usuário
    acessos_futuros: list[dict]  # Lista de notícias recomendadas
