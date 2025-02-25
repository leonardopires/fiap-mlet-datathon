from pydantic import BaseModel
from typing import Optional, List

class UserRequest(BaseModel):
    user_id: str
    keywords: Optional[List[str]] = None  # Lista opcional de palavras-chave

class TrainRequest(BaseModel):
    subsample_frac: Optional[float] = None
    force_reprocess: Optional[bool] = False
    force_retrain: Optional[bool] = False

class PredictionResponse(BaseModel):
    user_id: str
    acessos_futuros: list[dict]