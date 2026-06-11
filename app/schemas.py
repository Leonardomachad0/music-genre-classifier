from pydantic import BaseModel
from datetime import datetime


class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    filename: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool