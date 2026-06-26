from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    PULocationID: int = Field(gt=0)
    hour: int = Field(gt=0, le=23)
    day_of_week: int = Field(gt=0, le=6)
    is_weekend: bool


class PredictionResponse(BaseModel):
    predicted_pickups: float
