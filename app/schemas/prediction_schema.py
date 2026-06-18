from pydantic import BaseModel


class PredictionRequest(BaseModel):
    PULocationID: int
    hour: int
    day_of_week: int
    is_weekend: bool


class PredictionResponse(BaseModel):
    predicted_pickups: float
