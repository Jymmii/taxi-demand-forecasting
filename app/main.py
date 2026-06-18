from fastapi import FastAPI

from app.schemas.prediction_schema import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import predict_taxi_demand

app = FastAPI(title="NYC Taxi Demand Forecasting API")


@app.get("/")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict taxi demand."""
    predicted_pickups = predict_taxi_demand(request)

    return PredictionResponse(predicted_pickups=predicted_pickups)
