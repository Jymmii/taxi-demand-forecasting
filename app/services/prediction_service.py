import joblib
import pandas as pd

from app.schemas.prediction_schema import PredictionRequest
from configs.training_config import CHAMPION_MODEL_PATH

model = joblib.load(CHAMPION_MODEL_PATH)


def predict_taxi_demand(request: PredictionRequest) -> float:
    """Predict hourly taxi demand from request features."""
    input_df = pd.DataFrame(
        {
            "PULocationID": [request.PULocationID],
            "hour": [request.hour],
            "day_of_week": [request.day_of_week],
            "is_weekend": [request.is_weekend],
        }
    )

    prediction = model.predict(input_df)

    return float(prediction[0])
