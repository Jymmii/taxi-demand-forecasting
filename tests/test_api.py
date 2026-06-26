from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_request() -> None:
    payload = {
        "PULocationID": 161,
        "hour": 17,
        "day_of_week": 4,
        "is_weekend": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_pickups" in response.json()
    assert isinstance(response.json()["predicted_pickups"], float)


def test_predict_rejects_invalid_hour() -> None:
    payload = {
        "PULocationID": 161,
        "hour": 30,
        "day_of_week": 4,
        "is_weekend": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_day_of_week() -> None:
    payload = {
        "PULocationID": 161,
        "hour": 17,
        "day_of_week": 9,
        "is_weekend": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_pickup_location() -> None:
    payload = {
        "PULocationID": 0,
        "hour": 17,
        "day_of_week": 4,
        "is_weekend": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_field() -> None:
    payload = {
        "PULocationID": 161,
        "hour": 17,
        "day_of_week": 4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
