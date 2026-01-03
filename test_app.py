from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Football Injury Prediction API is running!"}

def test_predict_injury():
    payload = {
        "age": 25,
        "previous_injuries": 1,
        "training_hours_per_week": 10.5,
        "sleep_hours_per_night": 7.5,
        "hydration_level": 1,
        "nutrition_habits": 1,
        "fitness_level": 2,
        "position": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "injury_likelihood" in data
    assert "preventive_techniques" in data
    assert "predicted_injury_type" in data
