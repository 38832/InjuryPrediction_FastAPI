from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from joblib import load
from sklearn.impute import SimpleImputer

# Load the trained model
model = load("injury_prediction_model.joblib")

# Define FastAPI app
app = FastAPI(title="Football Injury Prediction API")

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Root endpoint to check if API is running
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def home():
    return {"message": "Football Injury Prediction API is running!"}

# Define input schema with Position
class PlayerData(BaseModel):
    age: int
    previous_injuries: int
    training_hours_per_week: float
    sleep_hours_per_night: float
    hydration_level: int  # 1: Adequate, 2: Insufficient, 3: Optimal
    nutrition_habits: int  # 1: Balanced, 2: Varied, 3: High Protein
    fitness_level: int  # 1: Low, 2: Moderate, 3: High
    position: int  # 1: Forward, 2: Midfielder, 3: Defender, 4: GoalKeeper

# Explicitly handle CORS preflight (OPTIONS request)
@app.options("/predict")
async def options_predict(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# API endpoint for prediction
@app.post("/predict")
def predict_injury(player: PlayerData):
    # Convert input data to DataFrame
    player_df = pd.DataFrame([player.model_dump()])
    
    # Handle missing values (if any)
    imputer = SimpleImputer(strategy="mean")
    player_df_imputed = imputer.fit_transform(player_df)

    # Make prediction
    prediction = model.predict(player_df_imputed)

    return {
        "injury_likelihood": prediction[0][0],
        "preventive_techniques": prediction[0][1],
        "predicted_injury_type": prediction[0][2]
    }

