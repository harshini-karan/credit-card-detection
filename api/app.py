from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import numpy as np
import joblib
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions using a trained model.",
    version="1.0.0"
)

# Define the number of input features (V1-V28 + Amount_scaled)
N_FEATURES = 29

# Load the trained model and scaler (if available)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../rf_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../scaler.joblib')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

# Define the request schema using Pydantic v2 style
class Transaction(BaseModel):
    features: Annotated[
        list[float],
        Field(
            ...,
            min_length=N_FEATURES,
            max_length=N_FEATURES,
            description=f"List of {N_FEATURES} floats: V1-V28 and Amount_scaled"
        )
    ]

@app.post("/predict")
async def predict(transaction: Transaction):
    """
    Predict if a transaction is fraudulent.
    Input: JSON with 'features' (list of floats, length 29)
    Output: JSON with prediction and probability
    """
    data = np.array(transaction.features).reshape(1, -1)
    # If scaler is used, scale the features (if needed)
    if scaler:
        data = scaler.transform(data)
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]
    return {
        "fraudulent": bool(prediction),
        "probability": float(proba)
    }

@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is running."}
