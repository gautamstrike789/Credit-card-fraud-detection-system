from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class TransactionInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["transaction_hour"] = (df["Time"] // 3600) % 24
    df["amount_log"] = np.log1p(df["Amount"])

    df = df.drop(columns=["Time", "Amount"])

    return df


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict_transaction(data: TransactionInput):
    input_df = preprocess_input(data.model_dump())
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        "recommended_action": "Manual Review" if prediction == 1 else "Approve"
    }