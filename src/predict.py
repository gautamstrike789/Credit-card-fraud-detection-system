import joblib
import pandas as pd
import numpy as np


def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Convert raw input into the same feature format used during training.
    """
    df = pd.DataFrame([input_data])

    # Create engineered features
    df["transaction_hour"] = (df["Time"] // 3600) % 24
    df["amount_log"] = np.log1p(df["Amount"])

    # Drop raw columns used only for feature engineering
    df = df.drop(columns=["Time", "Amount"])

    return df


def predict_single_transaction(input_data: dict):
    """
    Load model and scaler, preprocess input, and return prediction.
    """
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    input_df = preprocess_input(input_data)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        "recommended_action": "Manual Review" if prediction == 1 else "Approve"
    }