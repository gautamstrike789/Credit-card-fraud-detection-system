# Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions, including a trained ML model, REST API, and an interactive analytics dashboard.

This project demonstrates how a fraud detection model can be developed, deployed, and served through a real-time prediction system.

---

## Live System

### Dashboard (Streamlit)
Interactive analytics dashboard:  
https://credit-card-fraud-detection-system-3v9qroqcgendr2wuwcksn5.streamlit.app/

### API (FastAPI + Swagger UI)
Fraud detection prediction API docs (test via Swagger UI):  
https://credit-card-fraud-detection-system-i6br.onrender.com/docs

---

## Project Overview

Credit card fraud detection is a critical problem for financial institutions. Fraudulent transactions represent a very small fraction of total transactions, making the dataset highly imbalanced.

Traditional accuracy metrics are misleading in this scenario because a model predicting all transactions as legitimate could still achieve very high accuracy.

This project focuses on building a system capable of identifying fraudulent transactions while maintaining strong recall and practical precision.

**The system includes:**
- Machine learning fraud detection model
- FastAPI prediction service
- Streamlit analytics dashboard
- Cloud deployment for both API and dashboard

---

## Problem Statement

Financial institutions process millions of transactions daily. Detecting fraudulent transactions in real time is essential to prevent financial losses.

**Key challenges:**
- Highly imbalanced datasets
- Rare fraud events
- High cost of false negatives
- Need for real-time prediction systems

**Goal:** Build a machine learning system that can detect fraudulent credit card transactions while handling severe class imbalance.

---

## Objectives

- Develop a fraud detection ML model capable of identifying fraudulent transactions.
- Handle severe class imbalance in financial transaction datasets.
- Deploy the trained model using a REST API.
- Build an interactive dashboard for fraud analytics and live predictions.
- Create an end-to-end deployable system accessible via the web.

---

## Dataset

The dataset used is the **European Credit Card Fraud Detection Dataset**.

- Kaggle link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Dataset characteristics:**
- Total transactions: **284,807**
- Fraudulent transactions: **492**
- Fraud rate: **0.172%**
- Highly imbalanced classification problem

**Features:**
- `Time`
- `Amount`
- PCA features `V1`–`V28` (anonymized principal components)
- `Class` (Fraud / Normal)

> Note: Because the dataset is large, the deployed dashboard uses a smaller sample dataset for visualization, while the machine learning model was trained using the full dataset.

---

## Machine Learning Pipeline

### Steps
- Data preprocessing
- Handling class imbalance
- Feature scaling
- Train-test split
- Feature selection
- Model training

### Models Evaluated
- Logistic Regression
- Random Forest

**Selected model:** Random Forest (better performance in detecting fraud cases)

---

## Evaluation Metrics

Since fraud detection is an imbalanced problem, the following metrics were used:
- Precision
- Recall
- F1 Score
- ROC-AUC

> Accuracy alone was not considered sufficient.

---

## Model Performance

**Best performing model:** Random Forest

**Key results:**
- ROC-AUC Score: **0.9839**
- Fraud Recall: **0.85**
- Fraud Precision: **0.43**

The model successfully identifies the majority of fraudulent transactions while keeping false positives manageable.

---

## System Architecture

```text
Streamlit Dashboard
        │
        │ REST API request
        ▼
FastAPI Fraud Detection API
        │
        │ Model inference
        ▼
Trained Fraud Detection Model
```

### Components
- **Dashboard:** Interactive analytics and live fraud prediction
- **API:** FastAPI service hosting the trained ML model
- **Model:** Random Forest fraud detection model trained using Scikit-learn

---

## API Usage

### Endpoint
`POST /predict`

### Example Request (Python)

```python
import requests

url = "https://credit-card-fraud-detection-system-i6br.onrender.com/predict"

sample_transaction = {
    "Time": 10000,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}

response = requests.post(url, json=sample_transaction)
print(response.json())
```

### Example Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.03,
  "risk_label": "Low Risk",
  "recommended_action": "Approve Transaction"
}
```

---

## Dashboard Features

The Streamlit dashboard provides:

### Dataset Overview
- Total transactions
- Fraud rate

### Fraud Analytics
- Fraud distribution
- Transaction amount distribution

### Live Fraud Prediction
- Sends transaction data to the API
- Receives fraud probability
- Displays risk classification

---

## Project Structure

```text
credit-card-fraud-detection-system
│
├── api
│   └── main.py
│
├── dashboard
│   └── app.py
│
├── models
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── data
│   └── sample dataset for dashboard
│
├── requirements.txt
└── README.md
```

---

## Challenges Faced

### Dataset Quality Issues
The initial dataset used contained a large number of duplicate rows which resulted in misleading model performance. Once duplicates were removed, performance dropped significantly, indicating data leakage.

To address this issue, a more reliable fraud detection dataset was selected.

### Severe Class Imbalance
Fraud cases represented less than 0.2% of transactions, so accuracy was not a reliable metric.

The model evaluation focused on:
- Recall for fraud cases
- ROC-AUC score
- Precision-Recall balance

### Deployment Issues
Several deployment challenges were resolved:
- API deployment configuration on Render
- Python dependency conflicts
- Streamlit deployment errors
- Python version compatibility issues
- Environment variable configuration between dashboard and API

---

## Conclusion

This project demonstrates how a machine learning fraud detection system can be developed and deployed as a full end-to-end application.

**Key outcomes:**
- Successful fraud detection model
- Deployed REST API
- Interactive analytics dashboard
- Cloud deployment of the entire system

The system illustrates how machine learning models can be integrated into real-world applications for financial risk detection.

---

## Future Enhancements

- Model interpretability using SHAP
- Threshold optimization for fraud detection
- Real-time transaction streaming
- Docker containerization
- CI/CD pipeline integration
- Automated model retraining pipeline
- Enhanced dashboard visualizations for fraud investigation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Render Cloud
- GitHub
