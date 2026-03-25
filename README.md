Credit Card Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions, including a trained ML model, REST API, and an interactive analytics dashboard.

The project demonstrates how a fraud detection model can be developed, deployed, and served through a real-time prediction system.

Live System
Dashboard

Interactive analytics dashboard

[https://credit-card-fraud-detection-system-yhnycblp3dxrfxmklt7hg.streamlit.app](https://credit-card-fraud-detection-system-3v9qroqcgendr2wuwcksn5.streamlit.app/)
API Endpoint

Fraud detection prediction API

https://credit-card-fraud-detection-system-i6br.onrender.com/docs

The API documentation page allows testing predictions directly through Swagger UI.

Project Overview

Credit card fraud detection is a critical problem for financial institutions. Fraudulent transactions represent a very small fraction of total transactions, making the dataset highly imbalanced.

Traditional accuracy metrics are misleading in this scenario because a model predicting all transactions as legitimate could still achieve very high accuracy.

This project focuses on building a system capable of identifying fraudulent transactions while maintaining strong recall and practical precision.

The system includes:

Machine learning fraud detection model
FastAPI prediction service
Streamlit analytics dashboard
Cloud deployment for both API and dashboard
Problem Statement

Financial institutions process millions of transactions daily. Detecting fraudulent transactions in real time is essential to prevent financial losses.

However, fraud detection presents several challenges:

Highly imbalanced datasets
Rare fraud events
High cost of false negatives
Need for real-time prediction systems

The goal of this project is to build a machine learning system that can detect fraudulent credit card transactions while handling severe class imbalance.

Objective

The primary objectives of this project were:

Develop a fraud detection machine learning model capable of identifying fraudulent transactions.
Handle severe class imbalance in financial transaction datasets.
Deploy the trained model using a REST API.
Build an interactive dashboard for fraud analytics and live predictions.
Create an end-to-end deployable system accessible via the web.
Dataset

The dataset used is the European Credit Card Fraud Detection Dataset.

Direct dataset link:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset characteristics:

Total transactions: 284,807
Fraudulent transactions: 492
Fraud rate: 0.172%
Highly imbalanced classification problem

The dataset features are anonymized principal components V1–V28 obtained using PCA transformation to protect sensitive financial data.

Key fields include:

Time
Amount
PCA features (V1–V28)
Class (Fraud / Normal)

Because the dataset is large, the deployed dashboard uses a smaller sample dataset for visualization while the machine learning model was trained using the full dataset.

Machine Learning Pipeline

The model development pipeline includes:

Data Preprocessing
Handling class imbalance
Feature scaling
Train-test split
Feature selection
Model Training

Multiple models were evaluated including:

Logistic Regression
Random Forest

Random Forest was selected due to better performance in detecting fraud cases.

Evaluation Metrics

Since fraud detection is an imbalanced problem, the following metrics were used:

Precision
Recall
F1 Score
ROC-AUC

Accuracy alone was not considered sufficient.

Model Performance

Best performing model: Random Forest

Key results:

ROC-AUC Score: 0.9839
Fraud Recall: 0.85
Fraud Precision: 0.43

The model successfully identifies the majority of fraudulent transactions while keeping false positives manageable.

System Architecture

The system follows a modular architecture:

Streamlit Dashboard
        │
        │ REST API request
        ▼
FastAPI Fraud Detection API
        │
        │ Model inference
        ▼
Trained Fraud Detection Model

Components:

Dashboard
Interactive analytics and live fraud prediction

API
FastAPI service hosting the trained ML model

Model
Random Forest fraud detection model trained using Scikit-learn

API Request Example

The API accepts transaction features and returns a fraud probability along with a risk classification.

Example request using Python

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

Example API response:

{
  "prediction": 0,
  "fraud_probability": 0.03,
  "risk_label": "Low Risk",
  "recommended_action": "Approve Transaction"
}
Dashboard Features

The Streamlit dashboard provides:

Dataset overview

total transactions
fraud rate

Fraud analytics

fraud distribution
transaction amount distribution

Live fraud prediction

sends transaction data to the API
receives fraud probability
displays risk classification
Project Structure
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
Challenges Faced

Several challenges were encountered during development.

Dataset Quality Issues

The initial dataset used contained a large number of duplicate rows which resulted in misleading model performance. Once duplicates were removed, performance dropped significantly, indicating data leakage.

To address this issue, a more reliable fraud detection dataset was selected.

Severe Class Imbalance

Fraud cases represented less than 0.2% of transactions. Accuracy was therefore not a reliable metric.

The model evaluation focused on:

Recall for fraud cases
ROC-AUC score
Precision-Recall balance
Deployment Issues

Several deployment challenges were resolved:

API deployment configuration on Render
Python dependency conflicts
Streamlit deployment errors
Python version compatibility issues
Environment variable configuration between dashboard and API
Conclusion

This project demonstrates how a machine learning fraud detection system can be developed and deployed as a full end-to-end application.

Key outcomes include:

successful fraud detection model
deployed REST API
interactive analytics dashboard
cloud deployment of the entire system

The system illustrates how machine learning models can be integrated into real-world applications for financial risk detection.

Future Enhancements

Possible improvements include:

Model interpretability using SHAP
Threshold optimization for fraud detection
Real-time transaction streaming
Docker containerization
CI/CD pipeline integration
Automated model retraining pipeline
Enhanced dashboard visualizations for fraud investigation
Technologies Used

Python
Pandas
NumPy
Scikit-learn
FastAPI
Streamlit
Render Cloud
GitHub
