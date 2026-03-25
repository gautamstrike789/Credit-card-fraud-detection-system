import os
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection System")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dashboard/sample_creditcard.csv")

# ---------------- PROJECT OVERVIEW ----------------
st.header("Project Overview")
st.write(
    """
This project detects fraudulent credit card transactions using machine learning.
The dataset is highly imbalanced, so fraud detection performance is evaluated using
precision, recall, F1-score, and ROC-AUC instead of accuracy alone.
"""
)

# ---------------- DATASET OVERVIEW ----------------
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Fraud Transactions", int(df["Class"].sum()))
col3.metric("Fraud Rate", f"{df['Class'].mean() * 100:.2f}%")

# ---------------- MODEL PERFORMANCE ----------------
st.header("Model Performance")

st.write("**Best Model:** Random Forest")
st.write("**ROC-AUC:** 0.9839")
st.write("**Fraud Recall:** 0.85")
st.write("**Fraud Precision:** 0.43")

st.info(
    "Recall is important in fraud detection because missing a fraudulent transaction "
    "can cause direct financial loss. Precision matters too, because too many false alarms "
    "increase manual review workload."
)

# Manual confusion matrix from your model output
cm = np.array([[56753, 111],
               [15, 83]])

fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm)

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Predicted Normal", "Predicted Fraud"])
ax_cm.set_yticklabels(["Actual Normal", "Actual Fraud"])
ax_cm.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)

# ---------------- FRAUD DISTRIBUTION ----------------
st.header("Fraud Distribution")

fig, ax = plt.subplots()
df["Class"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xticklabels(["Normal", "Fraud"], rotation=0)
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------- AMOUNT DISTRIBUTION ----------------
st.header("Log Transaction Amount Distribution")

df_plot = df.copy()
df_plot["Amount_log"] = np.log1p(df_plot["Amount"])

fig2, ax2 = plt.subplots()
df_plot[df_plot["Class"] == 0]["Amount_log"].hist(
    bins=50, alpha=0.5, label="Normal", ax=ax2
)
df_plot[df_plot["Class"] == 1]["Amount_log"].hist(
    bins=50, alpha=0.5, label="Fraud", ax=ax2
)

ax2.set_xlabel("Log(Amount + 1)")
ax2.set_ylabel("Frequency")
ax2.legend()
st.pyplot(fig2)

# ---------------- RANDOM SAMPLE SELECTION ----------------
st.header("Live Fraud Prediction")

sample_type = st.selectbox(
    "Select sample transaction type",
    ["Real Normal Transaction", "Real Fraud Transaction"]
)

if st.button("Load Random Sample"):
    if sample_type == "Real Normal Transaction":
        st.session_state["sample_row"] = df[df["Class"] == 0].sample(1, random_state=None).iloc[0]
    else:
        st.session_state["sample_row"] = df[df["Class"] == 1].sample(1, random_state=None).iloc[0]

# Default sample if nothing loaded yet
if "sample_row" not in st.session_state:
    st.session_state["sample_row"] = df[df["Class"] == 0].iloc[0]

sample_row = st.session_state["sample_row"]
actual_class = int(sample_row["Class"])
payload = sample_row.drop("Class").to_dict()
payload = {k: float(v) for k, v in payload.items()}

st.subheader("Transaction Input")
st.json(payload)

st.write(f"**Actual Class in Dataset:** {'Fraud' if actual_class == 1 else 'Normal'}")

if st.button("Predict Fraud Risk"):
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        predicted_class = result["prediction"]

        st.success(f"Predicted Risk Label: {result['risk_label']}")
        st.metric("Fraud Probability", result["fraud_probability"])
        st.write("**Recommended Action:**", result["recommended_action"])
        st.write(f"**Predicted Class:** {'Fraud' if predicted_class == 1 else 'Normal'}")

        if predicted_class == actual_class:
            st.success("Prediction matches the actual dataset label.")
        else:
            st.error("Prediction does NOT match the actual dataset label.")

        st.json(result)
    else:
        st.error("API request failed. Make sure FastAPI is running.")