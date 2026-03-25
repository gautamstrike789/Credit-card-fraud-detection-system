import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    df = df.copy()

    # create hour feature
    df["transaction_hour"] = (df["Time"] // 3600) % 24

    # log transform amount
    df["amount_log"] = np.log1p(df["Amount"])

    df = df.drop(columns=["Time", "Amount"])

    return df


def create_train_test_data(df):

    df = preprocess_data(df)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler