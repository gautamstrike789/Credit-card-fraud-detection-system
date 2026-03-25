import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

from preprocess import load_data, create_train_test_data


DATA_PATH = "data/raw/creditcard.csv"


def train_and_save_model():

    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test, scaler = create_train_test_data(df)

    # balance dataset
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():

        print(f"\nTraining model: {name}")

        model.fit(X_train_resampled, y_train_resampled)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        auc = roc_auc_score(y_test, y_prob)

        print("\nROC AUC:", auc)

        if auc > best_auc:
            best_auc = auc
            best_model = model

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nBest model saved.")


if __name__ == "__main__":
    train_and_save_model()