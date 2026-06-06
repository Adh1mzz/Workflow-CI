import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set tracking URI to local mlruns
mlflow.set_tracking_uri("mlruns")

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Heart Disease'],
                yticklabels=['Normal', 'Heart Disease'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color='#3498db')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heart_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}....")
    df = pd.read_csv(args.data_path)
    print(f"Shape: {df.shape}")

    # Mempersiapkan Fitur Dan Target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    print(f"Features: {list(X.columns)}")
    print(f"Distribusi target:\n{y.value_counts()}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # MLflow tracking
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("data_path", args.data_path)

        # Train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Log metrics
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))

        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artifacts
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        fi_path = plot_feature_importance(model, list(X_train.columns))
        mlflow.log_artifact(fi_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        # Simpan path model hasil run ke file teks agar bisa dibaca oleh GitHub Actions
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
        with open("model_path.txt", "w") as f:
            f.write(model_path)
            
        print("Model Berhasil Dilatih dan Dilog!!")

        # Cleanup
        for f in [cm_path, fi_path, "classification_report.json"]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()