import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("mlruns")
# Load Data
print("Loading data....")
df = pd.read_csv("heart_preprocessing.csv")
print(f"shape{df.shape}")
 
# Mempersiapkan Fitur Dan Target
# Target : HeartDisease (0 = Tidak ada penyakit, 1 = Ada penyakit jantung)
 
# Fitur yang digunakan
features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_encoded', 'ChestPainType_encoded', 'RestingECG_encoded',
            'ExerciseAngina_encoded', 'ST_Slope_encoded']
target = 'HeartDisease'
 
X = df[features]
y = df[target]
print(f"Distribusi target:\n{y.value_counts()}")
 
# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape}, Test : {X_test.shape}")
 
# Training Menggunakan MLFLOW AUTOLOG
mlflow.set_tracking_uri("mlruns")
experiment = mlflow.set_experiment("heart-failure-modelling")
 
with mlflow.start_run(run_name="RandomForest_baseline") as run:
    mlflow.sklearn.autolog()
 
    # Train Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
 
    # Evaluasi Model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
 
    print(f"\nAccuracy : {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
 
    # CATATAN UNTUK ADVANCED CI:
    # Simpan path model hasil run ke file teks agar bisa dibaca oleh GitHub Actions
    experiment_id = experiment.experiment_id
    run_id = run.info.run_id
    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    with open("model_path.txt", "w") as f:
        f.write(model_path)
 
print("Model Berhasil Dilatih!!")