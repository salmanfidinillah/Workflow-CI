# File ini untuk Kriteria Basic
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv("dataset_preprocessing.csv")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow & Autolog
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Membangun_Model_salmanfidinillah")
mlflow.sklearn.autolog()

# 3. Training (Autolog akan otomatis mencatat metrik, parameter, dan model)
with mlflow.start_run(run_name="Basic_RandomForest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Training selesai. Akurasi: {accuracy:.4f}")