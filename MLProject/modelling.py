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
# Kita TIDAK PAKAI set_experiment di sini biar nggak tabrakan sama command 'mlflow run'
mlflow.sklearn.autolog()

# 3. Training
# Parameter di dalam start_run() kita kosongin biar dia otomatis nerusin Run ID dari GitHub Actions
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Training selesai. Akurasi: {accuracy:.4f}")