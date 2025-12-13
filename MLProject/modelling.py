import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os  

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "train_clean.csv")

print(f"📂 Mencari dataset di: {train_path}")

if not os.path.exists(train_path):
    project_root = os.path.dirname(script_dir)
    fallback_path = os.path.join(project_root, "preprocessing", "drug_dataset_preprocessing", "train_clean.csv")
    
    print(f"⚠️ File tidak ada di script_dir. Mencoba mencari di: {fallback_path}")
    if os.path.exists(fallback_path):
        train_path = fallback_path
    else:
        raise FileNotFoundError(f"❌ FATAL ERROR: File 'train_clean.csv' tidak ditemukan dimanapun!")

df = pd.read_csv(train_path)
print("✅ Dataset berhasil di-load!")

df = df.dropna()

y = df["Effectiveness"]
X = df.drop(columns=["Effectiveness"])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Kriteria_2_Linear_Regression")

with mlflow.start_run():
    model = LinearRegression()
    # Training
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    # Logging Metrics
    print(f"📊 MSE Score: {mse}")
    mlflow.log_metric("mse", mse)

    signature = mlflow.models.infer_signature(X_train, preds)
    
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_train.head(5),
        signature=signature
    )

print("DONE!")
