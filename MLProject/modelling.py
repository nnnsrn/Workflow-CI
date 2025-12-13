import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import os
import shutil

if "MLFLOW_RUN_ID" in os.environ:
    print(f"⚠️ Menghapus MLFLOW_RUN_ID lama: {os.environ['MLFLOW_RUN_ID']}")
    del os.environ["MLFLOW_RUN_ID"]

script_dir = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(script_dir, "drug_dataset_preprocessing", "train_clean.csv")

if not os.path.exists(train_path):
    train_path = os.path.join(script_dir, "train_clean.csv")

if not os.path.exists(train_path):
    train_path = "drug_dataset_preprocessing/train_clean.csv"

print(f"📂 Menggunakan dataset di: {train_path}")

try:
    df = pd.read_csv(train_path)
except FileNotFoundError:
    raise FileNotFoundError("❌ ERROR: File train_clean.csv tidak ditemukan! Pastikan struktur folder benar.")

df = df.dropna()

y = df["Effectiveness"]
X = df.drop(columns=["Effectiveness"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

mlflow.set_experiment("Kriteria_2_Linear_Regression_Auto")

with mlflow.start_run():
    model = LinearRegression()

    model.fit(X_train, y_train)


print("DONE")
