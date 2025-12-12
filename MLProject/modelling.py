import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Load dataset
if __name__ == "__main__":
    data_path = "train_clean.csv" 
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        train_model(df)
    else:
        print(f"Error: File tidak ditemukan di {data_path}")

X = df.drop("Deposit", axis=1)
y = df["Deposit"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow autolog
mlflow.autolog()

with mlflow.start_run():

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metric
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

print("Training selesai. Cek MLflow UI.")
