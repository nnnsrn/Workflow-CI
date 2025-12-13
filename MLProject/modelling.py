import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Kriteria_2_Linear_Regression")

df = pd.read_csv("drug_dataset_preprocessing/train_clean.csv")

# choose a valid target column
y = df["Effectiveness"]
X = df.drop(columns=["Effectiveness"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_train.head(5)
    )

print("DONE")
