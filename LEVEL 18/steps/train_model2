from zenml import step
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Tuple, Any
@step
def train_model(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Trains a Linear Regression model on the cleaned DataFrame and logs it to MLflow.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear_regression_model")
        mlflow.log_params({"model_type": "Linear Regression"})
        mlflow.log_metrics({"mse": mean_squared_error(y_test, model.predict(X_test))})

    return model, X_train, X_test, y_train, y_test
