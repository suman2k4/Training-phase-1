from zenml import pipeline, step
from model_dev2 import LinearRegressionModel
from sklearn.datasets import load_diabetes
from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
@step
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    return data.data, data.target

@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegressionModel()
    return model.train(X_train, y_train)

@pipeline
def linear_regression_pipeline():
    X, y = load_data()
    train_model(X, y)

if __name__ == "__main__":
    linear_regression_pipeline()
    print("Pipeline run completed! View in ZenML dashboard with:")
    print("zenml up")