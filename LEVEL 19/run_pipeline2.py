from zenml import pipeline, step
from model_dev2 import LinearRegressionModel
from evaluation import MSE, R2
from sklearn.datasets import load_diabetes
from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression

@step
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split data into train/test sets."""
    data = load_diabetes()
    X, y = data.data, data.target
    # Simple split (for demo; use sklearn's train_test_split in practice)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:]

@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train the model."""
    model = LinearRegressionModel()
    return model.train(X_train, y_train)

@step
def evaluate_model(
    model: LinearRegression, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[float, float]:
    """Evaluate model using MSE and R2."""
    y_pred = model.predict(X_test)
    mse = MSE().calculate_scores(y_test, y_pred)
    r2 = R2().calculate_scores(y_test, y_pred)
    return mse, r2

@pipeline
def linear_regression_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run = linear_regression_pipeline()
    print("Pipeline run completed! View metrics in ZenML dashboard:")
    print("zenml up")