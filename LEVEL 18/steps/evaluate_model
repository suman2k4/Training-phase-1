from zenml import step
from sklearn.metrics import mean_squared_error
from typing import Any
import numpy as np

@step
def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> float:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return mse
