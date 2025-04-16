from zenml import step
import numpy as np
from sklearn.base import RegressorMixin
from metrics_utils import evaluate_model

@step
def deployment_trigger(model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray) -> bool:
    r2 = evaluate_model(model, X_test, y_test)
    print(f"R2 Score: {r2}")
    return r2 >= 0.7
