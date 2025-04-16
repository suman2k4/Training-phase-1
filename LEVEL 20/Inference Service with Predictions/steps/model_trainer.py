from zenml import step
from typing import Tuple
from sklearn.base import RegressorMixin
import pandas as pd
import numpy as np
from model_dev import preprocess_data, train_model

@step
def model_trainer(data: pd.DataFrame) -> Tuple[RegressorMixin, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Ensure output types match expected return types
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    model = train_model(X_train, y_train)
    return model, X_test, y_test
