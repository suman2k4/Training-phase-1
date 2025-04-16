from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class MSEStrategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class RMSEStrategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

class R2Strategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

class LinearRegressionModel(Model):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

class RandomForestModel(Model):
    def __init__(self, n_estimators=100, max_depth=None):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model