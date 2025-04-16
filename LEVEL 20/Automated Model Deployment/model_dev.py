from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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
    def train(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

class RandomForestModel(Model):
    def train(self, X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model