from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate_scores(self, y_true, y_pred):
        """Calculate evaluation scores."""
        pass

class MSE(Evaluation):
    """Mean Squared Error implementation."""
    
    def calculate_scores(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class R2(Evaluation):
    """R-squared implementation."""
    
    def calculate_scores(self, y_true, y_pred):
        return r2_score(y_true, y_pred)