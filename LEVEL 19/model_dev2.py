from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        """Trains a sklearn LinearRegression model."""
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg  # Return the trained model

# Example usage (testing without ZenML)
if __name__ == "__main__":
    # Load sample data
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Train and test
    lr_model = LinearRegressionModel()
    trained_model = lr_model.train(X, y)
    print(f"Model coefficients: {trained_model.coef_}")  # Verify training