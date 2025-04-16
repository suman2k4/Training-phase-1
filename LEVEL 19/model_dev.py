from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on given data.
        
        Args:
            X_train: Features for training.
            y_train: Labels for training.
        """
        pass  # No implementation (abstract method)

try:
    model = Model()  # This will fail (as expected)
except TypeError as e:
    print(f"Error: {e}")  # Expected output: Can't instantiate abstract class Model with abstract method train