import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

def evaluate_model():
    # Load the model using the specified file name
    model_path = "C:\\Users\\ASLAM\\Desktop\\Syncner Training\\Phase 1\\Level19\\model_dev3.py"  # Use the same path as when saving
    model = mlflow.sklearn.load_model(model_path)

    # Load California housing dataset
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse_score = mean_squared_error(y_test, predictions)
    r2_score_value = r2_score(y_test, predictions)

    # Log metrics
    with mlflow.start_run():
        mlflow.log_metric("mse", mse_score)
        mlflow.log_metric("r2", r2_score_value)

# If you want to run this script directly for testing
if __name__ == "__main__":
    evaluate_model()