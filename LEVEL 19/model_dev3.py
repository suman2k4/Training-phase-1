import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Load California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run() as run:
    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Define the model path
    model_path = "C:\\Users\\ASLAM\\Desktop\\Syncner Training\\Phase 1\\Level19\\model_dev3.py"  # Specify your desired path and filename

    # Save the model with a specific file name
    mlflow.sklearn.log_model(model, model_path)