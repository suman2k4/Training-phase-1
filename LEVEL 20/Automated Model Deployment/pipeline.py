from zenml import step, pipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
from metrics_utils import serialize_metrics, deserialize_metrics
from model_dev import (
    LinearRegressionModel,
    RandomForestModel,
    MSEStrategy,
    RMSEStrategy,
    R2Strategy
)

@step
def ingest_data() -> pd.DataFrame:
    """Load diabetes dataset."""
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data['target'] = diabetes.target
    return data

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning."""
    return data.dropna()

@step
def train_and_evaluate(data: pd.DataFrame, model_type: str) -> str:
    """Train model and return serialized metrics."""
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegressionModel() if model_type == "linear" else RandomForestModel()
    trained_model = model.train(X_train, y_train)
    y_pred = trained_model.predict(X_test)
    
    return serialize_metrics(
        model_type=model_type,
        mse=MSEStrategy().evaluate(y_test, y_pred),
        rmse=RMSEStrategy().evaluate(y_test, y_pred),
        r2=R2Strategy().evaluate(y_test, y_pred)
    )

@step
def compare_results(linear_results: str, rf_results: str) -> None:
    """Compare and print results."""
    linear = deserialize_metrics(linear_results)
    rf = deserialize_metrics(rf_results)
    
    print("\nModel Comparison:")
    print(f"{linear['model_type']}: MSE={linear['mse']:.2f}, RMSE={linear['rmse']:.2f}, R²={linear['r2']:.2f}")
    print(f"{rf['model_type']}: MSE={rf['mse']:.2f}, RMSE={rf['rmse']:.2f}, R²={rf['r2']:.2f}")

@pipeline
def model_comparison_pipeline():
    """Complete comparison pipeline."""
    data = ingest_data()
    cleaned_data = clean_data(data)
    linear_metrics = train_and_evaluate(cleaned_data, "linear")
    rf_metrics = train_and_evaluate(cleaned_data, "random_forest")
    compare_results(linear_metrics, rf_metrics)

if __name__ == "__main__":
    model_comparison_pipeline()