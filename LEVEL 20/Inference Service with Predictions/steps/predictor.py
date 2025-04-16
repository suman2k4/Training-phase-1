from zenml import step
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
import pandas as pd
import numpy as np
import requests

@step
def predictor(service: MLFlowDeploymentService, data: pd.DataFrame) -> np.ndarray:
    if not service.is_running:
        service.start()

    prediction_url = service.predict_url
    response = requests.post(
        prediction_url,
        json={"instances": data.to_dict(orient="records")}
    )

    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed: {response.text}")

    predictions = response.json()["predictions"]
    print("Predictions:", predictions)
    return np.array(predictions)
