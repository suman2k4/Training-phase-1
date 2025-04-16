from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

@step
def load_model_service() -> MLFlowDeploymentService:
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = deployer.find_model_server()
    if not services:
        raise RuntimeError("No active MLflow model service found.")
    service = services[0]
    if not service.is_running:
        service.start()
    return service
