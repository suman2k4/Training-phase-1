from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.model_trainer import model_trainer
from steps.deployment_trigger import deployment_trigger
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

@pipeline
def training_pipeline():
    data = ingest_data()
    model, X_test, y_test = model_trainer(data)
    decision = deployment_trigger(model, X_test, y_test)
    mlflow_model_deployer_step(model=model, deploy_decision=decision)

