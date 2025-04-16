from zenml import pipeline
from steps.dynamic_importer import dynamic_importer
from steps.load_model_service import load_model_service
from steps.predictor import predictor

@pipeline
def inference_pipeline():
    data = dynamic_importer()
    service = load_model_service()
    predictor(service=service, data=data)

