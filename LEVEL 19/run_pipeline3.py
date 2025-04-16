from zenml.pipelines import pipeline
from zenml.steps import step

@step(experiment_tracker="mlflow_tracker")
def model_training():
    # Call the model training function from model_dev.py
    import model_dev  # Ensure model_dev.py is in the same directory or adjust the import accordingly
    model_dev.train_model()  # Assuming you have a function named train_model in model_dev.py

@step(experiment_tracker="mlflow_tracker")
def model_evaluation():
    # Call the evaluation function from evaluation.py
    import evaluation2  # Ensure evaluation.py is in the same directory or adjust the import accordingly
    evaluation2.evaluate_model()  # Call the evaluate_model function

@pipeline
def training_pipeline():
    train = model_training()
    evaluate = model_evaluation()

# Run the pipeline
if __name__ == "__main__":
    training_pipeline()