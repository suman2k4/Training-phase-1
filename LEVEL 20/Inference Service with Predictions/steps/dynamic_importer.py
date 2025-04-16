from zenml import step
import pandas as pd

@step
def dynamic_importer() -> pd.DataFrame:
    # Example test data for inference
    return pd.DataFrame({
        "feature1": [0.03],
        "feature2": [0.05],
        "feature3": [0.07],
        "feature4": [0.01],
        "feature5": [0.06],
        "feature6": [0.09],
        "feature7": [0.03],
        "feature8": [0.02],
        "feature9": [0.08],
        "feature10": [0.04],
    })
