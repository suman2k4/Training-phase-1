from zenml import step
from sklearn.datasets import load_diabetes
import pandas as pd

@step
def ingest_data() -> pd.DataFrame:
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame
    df["target"] = dataset.target
    return df
