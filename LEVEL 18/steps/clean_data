from zenml import step
import pandas as pd
@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by dropping columns and filling missing values.
    """
    if 'unnecessary_column' in df.columns:
        df = df.drop(columns=['unnecessary_column'])

    df = df.fillna(df.median(numeric_only=True))

    df = df.fillna("no review")
    
    return df
