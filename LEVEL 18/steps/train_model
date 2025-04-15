from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Any
@step
def train_model(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Trains a Linear Regression model on the cleaned DataFrame.
    Splits the data into train and test sets.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
