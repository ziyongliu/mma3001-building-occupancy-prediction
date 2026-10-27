"""
Module containing machine learning model definitions for occupancy prediction.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

def train_occupancy_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains and returns a production Random Forest classifier pipeline.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Binary target vector (0 or 1).

    Returns:
        Pipeline: Fitted Scikit-Learn Pipeline object.
    """
    pipeline = Pipeline([
        ('model', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
