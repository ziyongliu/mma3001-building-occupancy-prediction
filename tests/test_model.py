"""
Unit tests for model training and prediction functions.
"""
import pytest
import pandas as pd
from src.model import train_occupancy_model

def test_model_training_and_prediction():
    # Small synthetic test set
    X = pd.DataFrame({
        'Carbon dioxide': [400, 600, 450, 700],
        'Temperature': [20.0, 22.0, 21.0, 23.5],
        'Humidity': [50, 65, 52, 70],
        'Light Volatile Organic Compounds': [5, 20, 8, 25]
    })
    y = pd.Series([0, 1, 0, 1])
    
    model = train_occupancy_model(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})
