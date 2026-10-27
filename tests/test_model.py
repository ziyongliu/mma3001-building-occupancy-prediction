"""
Unit tests for machine learning pipeline functions.
"""
import pytest
import pandas as pd
import numpy as np
from src.model import train_occupancy_model

def test_model_training_and_prediction():
    # Synthetic smoke test dataset
    X = pd.DataFrame({
        'Carbon dioxide': [400, 600, 450, 700],
        'Temperature': [20.0, 22.0, 21.0, 23.5],
        'Humidity': [50, 65, 52, 70],
        'Light Volatile Organic Compounds': [5, 20, 8, 25]
    })
    y = pd.Series([0, 1, 0, 1])
    
    model = train_occupancy_model(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 4
    assert set(preds).issubset({0, 1})
