"""
Unit tests for data preprocessing and interpolation.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_interpolate_sensor_grid

def test_interpolation_removes_nans():
    # Synthetic test series with missing points
    sample_data = pd.DataFrame({'co2': [400.0, np.nan, 450.0, np.nan, 500.0]})
    clean_data = clean_and_interpolate_sensor_grid(sample_data, method='linear')

    # Assertions
    assert clean_data.isnull().sum().sum() == 0
    assert clean_data['co2'].iloc[1] == 425.0
