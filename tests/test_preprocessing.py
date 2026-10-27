"""
Unit testing suite for preprocessing operations.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_interpolate_sensor_grid

def test_interpolation_eliminates_nans():
    df = pd.DataFrame({'co2': [400.0, np.nan, 450.0, np.nan, 500.0]})
    result = clean_and_interpolate_sensor_grid(df, method='linear')
    assert result.isnull().sum().sum() == 0
    assert result['co2'].iloc[1] == 425.0
