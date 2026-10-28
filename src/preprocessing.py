"""
Module for cleaning and interpolating building sensor telemetry.
"""
import pandas as pd
import numpy as np

def clean_and_interpolate_sensor_grid(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Interpolates missing telemetry across a regular 5-minute time series grid.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame containing missing values (NaNs).
        method (str): Interpolation scheme ('linear' or 'spline').

    Returns:
        pd.DataFrame: Imputed DataFrame with zero remaining NaNs.
    """
    if method == 'spline':
        return df.interpolate(method='spline', order=2).bfill().ffill()
    return df.interpolate(method='linear').bfill().ffill()
