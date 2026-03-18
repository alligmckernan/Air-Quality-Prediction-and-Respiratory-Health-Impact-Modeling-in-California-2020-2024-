# feature_engineering.py

# PURPOSE:
# Create features for time-series forecasting and regression models

import pandas as pd


def create_time_features(df, date_column):
    """
    Extract time-based features from a datetime column
    """
    df[date_column] = pd.to_datetime(df[date_column])

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["week"] = df[date_column].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df[date_column].dt.dayofweek

    return df


def create_lag_features(df, target_col, lags=[1, 2, 3]):
    """
    Create lag features for time-series models
    """
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(df, target_col, windows=[3, 7]):
    """
    Create rolling mean features
    """
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].rolling(window=window).mean()
        )

    return df


def drop_na_after_feature_engineering(df):
    """
    Drop NA values created by lag/rolling features
    """
    return df.dropna()
