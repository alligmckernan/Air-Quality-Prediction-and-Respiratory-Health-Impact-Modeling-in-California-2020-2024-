import pandas as pd


# =========================
# Lag Features (AQI)
# =========================
def add_aqi_lags(df, column="mean_aqi", lags=[1, 2, 3]):
    """
    Create lagged AQI features for time-series modeling
    """
    for lag in lags:
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


# =========================
# Date Features
# =========================
def add_date_features(df):
    """
    Extract useful time-based features
    """
    if "weekend" in df.columns:
        df["year"] = df["weekend"].dt.year
        df["month"] = df["weekend"].dt.month
    return df


# =========================
# Drop Missing from Lags
# =========================
def drop_na_lags(df):
    """
    Remove rows with NA values created by lagging
    """
    return df.dropna()
