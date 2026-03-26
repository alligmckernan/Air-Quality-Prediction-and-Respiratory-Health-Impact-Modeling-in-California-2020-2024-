import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =============================================================
# Time-Series Lag Features
# =============================================================

def add_time_series_lags(df, target_col, lags=[1, 2, 3]):
    """
    Add lag columns to a DataFrame for time-series modeling.
    Lag values represent the target variable shifted by 1, 2, 3 periods.
    Returns the DataFrame with new lag columns added in-place.
    """
    for lag in lags:
        df[f"lag{lag}"] = df[target_col].shift(lag)
    return df


def add_grouped_lag(df, group_col, target_col, lag=1):
    """
    Add a within-group lagged feature (e.g., prior year's AQI per county).
    Sorts by group and Year before shifting to ensure correct ordering.
    Returns the DataFrame with a new {target_col}_lag{lag} column added.
    """
    df = df.sort_values([group_col, "Year"])
    df[f"{target_col}_lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


# =============================================================
# Categorical Encoding
# =============================================================

def encode_county(df, county_col):
    """
    Label-encode a county column and append the result as County_encoded.
    Encoding is required to use county as a numeric feature in regression models.
    Returns (DataFrame with County_encoded column, fitted LabelEncoder).
    """
    le = LabelEncoder()
    df = df.copy()
    df["County_encoded"] = le.fit_transform(df[county_col])
    return df, le
