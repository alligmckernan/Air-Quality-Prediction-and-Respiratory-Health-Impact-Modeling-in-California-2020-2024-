import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =============================================================
# Train / Test Split (Chronological)
# =============================================================

def chronological_split(df, train_frac=0.8):
    """
    Split a time-ordered DataFrame chronologically into train and test sets.
    Uses positional slicing to preserve time order — no random shuffling.
    This is the correct approach for any time-series or forecasting model.
    Returns (train_df, test_df).
    """
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:].copy()
    return train, test


# =============================================================
# Time-Series Models
# =============================================================

def fit_arima(train_series, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the training series.
    order = (p, d, q) — autoregressive, differencing, moving-average terms.
    Returns the fitted ARIMA results object.
    """
    model = ARIMA(train_series, order=order)
    return model.fit()


def fit_sarima(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """
    Fit a SARIMA model to capture both trend and seasonal patterns.
    seasonal_order = (P, D, Q, s) where s=52 for weekly data (annual seasonality).
    Returns the fitted SARIMAX results object.
    """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    return model.fit()


def fit_rf_timeseries(train_df, test_df, features, target, n_estimators=300, random_state=42):
    """
    Train a Random Forest regressor using lagged time-series features.
    Uses lag1, lag2, lag3 as predictors instead of raw time index.
    Returns (fitted RandomForestRegressor, predictions array for test set).
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(train_df[features], train_df[target])
    preds = rf.predict(test_df[features])
    return rf, preds


# =============================================================
# Health Impact Regression Models
# =============================================================

def fit_health_models(X_train, y_train, X_test, random_state=42):
    """
    Train Linear Regression, Random Forest, and Gradient Boosting regressors
    on the same training data and generate predictions for the test set.
    Returns a dict: {model_name: predictions_array}.
    """
    models = {
        "Linear Regression":   LinearRegression(),
        "Random Forest":       RandomForestRegressor(n_estimators=300, random_state=random_state),
        "Gradient Boosting":   GradientBoostingRegressor(random_state=random_state),
    }
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
    return predictions
