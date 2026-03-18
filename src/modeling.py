# modeling.py

# PURPOSE:
# Train models for AQI forecasting and respiratory health prediction

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# -----------------------------
# MODEL 1: AQI FORECASTING
# -----------------------------

def train_arima(series, order=(1, 1, 1)):
    """
    Train ARIMA model for AQI forecasting
    """
    model = ARIMA(series, order=order)
    return model.fit()


def train_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """
    Train SARIMA model for seasonal AQI forecasting
    """
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    return model.fit()


def train_rf_forecast(X_train, y_train):
    """
    Train Random Forest for AQI prediction using time features
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# MODEL 2: HEALTH IMPACT
# -----------------------------

def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model for health outcomes
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest for health outcome prediction
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting model for health outcomes
    """
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------

def forecast(model, steps=10):
    """
    Forecast future values for time-series models
    """
    return model.forecast(steps=steps)


def predict(model, X_test):
    """
    Generate predictions for regression models
    """
    return model.predict(X_test)
