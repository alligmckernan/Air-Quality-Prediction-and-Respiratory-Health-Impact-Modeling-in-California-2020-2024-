import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================
# AQI Time Series Plot
# =============================================================

def plot_weekly_aqi(weekly_state, title="Population-Weighted Weekly Statewide AQI"):
    """
    Plot population-weighted weekly AQI time series with Good/Moderate reference lines.
    Expects a DataFrame with columns: WeekEnd (datetime), PopWeighted_AQI (float).
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(weekly_state["WeekEnd"], weekly_state["PopWeighted_AQI"],
            color="steelblue", linewidth=1.2, marker="o", markersize=2)
    ax.axhline(50,  color="green",  linestyle="--", linewidth=0.9, label="Good (50)")
    ax.axhline(100, color="orange", linestyle="--", linewidth=0.9, label="Moderate (100)")
    ax.set_title(title)
    ax.set_ylabel("AQI (Pop-Weighted)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =============================================================
# Forecast Comparison Plot
# =============================================================

def plot_forecast_comparison(train, test, rf_test):
    """
    Plot actual vs forecasted AQI for ARIMA, SARIMA, and Random Forest.
    train   — training DataFrame with WeekEnd and PopWeighted_AQI columns
    test    — test DataFrame with WeekEnd, PopWeighted_AQI, ARIMA_Forecast, SARIMA_Forecast
    rf_test — test DataFrame with WeekEnd and RF_Forecast columns
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["WeekEnd"],   train["PopWeighted_AQI"],  label="Train")
    ax.plot(test["WeekEnd"],    test["PopWeighted_AQI"],   label="Actual")
    ax.plot(test["WeekEnd"],    test["ARIMA_Forecast"],    label="ARIMA")
    ax.plot(test["WeekEnd"],    test["SARIMA_Forecast"],   label="SARIMA")
    ax.plot(rf_test["WeekEnd"], rf_test["RF_Forecast"],    label="Random Forest")
    ax.legend()
    ax.set_title("AQI Forecast Model Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    plt.tight_layout()
    return fig


# =============================================================
# Feature Importance Plot
# =============================================================

def plot_feature_importance(importances, features, title="Feature Importance"):
    """
    Plot a horizontal bar chart of model feature importances.
    importances — array of importance values (e.g., rf.feature_importances_)
    features    — list of feature names matching the importance array
    Returns the matplotlib Figure object.
    """
    importance_series = pd.Series(importances, index=features)
    fig, ax = plt.subplots()
    importance_series.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
