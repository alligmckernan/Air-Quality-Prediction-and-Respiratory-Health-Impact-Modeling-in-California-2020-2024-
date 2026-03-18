import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Regression Metrics
# =========================
def regression_metrics(y_true, y_pred):
    """
    Return MAE, RMSE, and R2
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred)
    }


# =========================
# Compare Models
# =========================
def compare_models(results_dict):
    """
    Convert multiple model results into DataFrame
    """
    return pd.DataFrame(results_dict)


# =========================
# Save Results
# =========================
def save_results(df, filepath):
    """
    Save evaluation results
    """
    df.to_csv(filepath, index=False)
