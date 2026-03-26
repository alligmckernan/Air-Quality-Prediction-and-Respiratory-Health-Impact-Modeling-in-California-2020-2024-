import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================
# Core Metric Calculation
# =============================================================

def evaluate_model(true, pred):
    """
    Calculate MAE, RMSE, and R² for a set of model predictions.
    MAE  — Mean Absolute Error (average magnitude of errors)
    RMSE — Root Mean Squared Error (penalizes large errors more)
    R²   — Coefficient of determination (1.0 = perfect fit)
    Returns (mae, rmse, r2) as floats.
    """
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2   = r2_score(true, pred)
    return mae, rmse, r2


# =============================================================
# Results Tables
# =============================================================

def build_aqi_metrics_table(model_names, true_vals, pred_lists):
    """
    Build a summary DataFrame of AQI forecast metrics across multiple models.
    model_names : list of str — model labels
    true_vals   : array-like — actual AQI values
    pred_lists  : list of arrays — one prediction array per model
    Returns a DataFrame with columns: Model, MAE, RMSE, R2.
    """
    rows = []
    for name, preds in zip(model_names, pred_lists):
        mae, rmse, r2 = evaluate_model(true_vals, preds)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows)


def build_health_metrics_table(model_names, true_hosp, preds_hosp, true_ed, preds_ed):
    """
    Build a summary DataFrame of health model metrics for both targets
    (hospitalization rate and ED visit rate) across multiple models.
    Returns a DataFrame with columns: Model, Hosp_MAE, Hosp_RMSE, Hosp_R2,
    ED_MAE, ED_RMSE, ED_R2.
    """
    rows = []
    for name, pred_h, pred_e in zip(model_names, preds_hosp, preds_ed):
        mae_h, rmse_h, r2_h = evaluate_model(true_hosp, pred_h)
        mae_e, rmse_e, r2_e = evaluate_model(true_ed,   pred_e)
        rows.append({
            "Model":     name,
            "Hosp_MAE":  round(mae_h, 4), "Hosp_RMSE": round(rmse_h, 4), "Hosp_R2": round(r2_h, 4),
            "ED_MAE":    round(mae_e, 4), "ED_RMSE":   round(rmse_e, 4), "ED_R2":   round(r2_e, 4),
        })
    return pd.DataFrame(rows)
