# evaluation.py

# PURPOSE:
# Evaluate model performance using standard regression metrics

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    """
    R-squared score
    """
    return r2_score(y_true, y_pred)


def evaluate_model(y_true, y_pred):
    """
    Return all evaluation metrics as a dictionary
    """
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
    }


def print_metrics(y_true, y_pred):
    """
    Print evaluation metrics clearly
    """
    metrics = evaluate_model(y_true, y_pred)

    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
