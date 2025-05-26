import numpy as np

def r2_score(y_true, y_pred):
    """Calculate the R2 score."""
    ss_total = ((y_true - np.mean(y_true)) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    return 1 - (ss_residual / ss_total)

def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Calculate the Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_log_error(y_true, y_pred):
    """Calculate the Mean Squared Log Error."""
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def mean_relative_squared_error(y_true, y_pred):
    """Calculate the Mean Relative Squared Error."""
    return np.mean(((y_true - y_pred) / y_true) ** 2)