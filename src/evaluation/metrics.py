from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error


def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))
