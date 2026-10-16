"""Forecast-accuracy metrics shared by Mini-Projects 1 and 5.

All functions accept array-likes of equal, non-zero length and return floats.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("cannot score an empty series")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error, in the units of the data."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error, in the units of the data (penalises large misses)."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, in percent. Undefined if any actual is zero."""
    a, p = _pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def score_all(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and MAPE together as a dict."""
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted),
            "MAPE (%)": mape(actual, predicted)}
