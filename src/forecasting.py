"""Abstract forecasting interface plus the model families used in Mini-Projects 1 and 5.

Every forecaster follows the same contract::

    model = SomeForecaster(...).fit(y, t)   # t (time index) is optional
    model.predict(horizon)                   # -> np.ndarray of length `horizon`
    model.fitted_values                      # in-sample fit (NaN where undefined)

Keeping one interface means backtesting, model selection and plotting code is
written once and works for every model (polymorphism instead of if/else chains).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Forecaster(ABC):
    """Base class: stores the training series and enforces fit-before-predict."""

    #: minimum number of observations the model needs to be fitted
    min_obs: int = 2
    name: str = "Forecaster"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._t: np.ndarray | None = None

    # ---- template method -------------------------------------------------
    def fit(self, y: ArrayLike, t: ArrayLike | None = None) -> "Forecaster":
        """Validate the series, store it, then delegate to :meth:`_fit`."""
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if y_arr.size < self.min_obs:
            raise ValueError(f"{self.name} needs at least {self.min_obs} observations, got {y_arr.size}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN or infinite values")
        t_arr = np.arange(y_arr.size, dtype=float) if t is None else np.asarray(t, dtype=float)
        if t_arr.shape != y_arr.shape:
            raise ValueError("t and y must have the same length")
        self._y, self._t = y_arr, t_arr
        self._fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast `horizon` steps beyond the end of the training series."""
        if self._y is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return self._predict(int(horizon))

    @property
    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values (NaN where the model cannot produce one)."""
        if self._y is None:
            raise RuntimeError("model is not fitted")
        return self._fitted()

    @property
    def residuals(self) -> np.ndarray:
        """Actual minus fitted, NaN where fitted is undefined."""
        return self._y - self.fitted_values

    # ---- hooks for subclasses ---------------------------------------------
    @abstractmethod
    def _fit(self) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    @abstractmethod
    def _fitted(self) -> np.ndarray: ...

    def _future_t(self, horizon: int) -> np.ndarray:
        step = self._t[1] - self._t[0] if self._t.size > 1 else 1.0
        return self._t[-1] + step * np.arange(1, horizon + 1)

    def __repr__(self) -> str:
        state = "fitted" if self._y is not None else "unfitted"
        return f"{type(self).__name__}({state})"


class LinearTrendForecaster(Forecaster):
    """Ordinary least-squares straight line y = a + b·t, fitted with ``np.polyfit``."""

    name = "Linear trend"

    def _fit(self) -> None:
        self.slope, self.intercept = np.polyfit(self._t, self._y, deg=1)

    def _line(self, t: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * t

    def _predict(self, horizon: int) -> np.ndarray:
        return self._line(self._future_t(horizon))

    def _fitted(self) -> np.ndarray:
        return self._line(self._t)


class CAGRForecaster(Forecaster):
    """Constant-percentage (exponential) growth at the compound annual growth rate.

    CAGR = (y_last / y_first) ** (1 / (n - 1)) - 1, and forecasts are
    y_last · (1 + CAGR) ** h.
    """

    name = "Exponential (CAGR)"

    def _fit(self) -> None:
        if np.any(self._y <= 0):
            raise ValueError("CAGR needs strictly positive values")
        self.rate = (self._y[-1] / self._y[0]) ** (1 / (self._y.size - 1)) - 1

    def _predict(self, horizon: int) -> np.ndarray:
        return self._y[-1] * (1 + self.rate) ** np.arange(1, horizon + 1)

    def _fitted(self) -> np.ndarray:
        return self._y[0] * (1 + self.rate) ** np.arange(self._y.size)


def fibonacci(n: int) -> np.ndarray:
    """First `n` Fibonacci numbers starting 1, 1, 2, 3, ... (integer array)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out = np.zeros(n, dtype=np.int64)
    a, b = 1, 1
    for i in range(n):
        out[i] = a
        a, b = b, a + b
    return out


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: scale the last value by successive Fibonacci ratios.

    Forecast h is y_last · Π_{k=start}^{start+h-1} F(k+1)/F(k). With the default
    ``start=1`` the ratios are 1, 2, 1.5, 1.67, 1.6, ... which converge quickly to the
    golden ratio φ ≈ 1.618, i.e. roughly +62% growth per step regardless of the data.
    The model has no parameter learned from the data, which is its core weakness.
    In-sample "fitted" values use the limiting ratio φ one step ahead.
    """

    name = "Fibonacci ratio"
    PHI = (1 + 5 ** 0.5) / 2

    def __init__(self, start: int = 1) -> None:
        super().__init__()
        if start < 1:
            raise ValueError("start must be >= 1")
        self.start = start

    def _fit(self) -> None:  # nothing to estimate: the ratios are fixed in advance
        pass

    def ratios(self, horizon: int) -> np.ndarray:
        fib = fibonacci(self.start + horizon + 1).astype(float)
        idx = np.arange(self.start - 1, self.start - 1 + horizon)
        return fib[idx + 1] / fib[idx]

    def _predict(self, horizon: int) -> np.ndarray:
        return self._y[-1] * np.cumprod(self.ratios(horizon))

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[1:] = self._y[:-1] * self.PHI
        return out


class MovingAverageForecaster(Forecaster):
    """k-period simple moving average; the forecast is flat at the last window's mean."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.min_obs = window
        self.name = f"{window}-day moving average"

    def _fit(self) -> None:
        self.level = float(np.mean(self._y[-self.window:]))

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        for i in range(self.window, self._y.size):
            out[i] = self._y[i - self.window:i].mean()
        return out


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing: level_t = α·y_t + (1−α)·level_{t−1}."""

    min_obs = 1

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (α={alpha:.2f})"

    def _fit(self) -> None:
        levels = np.empty(self._y.size)
        levels[0] = self._y[0]  # initialise with the first observation
        for i in range(1, self._y.size):
            levels[i] = self.alpha * self._y[i] + (1 - self.alpha) * levels[i - 1]
        self._levels = levels
        self.level = float(levels[-1])

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[1:] = self._levels[:-1]  # one-step-ahead fit
        return out


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value observed one season (e.g. 7 days) earlier."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.min_obs = period
        self.name = f"Seasonal naïve (m={period})"

    def _fit(self) -> None:
        self._last_season = self._y[-self.period:]

    def _predict(self, horizon: int) -> np.ndarray:
        return np.resize(self._last_season, horizon)

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[self.period:] = self._y[:-self.period]
        return out


def walk_forward(model_factory, y: ArrayLike, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin backtest: for each origin i >= start, fit on y[:i], predict y[i].

    Parameters
    ----------
    model_factory : callable returning a fresh, unfitted :class:`Forecaster`.
    y : the full series.
    start : index of the first point to forecast (0-based).

    Returns (actuals, one-step forecasts) as arrays.
    """
    y_arr = np.asarray(y, dtype=float)
    if not 1 <= start < y_arr.size:
        raise ValueError("start must lie inside the series")
    preds = np.array([model_factory().fit(y_arr[:i]).predict(1)[0] for i in range(start, y_arr.size)])
    return y_arr[start:], preds
