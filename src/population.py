"""Mini-Project 1: UBOS district population forecaster.

Populations are in thousands throughout (as in the brief).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .forecasting import (CAGRForecaster, FibonacciRatioForecaster, Forecaster,
                          LinearTrendForecaster)
from .metrics import score_all


class DistrictPopulation:
    """A district's annual population series.

    Parameters
    ----------
    name : district name.
    years : calendar years, strictly increasing.
    populations : population for each year (thousands), non-negative.
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        years_arr = np.asarray(years, dtype=int)
        pop_arr = np.asarray(populations, dtype=float)
        if not name or not str(name).strip():
            raise ValueError("district name must be non-empty")
        if years_arr.ndim != 1 or pop_arr.ndim != 1:
            raise ValueError("years and populations must be one-dimensional")
        if years_arr.size != pop_arr.size:
            raise ValueError(f"length mismatch: {years_arr.size} years vs {pop_arr.size} populations")
        if years_arr.size == 0:
            raise ValueError("a district needs at least one observation")
        if np.any(pop_arr < 0):
            raise ValueError("populations cannot be negative")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = str(name)
        self.years = years_arr
        self.populations = pop_arr

    # ---- dunder methods ---------------------------------------------------
    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, years={self.years[0]}–{self.years[-1]}, "
                f"n={len(self)}, latest={self.populations[-1]:,.0f}k)")

    def __len__(self) -> int:
        return int(self.years.size)

    def __getitem__(self, year: int) -> float:
        """Population for a given calendar year, e.g. ``kampala[2020]``."""
        idx = np.flatnonzero(self.years == year)
        if idx.size == 0:
            raise KeyError(year)
        return float(self.populations[idx[0]])

    # ---- descriptive statistics --------------------------------------------
    def stats_statistics(self) -> dict[str, float]:
        """Mean, median, variance and SD using the ``statistics`` module (sample, n−1)."""
        data = self.populations.tolist()
        return {"mean": statistics.mean(data), "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": statistics.stdev(data)}

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Same statistics with NumPy. ``ddof=0`` (NumPy default) is the population variance."""
        p = self.populations
        return {"mean": float(np.mean(p)), "median": float(np.median(p)),
                "variance": float(np.var(p, ddof=ddof)), "stdev": float(np.std(p, ddof=ddof))}

    # ---- growth -----------------------------------------------------------
    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth as fractions, length n−1."""
        if len(self) < 2:
            raise ValueError("need at least two years to compute growth")
        if np.any(self.populations[:-1] == 0):
            raise ValueError("growth undefined from a zero population")
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate between the first and last year."""
        if len(self) < 2:
            raise ValueError("need at least two years to compute CAGR")
        n_years = self.years[-1] - self.years[0]
        if self.populations[0] <= 0:
            raise ValueError("CAGR undefined from a zero starting population")
        return float((self.populations[-1] / self.populations[0]) ** (1 / n_years) - 1)

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) at `last_train_year` (inclusive in train)."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("split year must leave data on both sides")
        return (DistrictPopulation(self.name, self.years[mask], self.populations[mask]),
                DistrictPopulation(self.name, self.years[~mask], self.populations[~mask]))


#: The three model families required by the brief, as zero-argument factories.
MODEL_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "Linear trend": LinearTrendForecaster,
    "Exponential (CAGR)": CAGRForecaster,
    "Fibonacci ratio": FibonacciRatioForecaster,
}


@dataclass
class ValidationResult:
    """Hold-out scores for one district."""

    district: str
    scores: dict[str, dict[str, float]]          # model -> {MAE, RMSE, MAPE}
    test_predictions: dict[str, np.ndarray]

    @property
    def best_model(self) -> str:
        """Model with the lowest RMSE on the hold-out years (ties broken by MAE)."""
        return min(self.scores, key=lambda m: (self.scores[m]["RMSE"], self.scores[m]["MAE"]))


def validate_models(district: DistrictPopulation, last_train_year: int,
                    factories: dict[str, Callable[[], Forecaster]] = MODEL_FACTORIES) -> ValidationResult:
    """Fit every model on the training years and score it on the test years."""
    train, test = district.split(last_train_year)
    scores, preds = {}, {}
    for name, factory in factories.items():
        model = factory().fit(train.populations, train.years)
        p = model.predict(len(test))
        preds[name] = p
        scores[name] = score_all(test.populations, p)
    return ValidationResult(district.name, scores, preds)


def bootstrap_intervals(district: DistrictPopulation, factory: Callable[[], Forecaster],
                        horizon: int, n_boot: int = 1000, level: float = 0.95,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction intervals.

    For each resample: rebuild a synthetic history = fitted + resampled residuals,
    refit the model, forecast, then add resampled residuals to the forecast path
    (so both parameter and observation uncertainty are captured).

    Returns (point_forecast, lower, upper).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    base = factory().fit(district.populations, district.years)
    point = base.predict(horizon)
    fitted = base.fitted_values
    resid = base.residuals
    ok = np.isfinite(resid)
    resid_pool = resid[ok] - resid[ok].mean()
    paths = np.empty((n_boot, horizon))
    for b in range(n_boot):
        synth = fitted.copy()
        synth[~ok] = district.populations[~ok]
        synth[ok] = fitted[ok] + rng.choice(resid_pool, size=ok.sum(), replace=True)
        model = factory().fit(np.clip(synth, 1e-9, None), district.years)
        paths[b] = model.predict(horizon) + rng.choice(resid_pool, size=horizon, replace=True)
    alpha = (1 - level) / 2
    lower, upper = np.quantile(paths, [alpha, 1 - alpha], axis=0)
    return point, lower, upper


def additional_classrooms(pop_now_thousands: float, pop_future_thousands: float,
                          school_age_share: float = 0.18, pupils_per_classroom: int = 53) -> int:
    """Extra classrooms needed for the growth in school-age children (rounded up).

    Only *additional* demand is counted; existing classroom shortfalls are not
    known from the data. A shrinking population returns 0 rather than a negative.
    """
    if not 0 <= school_age_share <= 1:
        raise ValueError("school_age_share must be between 0 and 1")
    if pupils_per_classroom <= 0:
        raise ValueError("pupils_per_classroom must be positive")
    extra_pupils = (pop_future_thousands - pop_now_thousands) * 1000 * school_age_share
    return max(0, math.ceil(extra_pupils / pupils_per_classroom))
