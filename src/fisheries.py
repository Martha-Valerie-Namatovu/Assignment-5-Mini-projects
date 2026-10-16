"""Mini-Project 3: Lake Victoria fish stock & export risk model (Jinja cooperative).

Time step = one week. Stock in tonnes, price in UGX/kg, revenue in UGX.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .forecasting import fibonacci  # re-exported for the baseline task

__all__ = ["fibonacci", "FishStock", "PriceModel", "RiskAssessor", "revenue_stats",
           "msy", "KG_PER_TONNE"]

KG_PER_TONNE = 1000.0


def msy(r: float, k: float) -> float:
    """Maximum sustainable yield of the logistic model, rK/4 (reached at N = K/2, h = r/2)."""
    return r * k / 4


class FishStock:
    """Discrete logistic growth with proportional harvesting.

    N(t+1) = N(t) + r·N(t)·(1 − N(t)/K) − h·N(t)

    Parameters
    ----------
    r : intrinsic growth rate per week (0 < r < 2 keeps the map stable without harvest).
    k : carrying capacity in tonnes.
    n0 : initial stock in tonnes (0 ≤ n0).
    h : fraction of the stock harvested each week (0 ≤ h < 1).
    """

    def __init__(self, r: float = 0.4, k: float = 10_000.0, n0: float = 4_000.0, h: float = 0.1) -> None:
        if r <= 0:
            raise ValueError("growth rate r must be positive")
        if k <= 0:
            raise ValueError("carrying capacity K must be positive")
        if n0 < 0:
            raise ValueError("initial stock cannot be negative")
        if not 0 <= h < 1:
            raise ValueError("harvest rate h must be in [0, 1)")
        self.r, self.k, self.n0, self.h = r, k, n0, h

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.k:,.0f} t, N0={self.n0:,.0f} t, h={self.h})"

    @property
    def equilibrium(self) -> float:
        """Non-trivial steady state N* = K(1 − h/r); 0 if harvesting outpaces growth."""
        return max(0.0, self.k * (1 - self.h / self.r))

    @property
    def sustainable_yield(self) -> float:
        """Long-run weekly catch h·N* at this harvest rate."""
        return self.h * self.equilibrium

    def simulate(self, weeks: int = 52, closed_weeks: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run the model.

        Returns
        -------
        stock : array of length weeks+1 (stock at the start of each week, plus final).
        harvest : array of length weeks (tonnes caught in each week).

        `closed_weeks` lists week indices (0-based, modulo 52) with no harvesting.
        """
        if weeks < 0:
            raise ValueError("weeks must be non-negative")
        closed = set() if closed_weeks is None else {int(w) % 52 for w in closed_weeks}
        stock = np.empty(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            h_t = 0.0 if (t % 52) in closed else self.h
            harvest[t] = h_t * n
            stock[t + 1] = max(0.0, n + self.r * n * (1 - n / self.k) - harvest[t])
        return stock, harvest


class PriceModel:
    """Bounded Gaussian random walk for the ex-landing price (UGX/kg).

    p(t+1) = clip(p(t) + ε, low, high),  ε ~ N(drift, step_sd).
    """

    def __init__(self, start: float = 12_000.0, low: float = 9_000.0, high: float = 16_000.0,
                 step_sd: float = 400.0, drift: float = 0.0, seed: int = 42) -> None:
        if not low <= start <= high:
            raise ValueError("start price must lie within [low, high]")
        if low <= 0 or step_sd < 0:
            raise ValueError("low must be positive and step_sd non-negative")
        self.start, self.low, self.high = start, low, high
        self.step_sd, self.drift, self.seed = step_sd, drift, seed

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start:,.0f}, bounds=[{self.low:,.0f}, {self.high:,.0f}], sd={self.step_sd})"

    def simulate(self, weeks: int = 52, n_paths: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return prices with shape (n_paths, weeks); a single path if n_paths == 1 is squeezed."""
        if weeks <= 0 or n_paths <= 0:
            raise ValueError("weeks and n_paths must be positive")
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        shocks = rng.normal(self.drift, self.step_sd, size=(n_paths, weeks - 1))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + shocks[:, t - 1], self.low, self.high)
        return prices[0] if n_paths == 1 else prices


def revenue_stats(revenue: ArrayLike) -> dict[str, float]:
    """Mean, median, variance, SD and CV of a revenue series using ``statistics``."""
    data = [float(v) for v in np.asarray(revenue).ravel()]
    if len(data) < 2:
        raise ValueError("need at least two revenue values")
    mean = statistics.mean(data)
    sd = statistics.stdev(data)
    return {"mean": mean, "median": statistics.median(data), "variance": statistics.variance(data),
            "stdev": sd, "cv": sd / mean if mean else float("nan")}


@dataclass
class RiskAssessor:
    """Classify revenue risk by coefficient of variation and estimate Value-at-Risk.

    Thresholds (dimensionless, so they work at any revenue scale):
      * CV < low  → "Low"      (weekly swings under ~10% of a typical week)
      * CV < high → "Moderate"
      * otherwise → "High"     (a typical week is ±25% or more off the mean,
                                hard for a cooperative with fixed weekly wage and fuel bills)
    """

    low: float = 0.10
    high: float = 0.25

    def __post_init__(self) -> None:
        if not 0 < self.low < self.high:
            raise ValueError("thresholds must satisfy 0 < low < high")

    def classify(self, cv: float) -> str:
        if cv < 0 or not np.isfinite(cv):
            raise ValueError("CV must be a finite non-negative number")
        if cv < self.low:
            return "Low"
        return "Moderate" if cv < self.high else "High"

    @staticmethod
    def value_at_risk(annual_revenues: ArrayLike, level: float = 0.05) -> dict[str, float]:
        """Historical-simulation VaR from Monte Carlo annual revenues.

        Returns the `level` quantile (the revenue you fall below only 5% of the time)
        and VaR expressed as the shortfall from the expected (mean) revenue.
        """
        arr = np.asarray(annual_revenues, dtype=float)
        if arr.size == 0:
            raise ValueError("no simulated revenues supplied")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")
        q = float(np.quantile(arr, level))
        return {"quantile": q, "expected": float(arr.mean()), "VaR": float(arr.mean() - q)}

    def monte_carlo(self, harvest_tonnes: ArrayLike, price_model: PriceModel, n_paths: int = 1000,
                    seed: int = 123) -> np.ndarray:
        """Annual revenue (UGX) for `n_paths` independent price paths and a fixed harvest plan."""
        harvest_kg = np.asarray(harvest_tonnes, dtype=float) * KG_PER_TONNE
        prices = price_model.simulate(harvest_kg.size, n_paths=n_paths, rng=np.random.default_rng(seed))
        prices = np.atleast_2d(prices)
        return prices @ harvest_kg
