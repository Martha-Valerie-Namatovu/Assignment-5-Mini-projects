"""Mini-Project 5: matatu route revenue, pricing & fleet planner."""
from __future__ import annotations

import math
import statistics

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike


class Route:
    """A taxi route with daily passenger counts and a flat fare (UGX)."""

    def __init__(self, name: str, passengers: ArrayLike, fare_ugx: float) -> None:
        arr = np.asarray(passengers, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if np.any(arr < 0):
            raise ValueError("passenger counts cannot be negative")
        if fare_ugx <= 0:
            raise ValueError("fare must be positive")
        self.name = name
        self.passengers = arr
        self.fare = float(fare_ugx)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, variance and SD of passengers (sample statistics via ``statistics``)."""
        data = self.passengers.tolist()
        out = {"mean": statistics.mean(data)}
        if len(data) >= 2:
            out["variance"] = statistics.variance(data)
            out["stdev"] = statistics.stdev(data)
        else:
            out["variance"] = out["stdev"] = float("nan")
        return out


def market_equilibrium(demand: tuple[float, float], supply: tuple[float, float]) -> tuple[float, float]:
    """Solve Qd = a_d + b_d·P and Qs = a_s + b_s·P for (P*, Q*).

    Written as the linear system in unknowns [Q, P]:
        Q − b_d·P = a_d
        Q − b_s·P = a_s
    """
    a_d, b_d = demand
    a_s, b_s = supply
    if math.isclose(b_d, b_s):
        raise ValueError("parallel demand and supply curves: no unique equilibrium")
    A = np.array([[1.0, -b_d], [1.0, -b_s]])
    rhs = np.array([a_d, a_s])
    q, p = scipy.linalg.solve(A, rhs)
    return float(p), float(q)


def vehicles_needed(passengers: float, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15) -> int:
    """Vehicles to carry `passengers` one-way riders a day, with a safety buffer, rounded up."""
    if passengers < 0:
        raise ValueError("passengers cannot be negative")
    if trips_per_day <= 0 or seats <= 0:
        raise ValueError("trips and seats must be positive")
    if buffer < 0:
        raise ValueError("buffer cannot be negative")
    if passengers == 0:
        return 0
    return math.ceil(passengers * (1 + buffer) / (trips_per_day * seats))


def simulate_weekly_demand(days: int = 60, base: float = 50.0, seed: int = 11) -> np.ndarray:
    """Synthetic daily passengers with a weekly cycle (day 0 = Monday).

    Multipliers: Mon–Thu ~1.0, Fri 1.35 (busiest), Sat 1.1, Sun 0.6 (quietest), plus noise.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    weekly = np.array([1.0, 0.95, 1.0, 1.05, 1.35, 1.1, 0.6])
    pattern = np.resize(weekly, days)
    return np.round(np.clip(base * pattern + rng.normal(0, 3, days), 0, None))
