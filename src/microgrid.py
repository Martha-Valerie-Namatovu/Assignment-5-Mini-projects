"""Mini-Project 2: solar micro-grid dispatch planner for a rural health centre in Kasese.

Unknowns: x = solar energy (kWh), y = battery energy (kWh) [and z = diesel in the hybrid].
Constraints:  3x + 2y = D1 (daytime load),  4x + y = D2 (critical-equipment load).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from scipy.optimize import nnls


def parse_demand(text: str) -> float:
    """Convert one user-typed demand to a float, rejecting empty, non-numeric and negative input."""
    if text is None or not str(text).strip():
        raise ValueError("value is empty")
    try:
        value = float(str(text).strip())
    except ValueError:
        raise ValueError(f"{text!r} is not a number") from None
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    if value < 0:
        raise ValueError("demand cannot be negative")
    return value


def prompt_demand(prompt: str, input_fn: Callable[[str], str] = input,
                  max_attempts: int = 5, print_fn: Callable[[str], None] = print) -> float:
    """Ask until a valid demand is entered. `input_fn` is injectable so this can be tested."""
    for _ in range(max_attempts):
        raw = input_fn(prompt)
        try:
            return parse_demand(raw)
        except ValueError as err:
            print_fn(f"  Invalid input: {err}. Please try again.")
    raise ValueError(f"no valid input after {max_attempts} attempts")


def generate_demand_csv(path: str | Path, days: int = 30, seed: int = 2026) -> np.ndarray:
    """Write `days` of synthetic (D1, D2) demand to CSV and return a (days, 2) array.

    Model (illustrative): the clinic draws more on weekdays (outpatients) than at
    weekends. D1 ≈ 120 kWh weekday / 95 kWh weekend; critical load D2 ≈ 1.05·D1
    (fridges for vaccines, oxygen concentrator, lab) with independent noise, so a
    few days fall outside the physically feasible ratio band and must be handled.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    day_idx = np.arange(days)
    weekday = (day_idx % 7) < 5
    d1 = np.where(weekday, 120.0, 95.0) + rng.normal(0, 8, days)
    d2 = 1.05 * d1 + rng.normal(0, 18, days)
    demand = np.round(np.column_stack([np.clip(d1, 1, None), np.clip(d2, 1, None)]), 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "D1_kwh", "D2_kwh"])
        for i, (a, b) in enumerate(demand, start=1):
            writer.writerow([i, a, b])
    return demand


def load_demand_csv(path: str | Path) -> np.ndarray:
    """Read a demand CSV (columns day, D1_kwh, D2_kwh) into a (days, 2) array, validating values."""
    rows = []
    with Path(path).open() as fh:
        for rec in csv.DictReader(fh):
            rows.append([parse_demand(rec["D1_kwh"]), parse_demand(rec["D2_kwh"])])
    if not rows:
        raise ValueError("CSV contains no demand rows")
    return np.array(rows)


@dataclass
class DispatchResult:
    """Solution for a batch of days."""

    raw: np.ndarray            # unconstrained solution, shape (n_sources, days)
    dispatch: np.ndarray       # after the infeasibility strategy is applied
    infeasible: np.ndarray     # boolean mask of days with any negative source
    residual: np.ndarray       # ||A·dispatch − b|| per day (0 for feasible days)
    notes: list[str] = field(default_factory=list)


class MicroGrid:
    """Two-source (solar + battery) micro-grid governed by a square linear system A·s = d."""

    SOURCES: tuple[str, ...] = ("solar", "battery")
    DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]])

    def __init__(self, coefficients: ArrayLike | None = None,
                 tariffs_ugx: ArrayLike | None = None) -> None:
        a = self.DEFAULT_A if coefficients is None else np.asarray(coefficients, dtype=float)
        n = len(self.SOURCES)
        if a.shape != (n, n):
            raise ValueError(f"{type(self).__name__} needs a {n}×{n} coefficient matrix, got {a.shape}")
        self.A = a
        self.tariffs = np.asarray(tariffs_ugx if tariffs_ugx is not None else [150.0, 450.0][:n], dtype=float)
        if self.tariffs.shape != (n,):
            raise ValueError("one tariff per source is required")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self.SOURCES}, det={self.determinant:.3g})"

    # ---- well-posedness ---------------------------------------------------
    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.A))

    def is_well_posed(self, tol: float = 1e-10, max_cond: float = 1e8) -> bool:
        """True if A is non-singular and not so ill-conditioned that answers are untrustworthy."""
        return abs(self.determinant) > tol and self.condition_number < max_cond

    # ---- solving -----------------------------------------------------------
    def solve_day(self, *demands: float) -> np.ndarray:
        """Exact solution for one day (may contain negative, i.e. infeasible, values)."""
        if len(demands) != len(self.SOURCES):
            raise ValueError(f"expected {len(self.SOURCES)} demand values")
        for d in demands:
            parse_demand(str(d))
        if not self.is_well_posed():
            raise np.linalg.LinAlgError("system is singular or ill-conditioned; cannot solve uniquely")
        return scipy.linalg.solve(self.A, np.asarray(demands, dtype=float))

    def solve_loop(self, demand: np.ndarray) -> np.ndarray:
        """Solve day by day in a Python loop. `demand` has shape (days, n). Returns (n, days)."""
        return np.column_stack([scipy.linalg.solve(self.A, row) for row in np.asarray(demand)])

    def solve_vectorised(self, demand: np.ndarray) -> np.ndarray:
        """Solve all days in one call by passing an (n × days) right-hand side."""
        return scipy.linalg.solve(self.A, np.asarray(demand, dtype=float).T)

    def dispatch(self, demand: np.ndarray, strategy: str = "nnls") -> DispatchResult:
        """Solve every day and repair physically infeasible ones.

        strategy:
          * ``"nnls"`` – non-negative least squares: the closest non-negative
            dispatch to meeting both loads (some load goes unmet, which is reported).
          * ``"clip"`` – set negatives to zero (simple but ignores the constraints).
        """
        if strategy not in {"nnls", "clip"}:
            raise ValueError("strategy must be 'nnls' or 'clip'")
        demand = np.asarray(demand, dtype=float)
        if demand.ndim != 2 or demand.shape[1] != len(self.SOURCES) or demand.shape[0] == 0:
            raise ValueError(f"demand must have shape (days, {len(self.SOURCES)})")
        raw = self.solve_vectorised(demand)
        infeasible = np.any(raw < -1e-9, axis=0)
        fixed = raw.copy()
        notes: list[str] = []
        for j in np.flatnonzero(infeasible):
            if strategy == "nnls":
                fixed[:, j], _ = nnls(self.A, demand[j])
            else:
                fixed[:, j] = np.clip(raw[:, j], 0, None)
            short = demand[j] - self.A @ fixed[:, j]
            notes.append(f"day {j + 1}: raw={np.round(raw[:, j], 1)}, fixed={np.round(fixed[:, j], 1)}, "
                         f"load mismatch={np.round(short, 1)} kWh")
        residual = np.linalg.norm(self.A @ fixed - demand.T, axis=0)
        return DispatchResult(raw, fixed, infeasible, residual, notes)

    def daily_cost(self, dispatch: np.ndarray) -> np.ndarray:
        """Cost per day in UGX for an (n, days) dispatch array."""
        return self.tariffs @ np.asarray(dispatch, dtype=float)


class HybridMicroGrid(MicroGrid):
    """Adds a diesel generator z and a third constraint (3×3 system).

    Default third constraint (illustrative): total evening energy x + y + z = D3,
    with diesel also entering the first two loads:
        3x + 2y +  z = D1
        4x +  y + 2z = D2
         x +  y +  z = D3
    """

    SOURCES = ("solar", "battery", "diesel")
    DEFAULT_A = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0]])

    def __init__(self, coefficients: ArrayLike | None = None,
                 tariffs_ugx: ArrayLike | None = None) -> None:
        super().__init__(coefficients, tariffs_ugx if tariffs_ugx is not None else [150.0, 450.0, 1200.0])

    def solve_day(self, *demands: float) -> np.ndarray:
        """As the parent, but for a singular (linearly dependent) system fall back to
        the minimum-norm least-squares solution and say so, instead of crashing."""
        if self.is_well_posed():
            return super().solve_day(*demands)
        sol, *_ = scipy.linalg.lstsq(self.A, np.asarray(demands, dtype=float))
        return sol

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))


def sensitivity_monte_carlo(grid: MicroGrid, base_demand: ArrayLike, rel_perturb: float = 0.05,
                            n_draws: int = 1000, seed: int = 7) -> dict[str, np.ndarray | float]:
    """Perturb every demand uniformly within ±rel_perturb and record how the solution moves.

    Returns the draws, the base solution, the worst observed relative amplification
    (‖Δs‖/‖s‖)/(‖Δd‖/‖d‖) and the condition number (its theoretical upper bound).
    """
    base_d = np.asarray(base_demand, dtype=float)
    rng = np.random.default_rng(seed)
    perturbed = base_d * (1 + rng.uniform(-rel_perturb, rel_perturb, size=(n_draws, base_d.size)))
    base_s = scipy.linalg.solve(grid.A, base_d)
    sols = scipy.linalg.solve(grid.A, perturbed.T).T
    rel_in = np.linalg.norm(perturbed - base_d, axis=1) / np.linalg.norm(base_d)
    rel_out = np.linalg.norm(sols - base_s, axis=1) / np.linalg.norm(base_s)
    amp = rel_out / np.where(rel_in == 0, np.nan, rel_in)
    return {"demands": perturbed, "solutions": sols, "base_solution": base_s,
            "max_amplification": float(np.nanmax(amp)), "condition_number": grid.condition_number}
