"""Mini-Project 5 tests: Route, equilibrium and fleet sizing."""
import numpy as np
import pytest

from src.forecasting import MovingAverageForecaster, SeasonalNaiveForecaster, walk_forward
from src.metrics import mae
from src.taxi import Route, market_equilibrium, simulate_weekly_demand, vehicles_needed

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


def test_route_revenue_and_stats():
    r = Route("Kampala-Ntinda", NTINDA, 2000)
    assert r.total_revenue() == sum(NTINDA) * 2000
    assert r.daily_revenue()[0] == 70_000
    assert r.describe()["mean"] == pytest.approx(47.4)


def test_equilibrium_hand_solution():
    # 120 − 0.02P = 10 + 0.03P  →  P* = 2200, Q* = 76
    p, q = market_equilibrium((120, -0.02), (10, 0.03))
    assert p == pytest.approx(2200)
    assert q == pytest.approx(76)


def test_parallel_curves_have_no_equilibrium():
    with pytest.raises(ValueError):
        market_equilibrium((100, 0.01), (10, 0.01))


def test_vehicles_needed():
    assert vehicles_needed(112, buffer=0) == 1
    assert vehicles_needed(113, buffer=0) == 2
    assert vehicles_needed(200) == 3        # 230 / 112 = 2.05 → 3
    assert vehicles_needed(0) == 0


@pytest.mark.parametrize("kwargs", [
    {"passengers": [], "fare_ugx": 2000},
    {"passengers": [10, -1], "fare_ugx": 2000},
    {"passengers": [10], "fare_ugx": 0},
])
def test_route_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Route("Bad", **kwargs)


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(60, seed=11)
    ma = mae(*walk_forward(lambda: MovingAverageForecaster(3), y, 14))
    sn = mae(*walk_forward(lambda: SeasonalNaiveForecaster(7), y, 14))
    assert sn < ma


def test_weekly_pattern_friday_busier_than_sunday():
    y = simulate_weekly_demand(70, seed=0).reshape(-1, 7)
    assert y[:, 4].mean() > y[:, 6].mean()
