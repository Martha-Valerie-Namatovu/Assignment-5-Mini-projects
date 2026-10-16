"""Tests for the shared forecasting base classes and metrics."""
import numpy as np
import pytest

from src.forecasting import (CAGRForecaster, ExponentialSmoothingForecaster, FibonacciRatioForecaster,
                             LinearTrendForecaster, MovingAverageForecaster, SeasonalNaiveForecaster,
                             fibonacci, walk_forward)
from src.metrics import mae, mape, rmse


def test_linear_trend_recovers_exact_line():
    t = np.arange(2015, 2022)
    y = 100 + 5 * (t - 2015)
    pred = LinearTrendForecaster().fit(y, t).predict(3)
    assert pred == pytest.approx([135, 140, 145])


def test_cagr_matches_hand_calculation():
    model = CAGRForecaster().fit([100, 110, 121])
    assert model.rate == pytest.approx(0.10)
    assert model.predict(1)[0] == pytest.approx(133.1)


def test_fibonacci_ratios_converge_to_golden_ratio():
    assert list(fibonacci(7)) == [1, 1, 2, 3, 5, 8, 13]
    ratios = FibonacciRatioForecaster().fit([1, 2]).ratios(20)
    assert ratios[:3] == pytest.approx([1.0, 2.0, 1.5])
    assert ratios[-1] == pytest.approx(FibonacciRatioForecaster.PHI, rel=1e-6)


def test_ses_alpha_one_is_naive_forecast():
    assert ExponentialSmoothingForecaster(1.0).fit([3, 7, 5]).predict(2) == pytest.approx([5, 5])


def test_seasonal_naive_repeats_last_week():
    y = np.arange(14)
    assert list(SeasonalNaiveForecaster(7).fit(y).predict(3)) == [7, 8, 9]


def test_walk_forward_moving_average():
    actual, preds = walk_forward(lambda: MovingAverageForecaster(3), [1, 2, 3, 4, 5], start=3)
    assert list(actual) == [4, 5]
    assert preds == pytest.approx([2, 3])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(1)


@pytest.mark.parametrize("bad", [0, -2])
def test_invalid_horizon_rejected(bad):
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1, 2, 3]).predict(bad)


def test_too_short_series_rejected():
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1, 2])


def test_metrics_known_values_and_edge_cases():
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])
    with pytest.raises(ValueError):
        mae([], [])
