"""Mini-Project 1 tests: DistrictPopulation, validation and planning helpers."""
import statistics

import numpy as np
import pytest

from src.population import (DistrictPopulation, LinearTrendForecaster, additional_classrooms,
                            bootstrap_intervals, validate_models)

YEARS = list(range(2015, 2025))
GULU = [320, 330, 345, 360, 375, 390, 410, 430, 455, 480]


@pytest.fixture
def gulu():
    return DistrictPopulation("Gulu", YEARS, GULU)


def test_repr_len_and_indexing(gulu):
    assert len(gulu) == 10
    assert "Gulu" in repr(gulu)
    assert gulu[2020] == 390


def test_cagr_hand_calculation(gulu):
    # 480/320 = 1.5 over 9 years
    assert gulu.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)


def test_statistics_vs_numpy_variance(gulu):
    s = gulu.stats_statistics()
    n = gulu.stats_numpy()
    assert s["variance"] == pytest.approx(statistics.variance(GULU))
    # np.var (ddof=0) = sample variance × (n−1)/n
    assert n["variance"] == pytest.approx(s["variance"] * 9 / 10)
    assert gulu.stats_numpy(ddof=1)["variance"] == pytest.approx(s["variance"])


def test_growth_rates_first_value(gulu):
    assert gulu.growth_rates()[0] == pytest.approx(10 / 320)


def test_validation_picks_a_model_and_scores_all_three(gulu):
    result = validate_models(gulu, 2021)
    assert set(result.scores) == {"Linear trend", "Exponential (CAGR)", "Fibonacci ratio"}
    assert result.best_model != "Fibonacci ratio"


def test_bootstrap_interval_brackets_point_forecast(gulu):
    point, lo, hi = bootstrap_intervals(gulu, LinearTrendForecaster, horizon=5, n_boot=200, seed=1)
    assert np.all(lo <= point) and np.all(point <= hi)


@pytest.mark.parametrize("years, pops", [
    ([2015, 2016], [1, 2, 3]),          # unequal lengths
    ([2015, 2016], [100, -5]),          # negative value
    ([], []),                           # empty
    ([2016, 2015], [1, 2]),             # years not increasing
])
def test_invalid_inputs_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", years, pops)


def test_classrooms_hand_calculation_and_zero_growth():
    # 100k extra people × 18% = 18,000 pupils / 53 = 339.6 → 340
    assert additional_classrooms(1000, 1100) == 340
    assert additional_classrooms(500, 500) == 0
    assert additional_classrooms(500, 450) == 0   # shrinking → no new rooms
    with pytest.raises(ValueError):
        additional_classrooms(1, 2, pupils_per_classroom=0)
