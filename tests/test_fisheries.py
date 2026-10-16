"""Mini-Project 3 tests: FishStock, PriceModel and RiskAssessor."""
import numpy as np
import pytest

from src.fisheries import FishStock, PriceModel, RiskAssessor, fibonacci, msy, revenue_stats


def test_one_logistic_step_by_hand():
    stock, harvest = FishStock(r=0.4, k=10_000, n0=4_000, h=0.1).simulate(1)
    # 4000 + 0.4·4000·0.6 − 400 = 4560
    assert stock[1] == pytest.approx(4560)
    assert harvest[0] == pytest.approx(400)


def test_stock_converges_to_equilibrium():
    fs = FishStock(h=0.2)
    stock, _ = fs.simulate(300)
    assert fs.equilibrium == pytest.approx(5000)
    assert stock[-1] == pytest.approx(5000, rel=1e-4)
    assert fs.sustainable_yield == pytest.approx(msy(0.4, 10_000))


def test_zero_stock_stays_zero_and_overharvest_collapses():
    stock, harvest = FishStock(n0=0).simulate(10)
    assert np.all(stock == 0) and np.all(harvest == 0)
    assert FishStock(h=0.5).equilibrium == 0


def test_closed_season_stops_harvest():
    _, harvest = FishStock().simulate(52, closed_weeks=range(8))
    assert np.all(harvest[:8] == 0) and np.all(harvest[8:] > 0)


def test_price_walk_respects_bounds_and_seed():
    pm = PriceModel(step_sd=2_000, seed=3)
    p = pm.simulate(200)
    assert p.min() >= 9_000 and p.max() <= 16_000
    np.testing.assert_array_equal(p, PriceModel(step_sd=2_000, seed=3).simulate(200))


def test_cv_is_scale_free():
    rev = np.array([100.0, 120.0, 80.0, 110.0])
    assert revenue_stats(rev)["cv"] == pytest.approx(revenue_stats(rev * 1e6)["cv"])


def test_risk_classes_and_var():
    ra = RiskAssessor()
    assert [ra.classify(c) for c in (0.05, 0.15, 0.40)] == ["Low", "Moderate", "High"]
    var = ra.value_at_risk(np.arange(1, 101))
    assert var["quantile"] == pytest.approx(5.95)
    assert var["VaR"] == pytest.approx(50.5 - 5.95)


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"k": -1}, {"n0": -10}, {"h": 1.0}])
def test_invalid_fishstock_parameters(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_empty_or_invalid_inputs():
    with pytest.raises(ValueError):
        RiskAssessor().value_at_risk([])
    with pytest.raises(ValueError):
        revenue_stats([1.0])
    assert list(fibonacci(0)) == []
