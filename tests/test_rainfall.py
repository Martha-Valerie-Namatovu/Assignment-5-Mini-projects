"""Mini-Project 4 tests: Region, CropRule and the similarity measures."""
import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import CropRule, Region, cosine_similarity, euclidean, pairwise_matrix, pearson

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


def test_region_summary():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total() == 1600
    assert r.wettest_month() == "May"
    assert r.driest_month() == "Sep"


def test_cosine_matches_scipy():
    assert cosine_similarity(KAMPALA, GULU) == pytest.approx(1 - scipy_cosine_distance(KAMPALA, GULU))


def test_cosine_ignores_scale_but_euclidean_does_not():
    a = np.array(KAMPALA, dtype=float)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)
    assert euclidean(a, 3 * a) > 0


def test_pearson_matches_numpy():
    assert pearson(GULU, MBARARA) == pytest.approx(np.corrcoef(GULU, MBARARA)[0, 1])


def test_pairwise_matrix_symmetric_unit_diagonal():
    regions = [Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)]
    m = pairwise_matrix(regions, cosine_similarity)
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.diag(m), 1.0)


def test_modality_matches_known_climate_zones():
    assert Region("Kampala", KAMPALA).modality() == "bimodal"
    assert Region("Mbarara", MBARARA).modality() == "bimodal"
    assert Region("Gulu", GULU).modality() == "unimodal"


def test_crop_rule_boundaries():
    maize = CropRule("maize", 100, 200)
    assert maize.classify(99.9) == "Drought risk"
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(200) == "Good for maize"
    assert maize.classify(201) == "Waterlogging risk"


@pytest.mark.parametrize("bad", [[1] * 11, [-1] + [10] * 11])
def test_region_rejects_bad_rainfall(bad):
    with pytest.raises(ValueError):
        Region("Bad", bad)


def test_zero_vector_and_invalid_rule():
    with pytest.raises(ValueError):
        cosine_similarity([0] * 12, KAMPALA)
    with pytest.raises(ValueError):
        CropRule("x", 200, 100)


def test_parse_power_monthly_converts_rates_and_skips_annual():
    from src.rainfall import parse_power_monthly
    payload = {"parameters": {"PRECTOTCORR": {"units": "mm/day"}},
               "properties": {"parameter": {"PRECTOTCORR": {
                   "202201": 2.0, "202202": 1.0, "202203": -999.0, "202213": 5.0}}}}
    parsed = parse_power_monthly(payload)
    assert list(parsed) == [2022]
    assert parsed[2022][0] == pytest.approx(62.0)    # 2 mm/day × 31 days
    assert parsed[2022][1] == pytest.approx(28.0)    # Feb 2022 has 28 days
    assert np.isnan(parsed[2022][2])                 # fill value → NaN
    with pytest.raises(ValueError):
        parse_power_monthly({"properties": {}})
