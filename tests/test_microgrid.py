"""Mini-Project 2 tests: MicroGrid solving, validation and the hybrid subclass."""
import numpy as np
import pytest

from src.microgrid import (HybridMicroGrid, MicroGrid, generate_demand_csv, load_demand_csv,
                           parse_demand, prompt_demand)


def test_solve_day_matches_hand_solution():
    # x = (2·D2 − D1)/5, y = (4·D1 − 3·D2)/5
    x, y = MicroGrid().solve_day(100, 110)
    assert x == pytest.approx(24.0)
    assert y == pytest.approx(14.0)


def test_determinant_and_condition_number():
    grid = MicroGrid()
    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number == pytest.approx(3 + 2 * np.sqrt(2))
    assert grid.is_well_posed()


def test_loop_and_vectorised_agree(tmp_path):
    demand = generate_demand_csv(tmp_path / "d.csv", days=30, seed=0)
    grid = MicroGrid()
    np.testing.assert_allclose(grid.solve_loop(demand), grid.solve_vectorised(demand))
    np.testing.assert_allclose(load_demand_csv(tmp_path / "d.csv"), demand)


def test_infeasible_day_is_repaired_non_negative():
    # Feasible only if 0.5 ≤ D2/D1 ≤ 4/3. D2/D1 = 1.6 forces a negative battery value.
    result = MicroGrid().dispatch(np.array([[100.0, 160.0], [100.0, 110.0]]))
    assert list(result.infeasible) == [True, False]
    assert np.all(result.dispatch >= 0)
    assert len(result.notes) == 1


@pytest.mark.parametrize("text", ["", "   ", "abc", "-5", "nan"])
def test_parse_demand_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_demand(text)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "ten", "-1", "42.5"])
    messages = []
    value = prompt_demand("D1: ", input_fn=lambda _: next(answers), print_fn=messages.append)
    assert value == 42.5
    assert len(messages) == 3


def test_singular_matrix_rejected_by_base_class():
    grid = MicroGrid([[1, 2], [2, 4]])
    assert not grid.is_well_posed()
    with pytest.raises(np.linalg.LinAlgError):
        grid.solve_day(1, 2)


def test_hybrid_solves_and_handles_dependent_rows():
    hybrid = HybridMicroGrid()
    s = hybrid.solve_day(100, 110, 50)
    np.testing.assert_allclose(hybrid.A @ s, [100, 110, 50])
    dependent = HybridMicroGrid([[3, 2, 1], [4, 1, 2], [7, 3, 3]])   # row3 = row1 + row2
    assert dependent.rank() == 2
    assert dependent.solve_day(100, 110, 210).shape == (3,)


def test_wrong_matrix_shape_rejected():
    with pytest.raises(ValueError):
        MicroGrid(np.eye(3))
