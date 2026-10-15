import cvxpy as cp
import numpy as np
import pandas as pd
from repair_feasibility_audit import absolute_linear_minimum, linear_minimum, solve_bound


def test_analytical_support_minima_equal_independent_convex_programs():
    rng = np.random.default_rng(41)
    for _ in range(12):
        gradient = rng.normal(size=10)
        current = rng.uniform(0, 0.2, 10)
        w = cp.Variable(10)
        constraints = [w >= 0, w <= 0.15, cp.sum(w) == 0.9]
        for absolute in (True, False):
            objective = gradient @ w + (cp.norm1(w - current) if absolute else 0)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10)
            actual = (
                absolute_linear_minimum(gradient, current, 0.9, 0.15)
                if absolute
                else linear_minimum(gradient, 0.9, 0.15)
            )
            assert abs(actual - problem.value) < 1e-7


def test_risk_floor_has_an_independent_supporting_plane_lower_bound():
    data = {
        "exposures": pd.DataFrame({"market": np.ones(25)}),
        "F_daily_rate": np.array([[0.0004]]),
        "D_daily_rate": np.ones(25) * 0.0001,
    }
    benchmark = pd.Series([0.5] + [0.5 / 24] * 24)
    result = solve_bound(data, benchmark, 0.8)
    assert 0 <= result["annual_te"] - result["certified_lower_te"] < 1e-7
    assert result["certified_lower_te"] > 0.03
    current = np.random.default_rng(12).uniform(0.001, 0.05, 25)
    current *= 0.8 / current.sum()
    turnover = solve_bound(data, benchmark, 0.8, current=current, limit=result["annual_te"] * 1.001)
    assert -1e-8 <= turnover["turnover"] - turnover["certified_lower_turnover"] < 1e-6
