import numpy as np
import pandas as pd
import pytest

from study_v3_feasibility import continuous_floor


def model():
    return {"exposures": pd.DataFrame({"factor": [0., 0., 0.]}),
            "F_daily_rate": np.array([[0.]]), "D_daily_rate": np.ones(3) / 252}


def test_cash_budget_retains_active_benchmark_risk():
    result = continuous_floor(model(), pd.Series([1/3]*3), .8, cap=.5)
    assert result["continuous_te_floor"] == pytest.approx(.2 / np.sqrt(3), abs=1e-8)
    assert result["residual"] < 1e-7


def test_concentrated_benchmark_and_cap_match_analytic_projection():
    result = continuous_floor(model(), pd.Series([.8, .1, .1]), 1., cap=.5)
    assert result["continuous_te_floor"] == pytest.approx(np.sqrt(.3**2+2*.15**2), abs=1e-8)
    with pytest.raises(ValueError, match="infeasible"):
        continuous_floor(model(), pd.Series([.8, .1, .1]), 1., cap=.2)
