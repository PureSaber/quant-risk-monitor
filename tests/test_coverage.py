import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor.coverage import RiskCoverageError, calendar_specific_variance
from quant_risk_monitor.risk_estimators import estimate_covariance


@pytest.mark.parametrize("half_life", [None, 60, 90])
@pytest.mark.parametrize("lags", [0, 4])
def test_complete_history_preserves_original_estimator(half_life, lags):
    calendar = pd.bdate_range("2022-01-01", periods=260)
    values = pd.Series(np.random.default_rng(12).normal(0, 0.02, 260), index=calendar)
    actual, info = calendar_specific_variance(
        values,
        calendar,
        calendar[-1],
        half_life=half_life,
        lags=lags,
    )
    expected, old = estimate_covariance(values.iloc[-252:], half_life=half_life, lags=lags)
    assert actual == pytest.approx(expected[0, 0], abs=1e-18)
    assert info["effective_observations"] == pytest.approx(old["effective_observations"])
    assert info["status"] == "direct_complete"


def test_missing_day_keeps_calendar_decay_pairs_and_future_cannot_change_estimate():
    days = pd.bdate_range("2020-01-01", periods=300)
    values = pd.Series(np.random.default_rng(12).normal(0, 0.02, 300), index=days)
    values.iloc[10:20] = np.nan
    actual, info = calendar_specific_variance(values, days, days[251], half_life=60, lags=4)
    future = values.copy()
    future.iloc[252:] = 1e100
    assert calendar_specific_variance(future, days, days[251], half_life=60, lags=4)[0] == actual
    compressed, _ = estimate_covariance(values.iloc[:252].dropna(), half_life=60, lags=4)
    assert abs(compressed[0, 0] - actual) > 1e-9
    assert info["valid_observations"] == 242
    assert info["valid_calendar_pairs"]["1"] == 240


def test_one_missing_session_does_not_permanently_disqualify_asset():
    days = pd.bdate_range("2020-01-01", periods=252)
    values = pd.Series(np.arange(252) / 10000, index=days)
    values.iloc[3] = np.nan
    variance, info = calendar_specific_variance(values, days, days[-1])
    assert variance > 0 and info["valid_observations"] == 251
    values.iloc[:130] = np.nan
    with pytest.raises(RiskCoverageError, match="insufficient_calendar"):
        calendar_specific_variance(values, days, days[-1])


def test_effective_observations_are_separate_from_raw_count():
    days = pd.bdate_range("2020-01-01", periods=252)
    values = pd.Series(np.arange(252), index=days)
    with pytest.raises(RiskCoverageError, match="insufficient_effective"):
        calendar_specific_variance(values, days, days[-1], half_life=1)
