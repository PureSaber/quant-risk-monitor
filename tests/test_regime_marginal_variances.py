"""The marginal shortcut preserves the full-covariance regime calculation."""

import numpy as np
import pytest

from quant_risk_monitor.factor_model import FactorModelConfig
from quant_risk_monitor.risk_estimators import (
    estimate_covariance,
    regime_multiplier,
    time_weights,
)


@pytest.mark.parametrize("half_life", [None, 15.0, 60.0])
def test_regime_matches_full_covariance_reference(half_life):
    values = np.random.default_rng(712).normal(size=(90, 17))
    values[:, -1] = 0
    config = FactorModelConfig(
        model_kind="fundamental_style",
        regime_window=20,
        regime_min_periods=60,
        regime_half_life=10,
    )
    expected = []
    for position in range(70, 90):
        prior = values[:position]
        covariance, _ = estimate_covariance(prior, half_life=half_life)
        variance = np.diag(covariance)
        innovation = values[position] - time_weights(len(prior), half_life) @ prior
        positive = variance > 0
        expected.append(float(np.mean(innovation[positive] ** 2 / variance[positive])))
    multiplier, diagnostic = regime_multiplier(values, config, half_life=half_life)
    np.testing.assert_allclose(diagnostic["innovation_scores"], expected, rtol=1e-12)
    raw = time_weights(20, 10) @ expected
    assert multiplier == pytest.approx(np.clip(raw, 0.25, 4.0), rel=1e-12)


def test_zero_variance_then_nonzero_innovation_still_rejected():
    values = np.zeros((90, 2))
    values[-1, 0] = 1
    config = FactorModelConfig(
        model_kind="fundamental_style", regime_window=20, regime_min_periods=60
    )
    with pytest.raises(ValueError, match="zero-variance history"):
        regime_multiplier(values, config, half_life=None)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_history_is_rejected(value):
    config = FactorModelConfig(
        model_kind="fundamental_style", regime_window=20, regime_min_periods=60
    )
    with pytest.raises(ValueError, match="finite time-by-series"):
        regime_multiplier(np.full((90, 2), value), config, half_life=None)
