import pandas as pd

from quant_risk_monitor.rules.factor_exposure import check_factor_exposure_drift


def test_factor_exposure_drift_alerts() -> None:
    baseline = pd.Series({"momentum_20d": 0.2, "reversal_5d": -0.1})
    current = pd.Series({"momentum_20d": 0.8, "reversal_5d": -0.1})
    alerts = check_factor_exposure_drift(current, baseline, z_threshold=1.0)
    assert alerts
    assert alerts[0].rule_id == "factor_exposure_drift"


def test_factor_exposure_no_alert_when_stable() -> None:
    s = pd.Series({"momentum_20d": 0.2, "reversal_5d": -0.1})
    assert check_factor_exposure_drift(s, s) == []


def test_factor_exposure_preserves_legacy_no_overlap_behavior() -> None:
    assert check_factor_exposure_drift(pd.Series(dtype=float), pd.Series({"value": 1.0})) == []
    assert (
        check_factor_exposure_drift(
            pd.Series({"value": float("nan")}),
            pd.Series({"value": 1.0}),
        )
        == []
    )
