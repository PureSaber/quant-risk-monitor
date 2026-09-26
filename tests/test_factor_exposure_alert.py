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


def test_factor_exposure_missing_inputs_raise_critical_not_evaluable_alert() -> None:
    for current in (pd.Series(dtype=float), pd.Series({"value": float("nan")})):
        alerts = check_factor_exposure_drift(current, pd.Series({"value": 1.0}))
        assert len(alerts) == 1
        assert alerts[0].rule_id == "factor_exposure_drift_not_evaluable"
        assert alerts[0].severity.value == "critical"


def test_factor_exposure_changed_factor_set_raises_critical_alert() -> None:
    alerts = check_factor_exposure_drift(
        pd.Series({"value": 0.1, "momentum": 0.2}),
        pd.Series({"value": 0.1, "size": 0.2}),
    )
    assert alerts[0].rule_id == "factor_exposure_drift_not_evaluable"
    assert "factor sets differ" in alerts[0].details["reason"]
