from __future__ import annotations

import pandas as pd

from quant_risk_monitor.rules.concentration import check_single_name_weight
from quant_risk_monitor.rules.drawdown import check_daily_loss, check_drawdown


def test_drawdown_alert():
    nav = pd.Series([100, 105, 103, 95, 88], index=pd.date_range("2025-01-01", periods=5, freq="B"))
    alerts = check_drawdown(nav, max_drawdown=0.10)
    assert len(alerts) == 1
    assert alerts[0].rule_id == "max_drawdown"


def test_daily_loss_alert():
    nav = pd.Series([100, 96], index=pd.date_range("2025-01-01", periods=2, freq="B"))
    alerts = check_daily_loss(nav, daily_loss_limit=0.03)
    assert len(alerts) == 1
    assert alerts[0].rule_id == "daily_loss"


def test_concentration_alert():
    weights = pd.Series([0.2, 0.25, 0.35, 0.2], index=["A", "B", "C", "D"])
    alerts = check_single_name_weight(weights, max_weight=0.30)
    assert len(alerts) == 1
    assert alerts[0].details["symbol"] == "C"


def test_no_drawdown_when_within_limit():
    nav = pd.Series([100, 101, 102, 101, 103], index=pd.date_range("2025-01-01", periods=5, freq="B"))
    assert check_drawdown(nav, max_drawdown=0.10) == []
