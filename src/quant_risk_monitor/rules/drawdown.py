from __future__ import annotations

import pandas as pd

from quant_risk_monitor.models import Alert, CheckResult, Severity


def check_drawdown(nav: pd.Series, max_drawdown: float) -> list[Alert]:
    if nav.empty:
        return []
    peak = nav.cummax()
    dd = (nav - peak) / peak
    worst = float(dd.min())
    if worst < -max_drawdown:
        return [
            Alert(
                rule_id="max_drawdown",
                severity=Severity.CRITICAL,
                message=f"drawdown {worst:.2%} exceeds limit {-max_drawdown:.2%}",
                details={"worst_drawdown": round(worst, 6), "as_of": nav.index[-1].strftime("%Y-%m-%d")},
            )
        ]
    return []


def check_daily_loss(nav: pd.Series, daily_loss_limit: float) -> list[Alert]:
    if len(nav) < 2:
        return []
    daily_ret = nav.pct_change().iloc[-1]
    if float(daily_ret) < -daily_loss_limit:
        return [
            Alert(
                rule_id="daily_loss",
                severity=Severity.WARNING,
                message=f"daily return {daily_ret:.2%} below limit {-daily_loss_limit:.2%}",
                details={"daily_return": round(float(daily_ret), 6)},
            )
        ]
    return []


def run_nav_rules(nav: pd.Series, rules: dict) -> CheckResult:
    alerts: list[Alert] = []
    if rules.get("max_drawdown") is not None:
        alerts.extend(check_drawdown(nav, float(rules["max_drawdown"])))
    if rules.get("daily_loss") is not None:
        alerts.extend(check_daily_loss(nav, float(rules["daily_loss"])))
    return CheckResult(alerts=alerts)
