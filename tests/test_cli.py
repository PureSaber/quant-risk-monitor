from __future__ import annotations

from pathlib import Path

from quant_risk_monitor.cli import run_check


def test_run_check_default_config():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    result = run_check(cfg)
    assert not result.has_critical


def test_run_check_alert_demo():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "alert_demo.yaml"
    result = run_check(cfg)
    assert result.has_critical
