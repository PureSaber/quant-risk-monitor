"""Public CLI must replace stale green reports when liquidity cannot be evaluated."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from quant_risk_monitor.analytics import liquidity_days_to_exit
from quant_risk_monitor.cli import main
from quant_risk_monitor.models import CheckResult


def run_cli(tmp_path, *, market_value="100000", adv="10000", limit=2, mutation=None):
    positions = pd.DataFrame(
        {"symbol": ["000001"], "weight": [0.5], "market_value": [market_value]}
    )
    liquidity = pd.DataFrame({"symbol": ["000001"], "average_daily_value": [adv]})
    config = {
        "advanced": {
            "positions": {"path": "positions.csv"},
            "liquidity": {"path": "liquidity.csv"},
        },
        "rules": {"max_days_to_exit": limit},
    }
    if mutation == "missing_value_column":
        positions = positions.drop(columns="market_value")
    elif mutation == "missing_adv_column":
        liquidity = liquidity.drop(columns="average_daily_value")
    elif mutation == "missing_symbol_column":
        liquidity = liquidity.drop(columns="symbol")
    elif mutation == "empty_positions":
        positions = positions.iloc[:0]
    elif mutation == "missing_positions":
        del config["advanced"]["positions"]
    elif mutation == "duplicate_liquidity":
        liquidity = pd.concat([liquidity, liquidity])
    elif mutation == "missing_coverage":
        liquidity["symbol"] = "000002"
    elif mutation == "blank_symbol":
        liquidity["symbol"] = ""
    elif mutation == "bad_participation":
        config["advanced"]["liquidity"]["max_participation"] = "nan"
    elif mutation == "null_participation":
        config["advanced"]["liquidity"]["max_participation"] = None
    elif mutation == "duplicate_positions":
        positions = pd.concat([positions, positions])
    positions.to_csv(tmp_path / "positions.csv", index=False)
    liquidity.to_csv(tmp_path / "liquidity.csv", index=False)
    path, out = tmp_path / "config.json", tmp_path / "alerts.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out.write_text('{"has_critical":false,"alerts":[]}', encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_risk_monitor.cli",
            "check",
            "--config",
            str(path),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    raw = out.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    return proc, json.loads(raw)


@pytest.mark.parametrize("field", ["market_value", "adv", "limit"])
@pytest.mark.parametrize("value", ["", "nan", "inf", "-inf", "corrupt"])
def test_nonfinite_liquidity_replaces_green_report(tmp_path, field, value):
    proc, payload = run_cli(tmp_path, **{field: value})
    assert proc.returncode == 2, proc.stderr
    assert payload["has_critical"]
    assert payload["metrics"]["evaluation_status"] == "unavailable"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_value_column",
        "missing_adv_column",
        "missing_symbol_column",
        "empty_positions",
        "missing_positions",
        "duplicate_liquidity",
        "missing_coverage",
        "blank_symbol",
        "bad_participation",
        "null_participation",
        "duplicate_positions",
    ],
)
def test_missing_or_ambiguous_liquidity_is_unavailable(tmp_path, mutation):
    proc, payload = run_cli(tmp_path, mutation=mutation)
    assert proc.returncode == 2, proc.stderr
    assert payload["metrics"]["evaluation_status"] == "unavailable"


@pytest.mark.parametrize("kwargs", [{"adv": "0"}, {"adv": "-1"}, {"limit": -1}])
def test_invalid_liquidity_domain_is_unavailable(tmp_path, kwargs):
    proc, payload = run_cli(tmp_path, **kwargs)
    assert proc.returncode == 2
    assert payload["has_critical"]


@pytest.mark.parametrize("limit,exit_code", [(2, 1), (100, 0)])
def test_valid_liquidity_has_expected_exit_and_preserves_symbol(tmp_path, limit, exit_code):
    proc, payload = run_cli(tmp_path, limit=limit)
    assert proc.returncode == exit_code, proc.stderr
    assert payload["metrics"]["liquidity"][0]["symbol"] == "000001"
    assert payload["metrics"]["liquidity"][0]["days_to_exit"] == 100


def test_derived_overflow_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        liquidity_days_to_exit(pd.Series({"A": 1e308}), pd.Series({"A": 1e-300}))


def test_nullable_missing_market_value_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        liquidity_days_to_exit(
            pd.Series([pd.NA], index=["A"], dtype="Float64"), pd.Series({"A": 10000})
        )


def test_nonfinite_output_guard_replaces_old_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "quant_risk_monitor.cli.run_check", lambda _: CheckResult(metrics={"bad": float("nan")})
    )
    out = tmp_path / "alerts.json"
    with pytest.raises(SystemExit) as exc:
        main(["check", "--config", "unused", "--out", str(out)])
    assert exc.value.code == 2
    assert json.loads(out.read_text())["metrics"]["evaluation_status"] == "unavailable"
