import json
import subprocess
import sys

import pandas as pd
import pytest

from quant_risk_monitor.rules.concentration import check_single_name_weight
from quant_risk_monitor.rules.drawdown import check_daily_loss, check_drawdown


def risk_cli(tmp_path, *, nav=None, holdings=None, source="equity"):
    config = {"rules": {"max_drawdown": 0.1, "daily_loss": 0.03, "single_name_weight": 0.5}}
    if nav is not None:
        (tmp_path / "nav.csv").write_text(nav, encoding="utf-8")
        config["nav"] = {"source": source, "path": "nav.csv", "column": "nav"}
    if holdings is not None:
        (tmp_path / "holdings.csv").write_text(holdings, encoding="utf-8")
        config["holdings"] = {"path": "holdings.csv"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "result.json"
    # A failed evaluation must replace an earlier green result, not leave it stale.
    output.write_text('{"alerts": [], "has_critical": false}', encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_risk_monitor.cli",
            "check",
            "--config",
            str(path),
            "--out",
            str(output),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    return process.returncode, json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["equity", "spread"])
@pytest.mark.parametrize(
    "rows",
    [
        "2026-09-28,100\n2026-09-29,broken\n",
        "2026-09-28,broken\n2026-09-29,broken\n",
        "2026-09-28,100\n2026-09-29,\n",
        "2026-09-28,100\n2026-09-29,inf\n",
        "2026-09-28,100\n2026-09-29,-1\n",
        "2026-09-28,100\n2026-09-28,50\n",
        ",100\n2026-09-29,50\n",
        "2026-09-28,100\n",
        "",
    ],
)
def test_bad_nav_is_not_a_green_risk_result(tmp_path, source, rows):
    code, result = risk_cli(tmp_path, source=source, nav="date,nav\n" + rows)
    assert code == 2
    assert result["has_critical"]
    assert result["metrics"]["evaluation_status"] == "unavailable"
    assert result["alerts"][0]["rule_id"] == "input_data_invalid"


@pytest.mark.parametrize(
    "rows", ["A,broken\n", "A,\n", "A,inf\n", "A,-0.2\n", ",0.2\n", " ,0.2\n", ""]
)
def test_bad_weights_are_not_zero_exposure(tmp_path, rows):
    code, result = risk_cli(tmp_path, holdings="symbol,weight\n" + rows)
    assert code == 2 and result["has_critical"]


def test_cli_aggregates_duplicate_symbols_and_preserves_leading_zeroes(tmp_path):
    code, split = risk_cli(tmp_path, holdings="symbol,weight\n001,0.4\n001,0.4\nB,0.2\n")
    assert code == 1
    code, aggregated = risk_cli(tmp_path, holdings="symbol,weight\n001,0.8\nB,0.2\n")
    assert code == 1 and split == aggregated
    assert split["alerts"][0]["details"] == {"symbol": "001", "weight": 0.8}


def test_valid_loss_still_alerts_and_full_cash_weights_are_valid(tmp_path):
    code, result = risk_cli(tmp_path, nav="date,nav\n2026-09-28,100\n2026-09-29,0\n")
    assert code == 1 and result["alerts"][0]["rule_id"] == "max_drawdown"
    code, result = risk_cli(tmp_path, holdings="symbol,weight\nA,0\nB,0\n")
    assert code == 0 and result["count"] == 0


def test_direct_rules_validate_inputs_and_aggregate():
    with pytest.raises(ValueError, match="two observations"):
        check_drawdown(pd.Series(dtype=float), 0.1)
    with pytest.raises(ValueError, match="finite"):
        check_daily_loss(
            pd.Series([100, float("nan")], index=pd.date_range("2026-01-01", periods=2)), 0.1
        )
    with pytest.raises(ValueError, match="finite"):
        check_single_name_weight(pd.Series([0.2], index=["A"]), float("nan"))
    alerts = check_single_name_weight(pd.Series([0.4, 0.4], index=["A", " A "]), 0.5)
    assert alerts[0].details["weight"] == 0.8
