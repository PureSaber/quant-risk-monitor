import copy
import json
import math
import runpy
from pathlib import Path

import pytest

from quant_risk_monitor.cli import main
from quant_risk_monitor.model_diagnostics import diagnose_models, run_model_diagnostics

FIXTURE = runpy.run_path(str(Path(__file__).parents[1] / "examples/model_diagnostics_demo.py"))[
    "fixture"
]


def test_independent_null_change_and_uncertainty_calculation():
    null = diagnose_models(FIXTURE(changed=False))
    assert all(f["status"] == "no_clear_breach" for f in null["windows"][0]["findings"].values())
    change = diagnose_models(FIXTURE())
    assert all(f["status"] == "alert" for f in change["windows"][0]["findings"].values())
    # Independently specified Hoeffding bound: alpha/(7 domains * 2 windows * 2 intervals).
    radius = math.sqrt(math.log(2 / (0.05 / 28)) / (2 * 256))
    error = change["windows"][0]["findings"]["forecast_error"]
    assert error["current"]["mean"] == 1
    assert error["current"]["interval"] == pytest.approx([1 - radius, 1])
    decay = change["windows"][0]["findings"]["signal_decay"]
    assert decay["interval"] == pytest.approx([1 - 2 * radius, 1])
    assert change["strategy_invalid"] is None and not change["execution_authorized"]


def test_forecast_error_is_not_strategy_failure():
    payload = FIXTURE(changed=False)
    for row in payload["rows"][256:]:
        row["target"] = -0.1
    report = diagnose_models(payload)["windows"][0]["findings"]
    assert report["forecast_error"]["status"] == "alert"
    assert report["signal_decay"]["status"] == "alert"
    assert report["strategy_performance"]["status"] == "no_clear_breach"
    assert report["operational_failures"]["status"] == "no_clear_breach"


def test_low_sample_partial_window_and_missing_data():
    payload = FIXTURE(count=10)
    assert all(
        f["status"] == "insufficient_sample"
        for f in diagnose_models(payload)["windows"][0]["findings"].values()
    )
    payload = FIXTURE()
    payload["rows"] = payload["rows"][:-1]
    assert all(
        f["status"] == "insufficient_sample"
        for f in diagnose_models(payload)["windows"][0]["findings"].values()
    )
    payload = FIXTURE()
    for row in payload["rows"][256:]:
        row.pop("target")
    findings = diagnose_models(payload)["windows"][0]["findings"]
    assert findings["forecast_error"]["status"] == "partial_observation"
    assert findings["risk_miscalibration"]["status"] == "alert"


def test_selectively_removing_bad_outcomes_cannot_clear_alert():
    payload = FIXTURE()
    assert diagnose_models(payload)["windows"][0]["findings"]["forecast_error"]["status"] == "alert"
    for row in payload["rows"][256:450]:
        row.pop("target")
    finding = diagnose_models(payload)["windows"][0]["findings"]["forecast_error"]
    assert finding["status"] == "partial_observation"
    assert finding["current"]["complete_blocks"] == 62
    assert "interval" not in finding["current"]


def test_structural_change_and_dependence_disable_inference():
    payload = FIXTURE()
    payload["rows"][-1]["regime_id"] = "new-contract"
    assert all(
        f["status"] == "structural_change"
        for f in diagnose_models(payload)["windows"][0]["findings"].values()
    )
    payload = FIXTURE()
    payload["plan"]["independent_blocks_assumed"] = False
    assert all(
        f["status"] == "descriptive_only"
        for f in diagnose_models(payload)["windows"][0]["findings"].values()
    )


def test_unobserved_future_values_do_not_change_current_results():
    payload = FIXTURE()
    payload["evaluation_as_of"] = payload["rows"][300]["outcome_known_at"]
    first = diagnose_models(payload)
    for row in payload["rows"][301:]:
        row.update(target=999, net_return=-999, actual_cost_bps=999)
    later = diagnose_models(payload)
    assert first["windows"] == later["windows"]
    assert len(first["withheld_outcomes"]) == 211


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", 0.2),
        ("prediction", float("nan")),
        ("feature", 2),
        ("risk_limit", 0),
        ("actual_cost_bps", -1),
        ("actual_cost_bps", 101),
        ("net_return", -0.3),
        ("operation", "unknown"),
        ("source_sha256", "bad"),
    ],
)
def test_invalid_domain_inputs(field, value):
    payload = FIXTURE()
    payload["rows"][-1][field] = value
    with pytest.raises(ValueError):
        diagnose_models(payload)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "overlap", "late_forecast", "late_plan", "gap", "horizon", "unknown"]
)
def test_timing_and_closed_contract(mutation):
    payload = FIXTURE()
    if mutation == "duplicate":
        payload["rows"][-1]["row_id"] = payload["rows"][-2]["row_id"]
    elif mutation == "overlap":
        payload["rows"][-1]["period_start"] = payload["rows"][-2]["period_start"]
    elif mutation == "late_forecast":
        payload["rows"][0]["forecast_at"] = payload["rows"][1]["forecast_at"]
    elif mutation == "late_plan":
        payload["plan"]["frozen_at"] = payload["rows"][1]["forecast_at"]
    elif mutation == "gap":
        payload["rows"][0]["outcome_known_at"] = "2099-01-01T00:00:00Z"
    elif mutation == "horizon":
        payload["plan"]["window_rows"] = 50
    else:
        payload["unknown"] = True
    with pytest.raises(ValueError):
        diagnose_models(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"block_rows": 3},
        {"min_blocks": 1},
        {"reference_rows": True},
        {"family_alpha": 0},
        {"forecast_bound": -1},
        {"thresholds": {}},
        {"feature_bounds": [1, 0]},
        {"currency": ""},
        {"independent_blocks_assumed": 1},
        {"risk_exceedance_probability": 1},
    ],
)
def test_invalid_plan(change):
    payload = FIXTURE()
    payload["plan"].update(change)
    with pytest.raises(ValueError):
        diagnose_models(payload)


def test_block_sample_count_and_no_window():
    payload = FIXTURE()
    payload["plan"]["block_rows"] = 4
    result = diagnose_models(payload)
    assert result["windows"][0]["findings"]["signal_decay"]["current"]["complete_blocks"] == 64
    payload["rows"] = payload["rows"][:200]
    assert diagnose_models(payload)["status"] == "insufficient_sample"


def test_real_cli_exclusive_outputs_and_errors(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(FIXTURE()))
    main(["diagnose-model", "--input", str(path), "--out", str(tmp_path / "out")])
    saved = json.loads((tmp_path / "out/result.json").read_text())
    assert len(saved["windows"]) == 1
    with pytest.raises(SystemExit) as exc:
        main(["diagnose-model", "--input", str(path), "--out", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "unavailable" in capsys.readouterr().out
    changed = copy.deepcopy(FIXTURE())
    changed["rows"][0]["forecast_at"] = "2020-01-01"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        run_model_diagnostics(path, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
