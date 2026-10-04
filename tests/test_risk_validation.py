import json
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor import (
    AssetReturnObservation,
    ExposureSnapshot,
    FactorModelConfig,
    RiskForecastRequest,
    evaluate_risk_forecasts,
    forecast_factor_risk,
)
from quant_risk_monitor.cli import main
from quant_risk_monitor.risk_validation import (
    _digest,
    _score,
    _summary,
    evaluate_risk_spec,
    run_risk_validation,
    verify_risk_validation,
)


def spec():
    starts = pd.date_range("2025-01-01", periods=9, tz="UTC")
    returns = []
    for n, value in enumerate([0.01, -0.01, 0.02, -0.02, 0.03, -0.015, 0.005, -0.025]):
        returns.append(
            {
                "period_start": starts[n].isoformat(),
                "period_end": starts[n + 1].isoformat(),
                "available_at": starts[n + 1].isoformat(),
                "values": {"A": value, "B": value * 0.8, "C": -value * 0.3},
                "source": "synthetic fixed returns",
            }
        )
    requests = [
        {
            "period_start": returns[n]["period_start"],
            "period_end": returns[n]["period_end"],
            "weights_available_at": starts[0].isoformat(),
            "weights": {"A": 0.5, "B": 0.3},
            "benchmark_weights": {"B": 0.5, "C": 0.5},
            "source": "synthetic fixed opening weights",
        }
        for n in range(4, 8)
    ]
    return {
        "schema": "quant-risk.forecast-study/v1",
        "evidence_kind": "synthetic",
        "frequency": "one daily observation",
        "config": {
            "model_kind": "statistical_proxy",
            "min_periods": 4,
            "min_assets_per_period": 3,
            "min_asset_observations": 4,
            "annualization": 252,
            "covariance_shrinkage": 0.2,
            "specific_variance_shrinkage": 0.2,
        },
        "lookback": 4,
        "exposures": [
            {
                "effective_at": starts[0].isoformat(),
                "available_at": starts[0].isoformat(),
                "values": {"market": {"A": 1.0, "B": 1.0, "C": 1.0}},
                "source": "synthetic market proxy",
            }
        ],
        "returns": returns,
        "requests": requests,
        "evaluation_as_of": starts[-1].isoformat(),
    }


def forecast_one(value):
    return forecast_factor_risk(
        request=RiskForecastRequest(**value["requests"][0]),
        exposure_snapshots=[
            ExposureSnapshot(**{**e, "values": pd.DataFrame(e["values"])})
            for e in value["exposures"]
        ],
        return_observations=[
            AssetReturnObservation(**{**r, "values": pd.Series(r["values"])})
            for r in value["returns"]
        ],
        config=FactorModelConfig(**value["config"]),
        lookback=value["lookback"],
    )


def test_hand_calculated_covariance_units_and_outcome():
    value = spec()
    forecasts, report = evaluate_risk_spec(value)
    # Three assets share a single scalar return driver; the OLS market loading
    # is the cross-sectional mean and residuals are the deviations from it.
    loadings = np.array([1.0, 0.8, -0.3])
    market = loadings.mean()
    sample_variance = np.var([0.01, -0.01, 0.02, -0.02], ddof=1)
    raw_specific = (loadings - market) ** 2 * sample_variance
    specific = 0.8 * raw_specific + 0.2 * raw_specific.mean()
    sigma = np.ones((3, 3)) * market**2 * sample_variance + np.diag(specific)
    w, b = np.array([0.5, 0.3, 0]), np.array([0, 0.5, 0.5])
    first = forecasts[0]
    assert first["period_variance"]["portfolio"] == pytest.approx(w @ sigma @ w)
    assert first["period_variance"]["active"] == pytest.approx((w - b) @ sigma @ (w - b))
    row = report["scores"][0]
    assert row["realized_return"] == pytest.approx(0.03 * (0.5 + 0.3 * 0.8))
    assert row["qlike"] == pytest.approx(
        math.log(w @ sigma @ w) + row["realized_return"] ** 2 / (w @ sigma @ w)
    )
    assert report["forecast_periods"] == 4
    assert len(report["scores"]) == 20
    assert report["is_proxy"] is True
    assert report["summary"]["portfolio"]["periods"] == 4
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_future_poison_cannot_change_past_forecast_but_changes_outcome():
    original = spec()
    first = forecast_one(original)
    poison = deepcopy(original)
    poison["returns"][4]["values"] = {"A": 0.9, "B": -0.8, "C": 0.7}
    poison["exposures"].append(
        {
            **poison["exposures"][0],
            "effective_at": "2025-01-06T00:00:00Z",
            "available_at": "2025-01-06T00:00:00Z",
            "values": {"market": {"A": 100, "B": 10, "C": 50}},
        }
    )
    assert forecast_one(poison) == first
    _, a = evaluate_risk_spec(original)
    _, b = evaluate_risk_spec(poison)
    assert a["scores"][0]["realized_return"] != b["scores"][0]["realized_return"]


def test_late_return_not_in_training_until_available():
    value = spec()
    value["requests"] = value["requests"][1:]
    value["returns"][4]["available_at"] = "2025-01-07T00:00:00Z"
    first = forecast_one(value)
    assert first["training"]["last_period_end"] == "2025-01-05T00:00:00+00:00"
    assert first["training"]["observations"] == 4


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda s: s["requests"][0].update(weights_available_at="2025-01-05T12:00:00Z"),
            "weights available",
        ),
        (lambda s: s["requests"][0].update(period_end=s["requests"][0]["period_start"]), "horizon"),
        (lambda s: s["requests"][0].update(weights={"MISSING": 1}), "missing"),
        (lambda s: s["requests"][0].update(source=""), "source"),
        (lambda s: s["requests"][0].update(weights={}), "explicitly"),
        (lambda s: s["requests"][0].update(weights={"A": float("nan")}), "finite"),
        (lambda s: s.update(evaluation_as_of="2025-01-08T00:00:00Z"), "completed, available"),
        (
            lambda s: s["returns"][4].update(available_at="2025-01-20T00:00:00Z"),
            "completed, available",
        ),
        (lambda s: s["returns"].append(deepcopy(s["returns"][0])), "unique"),
        (lambda s: s["requests"].append(deepcopy(s["requests"][0])), "unique"),
        (lambda s: s.update(lookback=True), "lookback"),
        (lambda s: s.update(lookback=3), "lookback"),
        (lambda s: s.update(evidence_kind="real"), "evidence_kind"),
        (lambda s: s.update(frequency=""), "frequency"),
        (lambda s: s.update(unknown=1), "schema"),
        (lambda s: s["requests"][1].update(benchmark_weights=None), "coverage changed"),
    ],
)
def test_invalid_studies_fail_instead_of_dropping_rows(change, match):
    value = spec()
    change(value)
    with pytest.raises((ValueError, TypeError), match=match):
        evaluate_risk_spec(value)


def test_zero_variance_rows_are_retained_without_epsilon_or_partial_scores():
    zero = _score(0.02, 0)
    assert zero["qlike"] is None and zero["standardized_return"] is None
    summary = _summary([zero, _score(0.01, 0.0001)], 252)
    assert summary["periods"] == 2
    assert summary["zero_variance_nonzero_outcomes"] == 1
    assert summary["bias_statistic"] is None
    assert summary["mean_qlike"] is None
    assert _summary([_score(0.0, 0.0)], 252)["realized_volatility_annualized"] is None


def test_metrics_match_direct_calculation():
    rows = [_score(r, v) for r, v in zip([0.01, -0.03, 0.02], [0.0001, 0.0004, 0.0001])]
    result = _summary(rows, 252)
    normalized = [1.0, -1.5, 2.0]
    assert result["bias_statistic"] == pytest.approx(np.std(normalized, ddof=1))
    assert result["normalized_rms"] == pytest.approx(np.sqrt(np.mean(np.square(normalized))))
    assert result["variance_mse"] == pytest.approx(
        np.mean([(r * r - v) ** 2 for r, v in zip([0.01, -0.03, 0.02], [0.0001, 0.0004, 0.0001])])
    )


def test_forecast_tampering_and_fixed_configuration_rejected():
    value = spec()
    forecasts, _ = evaluate_risk_spec(value)
    returns = [
        AssetReturnObservation(**{**r, "values": pd.Series(r["values"])}) for r in value["returns"]
    ]
    damaged = deepcopy(forecasts)
    damaged[0]["period_variance"]["portfolio"] *= 2
    with pytest.raises(ValueError, match="content changed"):
        evaluate_risk_forecasts(
            forecasts=damaged,
            return_observations=returns,
            evaluation_as_of=value["evaluation_as_of"],
        )
    damaged[0]["sha256"] = _digest({k: v for k, v in damaged[0].items() if k != "sha256"})
    with pytest.raises(ValueError, match="differs from covariance"):
        evaluate_risk_forecasts(
            forecasts=damaged,
            return_observations=returns,
            evaluation_as_of=value["evaluation_as_of"],
        )
    damaged = deepcopy(forecasts)
    damaged[1]["lookback"] += 1
    damaged[1]["sha256"] = _digest({k: v for k, v in damaged[1].items() if k != "sha256"})
    with pytest.raises(ValueError, match="fixed model"):
        evaluate_risk_forecasts(
            forecasts=damaged,
            return_observations=returns,
            evaluation_as_of=value["evaluation_as_of"],
        )


def test_cli_roundtrip_and_rehashed_scores_cannot_hide_changes(tmp_path, capsys):
    source = tmp_path / "study.json"
    source.write_text(json.dumps(spec()), encoding="utf-8")
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    output = tmp_path / "result"
    main(["validate-forecasts", "--input", str(source), "--out", str(output)])
    main(["verify-forecasts", "--run", str(output)])
    assert '"forecast_periods": 4' in capsys.readouterr().out
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    with pytest.raises(SystemExit) as error:
        main(["validate-forecasts", "--input", str(source), "--out", str(output)])
    assert error.value.code == 2
    (output / "scores.csv").write_text("fake\n1\n", encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text())
    import hashlib

    manifest["files"]["scores.csv"] = hashlib.sha256(
        (output / "scores.csv").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="scores do not reproduce"):
        verify_risk_validation(output)


def test_missing_outcome_creates_no_success_output(tmp_path):
    value = spec()
    value["returns"].pop()
    source = tmp_path / "study.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exact completed"):
        run_risk_validation(source, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_input_cannot_be_contained_by_output(tmp_path):
    source = tmp_path / "input.json"
    with pytest.raises(ValueError, match="contain"):
        run_risk_validation(source, tmp_path)


def test_request_copies_weights_and_readonly_mapping():
    value = spec()["requests"][0]
    request = RiskForecastRequest(**value)
    value["weights"]["A"] = 999
    assert request.weights["A"] == 0.5
    with pytest.raises(TypeError):
        request.weights["A"] = 123


@pytest.mark.parametrize("cash", [False, True])
def test_no_benchmark_and_explicit_full_cash(cash):
    value = spec()
    for request in value["requests"]:
        request["benchmark_weights"] = None
        if cash:
            request["weights"] = {"A": 0.0}
    forecasts, report = evaluate_risk_spec(value)
    assert all(f["period_variance"]["active"] is None for f in forecasts)
    assert set(report["summary"]) == {"portfolio", "asset:A", "asset:B", "asset:C"}
    if cash:
        portfolio = report["summary"]["portfolio"]
        assert portfolio["zero_variance_forecasts"] == 4
        assert portfolio["zero_variance_nonzero_outcomes"] == 0
        assert portfolio["mean_return"] == portfolio["variance_mse"] == 0
        assert portfolio["bias_statistic"] is None


def test_identical_portfolio_and_benchmark_preserve_zero_active_risk():
    value = spec()
    for request in value["requests"]:
        request["benchmark_weights"] = dict(request["weights"])
    _, report = evaluate_risk_spec(value)
    active = report["summary"]["active"]
    assert active["zero_variance_forecasts"] == 4
    assert active["zero_variance_nonzero_outcomes"] == 0
    assert active["normalized_metrics_status"] == "unavailable_zero_variance"
    assert active["variance_mse"] == 0


def test_late_exposure_revision_does_not_rewrite_training_factor_returns():
    original = spec()
    first = forecast_one(original)
    revised = deepcopy(original)
    revised["exposures"].append(
        {
            **revised["exposures"][0],
            "available_at": "2025-01-05T00:00:00Z",
            "values": {"market": {"A": 2.0, "B": 2.0, "C": 2.0}},
        }
    )
    current = forecast_one(revised)
    assert current["model"]["factor_returns"] == first["model"]["factor_returns"]
    assert current["model"]["exposures"] != first["model"]["exposures"]
    revised["exposures"][1]["available_at"] = "2025-01-06T00:00:00Z"
    assert forecast_one(revised) == first


@pytest.mark.parametrize(
    "damage", ["extra", "missing", "content", "manifest_field", "manifest_list"]
)
def test_run_file_set_and_hash_changes_rejected(tmp_path, damage):
    source = tmp_path / "study.json"
    source.write_text(json.dumps(spec()), encoding="utf-8")
    output = tmp_path / "run"
    run_risk_validation(source, output)
    if damage == "extra":
        (output / "extra.json").write_text("{}", encoding="utf-8")
    elif damage == "missing":
        (output / "scores.csv").unlink()
    elif damage == "content":
        (output / "evaluation.json").write_text("{}", encoding="utf-8")
    else:
        path = output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        if damage == "manifest_list":
            manifest = ["schema", "files"]
        else:
            manifest["passed"] = True
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed|manifest"):
        verify_risk_validation(output)


def test_evidence_changed_during_recomputation_is_rejected(tmp_path, monkeypatch):
    import quant_risk_monitor.risk_validation as validation

    source = tmp_path / "study.json"
    source.write_text(json.dumps(spec()), encoding="utf-8")
    output = tmp_path / "run"
    run_risk_validation(source, output)
    original = validation.evaluate_risk_spec

    def change_after_calculation(value):
        result = original(value)
        with (output / "input.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return result

    monkeypatch.setattr(validation, "evaluate_risk_spec", change_after_calculation)
    with pytest.raises(ValueError, match="changed during verification"):
        verify_risk_validation(output)


def test_rehashed_model_and_future_training_rejected():
    from quant_risk_monitor.risk_validation import _validate_forecast

    original = forecast_one(spec())
    damaged = deepcopy(original)
    damaged["model"]["asset_covariance"]["data"][0][0] *= 2
    damaged["model_sha256"] = _digest(damaged["model"])
    damaged["sha256"] = _digest({k: v for k, v in damaged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="inconsistent with X F"):
        _validate_forecast(damaged)
    damaged = deepcopy(original)
    damaged["training"]["latest_return_available_at"] = "2025-01-06T00:00:00Z"
    damaged["sha256"] = _digest({k: v for k, v in damaged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="future training"):
        _validate_forecast(damaged)
