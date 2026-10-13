import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_risk_monitor.cli import main
from quant_risk_monitor.risk_research import (
    build_exposure_spec,
    calibrate_risk_spec,
    run_risk_research,
    verify_risk_research,
)
from quant_risk_monitor.risk_validation import _canonical, evaluate_risk_spec

fixture = runpy.run_path(str(Path(__file__).parents[1] / "examples/advanced_barra_research.py"))[
    "fixture"
]


def test_complete_advanced_model_and_horizon_variance_identity():
    _, study = fixture()
    forecast_spec = {
        "schema": "quant-risk.forecast-study/v1",
        **{
            key: study[key]
            for key in (
                "evidence_kind",
                "frequency",
                "lookback",
                "exposures",
                "returns",
                "outcomes",
                "evaluation_as_of",
            )
        },
        "config": study["candidates"]["enhanced"],
        "requests": study["selection_requests"],
    }
    forecasts, scores = evaluate_risk_spec(forecast_spec)
    for forecast in forecasts:
        stored = forecast["model"]["asset_covariance"]
        matrix = np.array(stored["data"])
        w = pd.Series(forecast["request"]["weights"]).reindex(stored["index"]).to_numpy()
        assert forecast["period_variance"]["portfolio"] == pytest.approx(w @ matrix @ w * 2 / 252)
        diagnostics = forecast["model"]["diagnostics"]["estimation"]
        assert diagnostics["factor_covariance"]["newey_west_lags"] == 1
        assert diagnostics["factor_regime"]["enabled"]
        assert diagnostics["specific_risk"]["target"] == "industry"
        assert diagnostics["specific_risk"]["method"] == "bayesian"
        assert max(row["constraint_error"] for row in diagnostics["regressions"]) < 1e-12
    realized = [row["realized_return"] for row in scores["scores"] if row["series"] == "portfolio"]
    assert scores["summary"]["portfolio"]["realized_volatility_annualized"] == pytest.approx(
        np.std(realized, ddof=1) * np.sqrt(252 / 2)
    )
    invalid = copy.deepcopy(forecast_spec)
    invalid["requests"][0]["horizon_periods"] = 1
    with pytest.raises(ValueError, match="horizon"):
        evaluate_risk_spec(invalid)
    invalid = copy.deepcopy(forecast_spec)
    invalid.pop("outcomes")
    with pytest.raises(ValueError, match="separate exact-horizon outcomes"):
        evaluate_risk_spec(invalid)


def test_holdout_poison_does_not_change_selected_parameters_or_selection_hash():
    _, study = fixture()
    original = calibrate_risk_spec(study)
    altered = copy.deepcopy(study)
    cutoff = study["selection_as_of"]
    for row in altered["outcomes"]:
        if row["period_start"] >= cutoff:
            row["values"] = {asset: value * 5 for asset, value in row["values"].items()}
    result = calibrate_risk_spec(altered)
    assert result["selection"] == original["selection"]
    assert result["selection_sha256"] == original["selection_sha256"]
    selected = result["selection"]["selected"]
    assert (
        result["holdout_reports"][selected]["summary"]
        != original["holdout_reports"][selected]["summary"]
    )
    assert result["market_calibration_certified"] is False
    assert set(result["holdout_reports"]) == {selected, "baseline"}


def test_hac_calibration_rejects_gaps_in_training_periods():
    _, study = fixture()
    del study["returns"][10]
    with pytest.raises(ValueError, match="contiguous return periods"):
        calibrate_risk_spec(study)


@pytest.mark.parametrize("kind", ["exposures", "calibration"])
def test_cli_and_recomputed_artifacts_reject_rehashed_tampering(tmp_path, capsys, kind):
    exposure, calibration = fixture()
    spec = exposure if kind == "exposures" else calibration
    path = tmp_path / "input.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"
    main(
        [
            "build-exposures" if kind == "exposures" else "calibrate-risk",
            "--input",
            str(path),
            "--out",
            str(output),
        ]
    )
    main(["verify-research", "--run", str(output)])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["verified"]
    result = json.loads((output / "result.json").read_text())
    result["status"] = "forged"
    altered = (_canonical(result) + "\n").encode()
    (output / "result.json").write_bytes(altered)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["result.json"] = hashlib.sha256(altered).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recomputation"):
        verify_risk_research(output)
    with pytest.raises(FileExistsError):
        run_risk_research(kind, path, output)


@pytest.mark.parametrize(
    "mutation",
    ["overlap", "short", "unknown", "horizon", "late_outcome", "baseline", "zero_variance"],
)
def test_invalid_calibration_cannot_silently_drop_bad_candidates(mutation):
    _, study = fixture()
    if mutation == "overlap":
        study["selection_as_of"] = study["holdout_requests"][0]["period_end"]
    elif mutation == "short":
        study["selection_requests"] = study["selection_requests"][:1]
    elif mutation == "unknown":
        study["unknown"] = True
    elif mutation == "horizon":
        study["candidates"]["baseline"]["risk_horizon"] = 1
    elif mutation == "late_outcome":
        study["outcomes"][0]["available_at"] = study["evaluation_as_of"]
    elif mutation == "baseline":
        study["baseline_id"] = "absent"
    else:
        for request in study["selection_requests"]:
            request["weights"] = {"S00": 0.0}
    with pytest.raises(ValueError):
        calibrate_risk_spec(study)


def test_exposure_study_and_artifact_file_guards(tmp_path):
    spec, _ = fixture()
    bad = copy.deepcopy(spec)
    bad["as_of"] *= 2
    with pytest.raises(ValueError, match="strictly increasing"):
        build_exposure_spec(bad)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="contain the input"):
        run_risk_research("exposures", path, tmp_path)
    output = tmp_path / "valid"
    run_risk_research("exposures", path, output)
    (output / "extra").write_text("extra")
    with pytest.raises(ValueError, match="file set"):
        verify_risk_research(output)


def test_late_candidate_plan_cannot_claim_untouched_selection():
    _, study = fixture()
    study["plan_available_at"] = study["selection_as_of"]
    with pytest.raises(ValueError, match="candidate plan"):
        calibrate_risk_spec(study)


def test_retrospective_design_keeps_actual_creation_date_and_holdout_separation():
    _, study = fixture()
    study["schema"] = "quant-risk.calibration-study/v2"
    study.pop("plan_available_at")
    study["research_design"] = {"kind": "retrospective", "created_at": study["evaluation_as_of"]}
    result = calibrate_risk_spec(study)
    assert result["schema"] == "quant-risk.calibration-result/v2"
    assert result["selection"]["research_design"]["kind"] == "retrospective"
    assert "plan_available_at" not in result["selection"]
    altered = copy.deepcopy(study)
    for row in altered["outcomes"]:
        if row["period_start"] >= study["selection_as_of"]:
            row["values"] = {a: r * 7 for a, r in row["values"].items()}
    assert calibrate_risk_spec(altered)["selection_sha256"] == result["selection_sha256"]


@pytest.mark.parametrize("mutation", ["kind", "future_creation", "unknown", "overlap"])
def test_retrospective_design_cannot_bypass_schema_or_time_boundaries(mutation):
    _, study = fixture()
    study["schema"] = "quant-risk.calibration-study/v2"
    study.pop("plan_available_at")
    study["research_design"] = {"kind": "retrospective", "created_at": study["evaluation_as_of"]}
    if mutation == "kind":
        study["research_design"]["kind"] = "historically_registered"
    elif mutation == "future_creation":
        study["research_design"]["created_at"] = "2099-01-01T00:00:00Z"
    elif mutation == "unknown":
        study["research_design"]["certified"] = True
    else:
        study["selection_as_of"] = study["holdout_requests"][0]["period_end"]
    with pytest.raises(ValueError):
        calibrate_risk_spec(study)
