"""Reproducible exposure builds and chronological risk-parameter calibration."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_risk_monitor.exposure_builder import ExposureBuildConfig, build_exposure_snapshot
from quant_risk_monitor.factor_model import FactorModelConfig, _timestamp
from quant_risk_monitor.risk_validation import (
    RiskForecastRequest,
    _canonical,
    _digest,
    _exposure_payload,
    evaluate_risk_spec,
)


def build_exposure_spec(spec):
    required = {
        "schema",
        "evidence_kind",
        "config",
        "assets",
        "as_of",
        "descriptor_history",
        "classification_history",
    }
    if set(spec) != required or spec["schema"] != "quant-risk.exposure-study/v1":
        raise ValueError("complete exposure study schema required; unknown fields rejected")
    if spec["evidence_kind"] not in {"synthetic", "retrospective", "historical_pit"}:
        raise ValueError("explicit evidence_kind required")
    cutoffs = [_timestamp(value, "as_of") for value in spec["as_of"]]
    if not cutoffs or cutoffs != sorted(set(cutoffs)):
        raise ValueError("exposure as_of values must be strictly increasing")
    config = ExposureBuildConfig(**spec["config"])
    exposures = [
        build_exposure_snapshot(
            descriptor_history=pd.DataFrame(spec["descriptor_history"]),
            classification_history=pd.DataFrame(spec["classification_history"]),
            assets=spec["assets"],
            as_of=cutoff,
            config=config,
        )
        for cutoff in cutoffs
    ]
    return {
        "schema": "quant-risk.exposure-result/v1",
        "status": "complete",
        "evidence_kind": spec["evidence_kind"],
        "input_sha256": _digest(spec),
        "exposures": [_exposure_payload(item) for item in exposures],
        "source_authenticated": False,
    }


def calibrate_risk_spec(spec):
    """Choose a frozen candidate on development QLIKE, then score holdout once.

    All candidates use exactly the same portfolios, horizons, observations and
    evidence classification. A failed or zero-risk candidate blocks the study;
    it cannot be silently dropped to improve the apparent selection result.
    """
    required = {
        "schema",
        "evidence_kind",
        "frequency",
        "candidates",
        "baseline_id",
        "lookback",
        "exposures",
        "returns",
        "selection_requests",
        "holdout_requests",
        "selection_as_of",
        "plan_available_at",
        "evaluation_as_of",
    }
    retrospective = spec.get("schema") == "quant-risk.calibration-study/v2"
    if retrospective:
        required = required - {"plan_available_at"} | {"research_design"}
    if set(spec) - {"outcomes"} != required or spec.get("schema") not in {
        "quant-risk.calibration-study/v1",
        "quant-risk.calibration-study/v2",
    }:
        raise ValueError("complete calibration study schema required; unknown fields rejected")
    candidates = spec["candidates"]
    if not isinstance(candidates, dict) or not 2 <= len(candidates) <= 32:
        raise ValueError("declare between two and 32 frozen candidate configurations")
    if any(not isinstance(name, str) or not name.strip() for name in candidates):
        raise ValueError("candidate IDs must be nonempty strings")
    if spec["baseline_id"] not in candidates:
        raise ValueError("baseline_id must name a frozen candidate")
    configs = {name: FactorModelConfig(**value) for name, value in candidates.items()}
    if (
        len(
            {
                (config.model_kind, config.risk_horizon, config.annualization)
                for config in configs.values()
            }
        )
        != 1
    ):
        raise ValueError("candidate classification, horizon and annualization must match")
    selection = [RiskForecastRequest(**item) for item in spec["selection_requests"]]
    holdout = [RiskForecastRequest(**item) for item in spec["holdout_requests"]]
    if len(selection) < 2 or len(holdout) < 2:
        raise ValueError("selection and holdout each require at least two forecasts")
    selection_as_of = _timestamp(spec["selection_as_of"], "selection_as_of")
    evaluation_as_of = _timestamp(spec["evaluation_as_of"], "evaluation_as_of")
    if retrospective:
        design = spec["research_design"]
        if (
            not isinstance(design, dict)
            or set(design) != {"kind", "created_at"}
            or design["kind"] != "retrospective"
        ):
            raise ValueError("v2 requires an explicit retrospective research_design")
        created_at = _timestamp(design["created_at"], "research_design.created_at")
        if created_at > evaluation_as_of:
            raise ValueError("research design cannot be created after evaluation_as_of")
        plan_binding = {
            "research_design": {"kind": "retrospective", "created_at": created_at.isoformat()}
        }
    else:
        plan_at = _timestamp(spec["plan_available_at"], "plan_available_at")
        if plan_at > min(item.period_start for item in selection):
            raise ValueError("candidate plan must be declared before the first selection forecast")
        plan_binding = {"plan_available_at": plan_at.isoformat()}
    if not (
        max(item.period_end for item in selection)
        <= selection_as_of
        <= min(item.period_start for item in holdout)
        < evaluation_as_of
    ):
        raise ValueError("selection must mature and freeze before the untouched holdout starts")
    base = {
        key: deepcopy(spec[key])
        for key in ("evidence_kind", "frequency", "lookback", "exposures", "returns")
    }
    base["schema"] = "quant-risk.forecast-study/v1"
    if "outcomes" in spec:
        base["outcomes"] = deepcopy(spec["outcomes"])
    developments, scores, forecasts = {}, {}, {}
    for name in sorted(candidates):
        study = {
            **base,
            "config": candidates[name],
            "requests": spec["selection_requests"],
            "evaluation_as_of": selection_as_of.isoformat(),
        }
        forecasts[name], developments[name] = evaluate_risk_spec(study)
        score = developments[name]["summary"]["portfolio"]["mean_qlike"]
        if score is None:
            raise ValueError(f"candidate {name} has unavailable selection QLIKE")
        scores[name] = score
    selected = min(scores, key=lambda name: (scores[name], name))
    # This binding contains only causally filtered training inputs and mature
    # selection outcomes, so future/holdout poisoning cannot change selection.
    selection_binding = {
        **plan_binding,
        "candidate_configs": candidates,
        "selection_as_of": selection_as_of.isoformat(),
        "forecast_sha256": {
            name: [row["sha256"] for row in rows] for name, rows in forecasts.items()
        },
        "outcomes_sha256": {
            name: result["outcomes_sha256"] for name, result in developments.items()
        },
        "scores": scores,
        "selected": selected,
    }
    holdout_results = {}
    for name in sorted({selected, spec["baseline_id"]}):
        _, holdout_results[name] = evaluate_risk_spec(
            {
                **base,
                "config": candidates[name],
                "requests": spec["holdout_requests"],
                "evaluation_as_of": evaluation_as_of.isoformat(),
            }
        )
    return {
        "schema": "quant-risk.calibration-result/v2"
        if retrospective
        else "quant-risk.calibration-result/v1",
        "status": "complete",
        "evidence_kind": spec["evidence_kind"],
        "input_sha256": _digest(spec),
        "selection": selection_binding,
        "selection_sha256": _digest(selection_binding),
        "selection_reports": developments,
        "holdout_reports": holdout_results,
        "selected_config": candidates[selected],
        "baseline_id": spec["baseline_id"],
        "selection_metric": "portfolio_qlike",
        "market_calibration_certified": False,
        "limitations": [
            "candidate plan timestamps are assertions, not authenticated preregistration",
            "holdout is scored without reselection; no automatic risk-limit changes",
            "finite candidate search and a completed run do not prove market effectiveness",
        ],
    }


def _evaluate(kind, spec):
    if kind == "exposures":
        return build_exposure_spec(spec)
    if kind == "calibration":
        return calibrate_risk_spec(spec)
    raise ValueError("unknown risk research kind")


def run_risk_research(kind, input_path, output):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if output == input_path or output in input_path.parents:
        raise ValueError("research output must not contain the input")
    if output.exists():
        raise FileExistsError("research output already exists")
    original = input_path.read_bytes()
    result = _evaluate(kind, json.loads(original))
    if input_path.read_bytes() != original:
        raise ValueError("research input changed during computation")
    result_bytes = (_canonical(result) + "\n").encode("utf-8")
    manifest = {
        "schema": "quant-risk.research-run/v1",
        "kind": kind,
        "files": {
            "input.json": hashlib.sha256(original).hexdigest(),
            "result.json": hashlib.sha256(result_bytes).hexdigest(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.json").write_bytes(original)
    (output / "result.json").write_bytes(result_bytes)
    (output / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
    return result


def verify_risk_research(output):
    output = Path(output)
    names = {"input.json", "result.json", "manifest.json"}
    if output.is_symlink() or {path.name for path in output.iterdir()} != names:
        raise ValueError("research run has an invalid file set")
    if any((output / name).is_symlink() or not (output / name).is_file() for name in names):
        raise ValueError("research files must be regular non-symlink files")
    contents = {name: (output / name).read_bytes() for name in names}
    manifest = json.loads(contents["manifest.json"])
    if (
        set(manifest) != {"schema", "kind", "files"}
        or manifest["schema"] != "quant-risk.research-run/v1"
        or set(manifest["files"]) != names - {"manifest.json"}
    ):
        raise ValueError("invalid research manifest")
    for name, expected in manifest["files"].items():
        if hashlib.sha256(contents[name]).hexdigest() != expected:
            raise ValueError("research hash mismatch")
    result = _evaluate(manifest["kind"], json.loads(contents["input.json"]))
    if (_canonical(result) + "\n").encode("utf-8") != contents["result.json"]:
        raise ValueError("research recomputation differs from recorded result")
    if {path.name for path in output.iterdir()} != names or any(
        (output / name).is_symlink() or (output / name).read_bytes() != value
        for name, value in contents.items()
    ):
        raise ValueError("research evidence changed during verification")
    return result
