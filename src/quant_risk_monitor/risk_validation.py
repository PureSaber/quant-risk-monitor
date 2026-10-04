"""Causal, one-period factor-risk forecasts and subsequent outcome evaluation.

Squared returns are noisy variance proxies under an explicit zero-mean assumption.
This module evaluates a fixed specification; it neither tunes nor certifies a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_risk_monitor.factor_model import (
    AssetReturnObservation,
    BarraStyleRiskModel,
    ExposureSnapshot,
    FactorModelConfig,
    FactorModelDiagnostics,
    _portfolio_weights,
    _timestamp,
    fit_barra_style_risk_model,
)


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _weights(value, name):
    normalized = _portfolio_weights(value, name)
    if normalized.empty:
        raise ValueError(f"{name} must explicitly specify at least one asset, including for cash")
    return MappingProxyType({str(k): float(v) for k, v in normalized.items()})


@dataclass(frozen=True, kw_only=True)
class RiskForecastRequest:
    """Opening NAV fractions held fixed over one declared model observation period."""

    period_start: pd.Timestamp | str
    period_end: pd.Timestamp | str
    weights_available_at: pd.Timestamp | str
    weights: Mapping[str, float]
    source: str
    benchmark_weights: Mapping[str, float] | None = None

    def __post_init__(self):
        start = _timestamp(self.period_start, "period_start")
        end = _timestamp(self.period_end, "period_end")
        available = _timestamp(self.weights_available_at, "weights_available_at")
        if start >= end or available > start:
            raise ValueError("forecast horizon must be positive and weights available at its start")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("forecast weights require an explicit source")
        object.__setattr__(self, "period_start", start)
        object.__setattr__(self, "period_end", end)
        object.__setattr__(self, "weights_available_at", available)
        object.__setattr__(self, "weights", _weights(self.weights, "weights"))
        if self.benchmark_weights is not None:
            object.__setattr__(
                self, "benchmark_weights", _weights(self.benchmark_weights, "benchmark_weights")
            )

    def to_dict(self):
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "weights_available_at": self.weights_available_at.isoformat(),
            "weights": dict(self.weights),
            "benchmark_weights": None
            if self.benchmark_weights is None
            else dict(self.benchmark_weights),
            "source": self.source,
        }


def _return_payload(item):
    return {
        "period_start": item.period_start.isoformat(),
        "period_end": item.period_end.isoformat(),
        "available_at": item.available_at.isoformat(),
        "values": {str(k): float(v) for k, v in item.values.items()},
        "source": item.source,
    }


def _exposure_payload(item):
    return {
        "effective_at": item.effective_at.isoformat(),
        "available_at": item.available_at.isoformat(),
        "values": {
            str(k): {str(a): float(v) for a, v in col.items()} for k, col in item.values.items()
        },
        "source": item.source,
    }


def _validated_returns(values):
    observations = []
    for item in values:
        if not isinstance(item, AssetReturnObservation):
            raise TypeError("returns must contain AssetReturnObservation values")
        # Reconstruct to validate even if a caller mutated the original Series.
        observations.append(
            AssetReturnObservation(**_return_payload(item) | {"values": item.values})
        )
    observations.sort(key=lambda item: (item.period_start, item.period_end))
    for previous, current in pairwise(observations):
        if current.period_start < previous.period_end:
            raise ValueError("return periods must be unique and non-overlapping")
    return observations


def forecast_factor_risk(
    *,
    request: RiskForecastRequest,
    exposure_snapshots: Sequence[ExposureSnapshot],
    return_observations: Sequence[AssetReturnObservation],
    config: FactorModelConfig,
    lookback: int,
):
    """Fit solely on inputs effective and available by the forecast's opening time.

    The horizon is one observation of the caller's declared frequency. Annualized
    covariance is divided by annualization once, never scaled by calendar days.
    """
    if not isinstance(request, RiskForecastRequest) or not isinstance(config, FactorModelConfig):
        raise TypeError("request/config must be RiskForecastRequest/FactorModelConfig")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < config.min_periods:
        raise ValueError("lookback must be an integer at least min_periods")
    observations = _validated_returns(return_observations)
    known = [
        item
        for item in observations
        if item.period_end <= request.period_start and item.available_at <= request.period_start
    ][-lookback:]
    snapshots = []
    for item in exposure_snapshots:
        if not isinstance(item, ExposureSnapshot):
            raise TypeError("exposures must contain ExposureSnapshot values")
        copy = ExposureSnapshot(
            effective_at=item.effective_at,
            available_at=item.available_at,
            values=item.values,
            source=item.source,
        )
        if copy.effective_at <= request.period_start and copy.available_at <= request.period_start:
            snapshots.append(copy)
    if any(not item.source.strip() for item in [*known, *snapshots]):
        raise ValueError("selected training inputs require explicit sources")
    model = fit_barra_style_risk_model(
        exposure_snapshots=snapshots,
        return_observations=known,
        as_of=request.period_start,
        config=config,
    )
    report = model.analyze(request.weights, request.benchmark_weights)
    model_payload = model.to_dict()
    prediction = {
        "portfolio": report.portfolio_risk.total_variance / config.annualization,
        "active": None
        if report.tracking_risk is None
        else report.tracking_risk.total_variance / config.annualization,
        "assets": {
            str(asset): float(model.asset_covariance.loc[asset, asset]) / config.annualization
            for asset in model.exposures.index
        },
    }
    payload = {
        "schema": "quant-risk.factor-forecast/v1",
        "request": request.to_dict(),
        "config": asdict(config),
        "lookback": lookback,
        "model_kind": model.model_kind,
        "is_proxy": model.model_kind == "statistical_proxy",
        "training": {
            "observations": len(known),
            "first_period_start": known[0].period_start.isoformat(),
            "last_period_end": known[-1].period_end.isoformat(),
            "latest_return_available_at": max(item.available_at for item in known).isoformat(),
            "input_sha256": _digest(
                {
                    "exposures": [_exposure_payload(item) for item in snapshots],
                    "returns": [_return_payload(item) for item in known],
                }
            ),
        },
        "model_sha256": _digest(model_payload),
        "model": model_payload,
        "period_variance": prediction,
        "assumptions": {
            "horizon": "one model observation period",
            "conditional_mean": 0.0,
            "portfolio": "fixed opening NAV fractions; unallocated cash earns zero; no fees or financing",
            "weights_timing": "portfolio and benchmark weights both known by period_start",
        },
    }
    return {**payload, "sha256": _digest(payload)}


def _restore_model(payload):
    def frame(name, *, dates=False):
        item = payload[name]
        index = pd.to_datetime(item["index"], utc=True) if dates else item["index"]
        return pd.DataFrame(item["data"], index=index, columns=item["columns"])

    diagnostics = dict(payload["diagnostics"])
    diagnostics.pop("is_proxy")
    for name in (
        "as_of",
        "first_period_start",
        "last_period_end",
        "exposure_effective_at",
        "exposure_available_at",
    ):
        diagnostics[name] = _timestamp(diagnostics[name], name)
    for name in (
        "factors",
        "assets",
        "cross_section_sizes",
        "regression_ranks",
        "condition_numbers",
    ):
        diagnostics[name] = tuple(diagnostics[name])
    specific = payload["specific_variances"]
    model = BarraStyleRiskModel(
        model_kind=payload["model_kind"],
        as_of=payload["as_of"],
        annualization=payload["annualization"],
        exposures=frame("exposures"),
        factor_returns=frame("factor_returns", dates=True),
        factor_covariance=frame("factor_covariance"),
        specific_variances=pd.Series(specific["data"], index=specific["index"]),
        asset_covariance=frame("asset_covariance"),
        diagnostics=FactorModelDiagnostics(**diagnostics),
    )
    if model.to_dict() != payload:
        raise ValueError("stored risk model fields are inconsistent")
    return model


def _validate_forecast(forecast):
    if forecast.get("schema") != "quant-risk.factor-forecast/v1":
        raise ValueError("unsupported risk forecast schema")
    if _digest({k: v for k, v in forecast.items() if k != "sha256"}) != forecast.get("sha256"):
        raise ValueError("risk forecast content changed")
    request = RiskForecastRequest(**forecast["request"])
    config = FactorModelConfig(**forecast["config"])
    lookback = forecast["lookback"]
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < config.min_periods:
        raise ValueError("invalid forecast lookback")
    if forecast["model_kind"] != config.model_kind or forecast["is_proxy"] != (
        config.model_kind == "statistical_proxy"
    ):
        raise ValueError("forecast model classification changed")
    if _digest(forecast["model"]) != forecast["model_sha256"]:
        raise ValueError("forecast model content changed")
    if _timestamp(forecast["model"]["as_of"], "model.as_of") != request.period_start:
        raise ValueError("forecast model must be fitted at period_start")
    for name in ("last_period_end", "latest_return_available_at"):
        if _timestamp(forecast["training"][name], name) > request.period_start:
            raise ValueError("future training data in forecast")
    model = _restore_model(forecast["model"])
    diagnostics = model.diagnostics
    if model.model_kind != config.model_kind or model.annualization != config.annualization:
        raise ValueError("stored model differs from forecast configuration")
    if (
        diagnostics.exposure_effective_at > request.period_start
        or diagnostics.exposure_available_at > request.period_start
    ):
        raise ValueError("future exposure in forecast")
    if (
        diagnostics.periods != forecast["training"]["observations"]
        or not config.min_periods <= diagnostics.periods <= lookback
        or diagnostics.first_period_start.isoformat() != forecast["training"]["first_period_start"]
        or diagnostics.last_period_end.isoformat() != forecast["training"]["last_period_end"]
        or diagnostics.covariance_shrinkage != config.covariance_shrinkage
        or diagnostics.specific_variance_shrinkage != config.specific_variance_shrinkage
    ):
        raise ValueError("forecast training diagnostics are inconsistent")
    covariance = model.asset_covariance / config.annualization
    expected_assets = {str(a): float(covariance.loc[a, a]) for a in covariance.index}
    if forecast["period_variance"]["assets"] != expected_assets:
        raise ValueError("asset forecast differs from covariance")

    report = model.analyze(request.weights, request.benchmark_weights)
    expected = {
        "portfolio": report.portfolio_risk.total_variance / config.annualization,
        "active": None
        if report.tracking_risk is None
        else report.tracking_risk.total_variance / config.annualization,
    }
    for name, value in expected.items():
        actual = forecast["period_variance"][name]
        if value is None:
            if actual is not None:
                raise ValueError("unexpected active risk forecast")
        elif (
            not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or actual < 0
            or not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-18)
        ):
            raise ValueError("portfolio forecast differs from covariance")
    return request, config


def _score(realized, variance):
    squared = realized * realized
    if variance < 0 or not math.isfinite(variance) or not math.isfinite(squared):
        raise ValueError("risk score requires finite returns and nonnegative variance")
    positive = variance > 0
    result = {
        "realized_return": realized,
        "variance_proxy": squared,
        "predicted_variance": variance,
        "predicted_volatility": math.sqrt(variance),
        "variance_squared_error": (squared - variance) ** 2,
        "standardized_return": realized / math.sqrt(variance) if positive else None,
        "qlike": math.log(variance) + squared / variance if positive else None,
        "normalization_status": "available" if positive else "zero_forecast_variance",
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise ValueError("risk score overflow; no finite calibration score")
    return result


def _summary(rows, annualization):
    realized = np.array([r["realized_return"] for r in rows])
    predicted = np.array([r["predicted_variance"] for r in rows])
    zero = sum(r["normalization_status"] != "available" for r in rows)
    standardized = None if zero else np.array([r["standardized_return"] for r in rows])
    return {
        "periods": len(rows),
        "zero_variance_forecasts": zero,
        "zero_variance_nonzero_outcomes": sum(
            r["predicted_variance"] == 0 and r["realized_return"] != 0 for r in rows
        ),
        "mean_return": float(realized.mean()),
        "realized_volatility_annualized": float(realized.std(ddof=1) * math.sqrt(annualization))
        if len(rows) > 1
        else None,
        "forecast_rms_volatility_annualized": float(math.sqrt(predicted.mean() * annualization)),
        "variance_mse": float(np.mean([r["variance_squared_error"] for r in rows])),
        "normalized_metrics_status": "unavailable_zero_variance" if zero else "available",
        "normalized_mean": None if zero else float(standardized.mean()),
        "normalized_rms": None if zero else float(np.sqrt(np.mean(standardized**2))),
        "bias_statistic": None if zero or len(rows) < 2 else float(standardized.std(ddof=1)),
        "mean_qlike": None if zero else float(np.mean([r["qlike"] for r in rows])),
    }


def evaluate_risk_forecasts(*, forecasts, return_observations, evaluation_as_of):
    """Score every declared forecast against its exact, mature return interval.

    Any missing outcome or inconsistent forecast rejects the evaluation. Zero-risk
    rows remain visible and make normalized aggregate scores unavailable, not zero.
    """
    if not forecasts:
        raise ValueError("at least one forecast is required")
    cutoff = _timestamp(evaluation_as_of, "evaluation_as_of")
    observations = _validated_returns(return_observations)
    outcomes = {(r.period_start, r.period_end): r for r in observations}
    prepared = [(forecast, *_validate_forecast(forecast)) for forecast in forecasts]
    prepared.sort(key=lambda item: item[1].period_start)
    kind = prepared[0][2].model_kind
    configuration = prepared[0][0]["config"]
    lookback = prepared[0][0]["lookback"]
    asset_set = set(prepared[0][0]["period_variance"]["assets"])
    benchmark_present = prepared[0][1].benchmark_weights is not None
    rows, prior_end = [], None
    used_outcomes = []
    for forecast, request, config in prepared:
        if forecast["config"] != configuration or forecast["lookback"] != lookback:
            raise ValueError("one evaluation requires one fixed model configuration")
        if (
            set(forecast["period_variance"]["assets"]) != asset_set
            or (request.benchmark_weights is not None) != benchmark_present
        ):
            raise ValueError("forecast coverage changed within evaluation")
        if prior_end is not None and request.period_start < prior_end:
            raise ValueError("forecast periods must be unique and non-overlapping")
        prior_end = request.period_end
        outcome = outcomes.get((request.period_start, request.period_end))
        if outcome is None or outcome.period_end > cutoff or outcome.available_at > cutoff:
            raise ValueError("every forecast requires its exact completed, available outcome")
        if not outcome.source.strip() or set(outcome.values.index) != asset_set:
            raise ValueError("outcome requires source and complete forecast asset coverage")
        used_outcomes.append(_return_payload(outcome))
        values = outcome.values.to_dict()
        realized = {"portfolio": sum(w * values.get(a, 0.0) for a, w in request.weights.items())}
        variances = {"portfolio": forecast["period_variance"]["portfolio"]}
        if benchmark_present:
            realized["active"] = realized["portfolio"] - sum(
                w * values.get(a, 0.0) for a, w in request.benchmark_weights.items()
            )
            variances["active"] = forecast["period_variance"]["active"]
        for asset in sorted(asset_set):
            realized["asset:" + asset] = float(values[asset])
            variances["asset:" + asset] = forecast["period_variance"]["assets"][asset]
        for series, value in realized.items():
            rows.append(
                {
                    "period_start": request.period_start.isoformat(),
                    "period_end": request.period_end.isoformat(),
                    "outcome_available_at": outcome.available_at.isoformat(),
                    "forecast_sha256": forecast["sha256"],
                    "series": series,
                    **_score(float(value), variances[series]),
                }
            )
    summary = {
        series: _summary([r for r in rows if r["series"] == series], configuration["annualization"])
        for series in sorted({r["series"] for r in rows})
    }
    result = {
        "schema": "quant-risk.forecast-evaluation/v1",
        "evaluation_as_of": cutoff.isoformat(),
        "status": "complete",
        "model_kind": kind,
        "is_proxy": kind == "statistical_proxy",
        "forecast_periods": len(prepared),
        "config": configuration,
        "lookback": lookback,
        "first_period_start": prepared[0][1].period_start.isoformat(),
        "last_period_end": prepared[-1][1].period_end.isoformat(),
        "forecast_sha256": [f["sha256"] for f, _, _ in prepared],
        "outcomes_sha256": _digest(used_outcomes),
        "scores": rows,
        "summary": summary,
        "definitions": {
            "variance_proxy": "squared period return, conditional mean assumed zero",
            "qlike": "log(predicted_variance) + squared_return / predicted_variance",
            "bias_statistic": "sample standard deviation of return / predicted_volatility (ddof=1)",
            "horizon": "one input observation period",
            "portfolio": "fixed opening NAV fractions, zero-return residual cash, no trading or financing costs",
        },
        "limitations": [
            "squared returns are noisy proxies, not observed latent variance",
            "completed evaluation is not a passing calibration gate",
            "historical causal filtering does not certify input provenance or independent forward evidence",
            "asset scores evaluate total variance, not separately identified latent specific variance",
            "no automatic model selection, calibration multiplier or risk-limit change",
            "normalized aggregates are unavailable if any requested variance is zero",
        ],
    }
    _canonical(result)
    return result


def evaluate_risk_spec(spec):
    """Rebuild all causal forecasts from a complete, explicitly classified input."""
    required = {
        "schema",
        "evidence_kind",
        "frequency",
        "config",
        "lookback",
        "exposures",
        "returns",
        "requests",
        "evaluation_as_of",
    }
    if set(spec) != required or spec["schema"] != "quant-risk.forecast-study/v1":
        raise ValueError("complete risk forecast study schema required; unknown fields rejected")
    if spec["evidence_kind"] not in {"synthetic", "retrospective", "historical_pit"}:
        raise ValueError("explicit evidence_kind required")
    if not isinstance(spec["frequency"], str) or not spec["frequency"].strip():
        raise ValueError("explicit observation frequency required")
    exposures = [
        ExposureSnapshot(**{**item, "values": pd.DataFrame(item["values"])})
        for item in spec["exposures"]
    ]
    returns = [
        AssetReturnObservation(**{**item, "values": pd.Series(item["values"])})
        for item in spec["returns"]
    ]
    config = FactorModelConfig(**spec["config"])
    forecasts = [
        forecast_factor_risk(
            request=RiskForecastRequest(**item),
            exposure_snapshots=exposures,
            return_observations=returns,
            config=config,
            lookback=spec["lookback"],
        )
        for item in spec["requests"]
    ]
    result = evaluate_risk_forecasts(
        forecasts=forecasts, return_observations=returns, evaluation_as_of=spec["evaluation_as_of"]
    )
    result.update(
        evidence_kind=spec["evidence_kind"], frequency=spec["frequency"], input_sha256=_digest(spec)
    )
    return forecasts, result


def run_risk_validation(input_path, output):
    """Write a new source-bound result after complete validation; never replace input."""
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if output == input_path or output in input_path.parents:
        raise ValueError("output must not contain the input")
    if output.exists():
        raise FileExistsError("risk validation output already exists")
    original = input_path.read_bytes()
    spec = json.loads(original)
    forecasts, result = evaluate_risk_spec(spec)
    if input_path.read_bytes() != original:
        raise ValueError("risk validation input changed during evaluation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.json").write_bytes(original)
    for name, payload in (("forecasts.json", forecasts), ("evaluation.json", result)):
        (output / name).write_text(_canonical(payload) + "\n", encoding="utf-8")
    pd.DataFrame(result["scores"]).to_csv(output / "scores.csv", index=False, lineterminator="\n")
    manifest = {
        "schema": "quant-risk.forecast-run/v1",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
    return result


def _read_run_files(output):
    output = Path(output)
    expected = {"input.json", "forecasts.json", "evaluation.json", "scores.csv", "manifest.json"}
    if output.is_symlink() or {p.name for p in output.iterdir()} != expected:
        raise ValueError("risk validation file set changed")
    contents = {}
    for name in sorted(expected):
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("risk validation evidence must be regular files: " + name)
        contents[name] = path.read_bytes()
    return contents


def verify_risk_validation(output):
    """Recompute from one byte snapshot; reject evidence changing during verification."""
    contents = _read_run_files(output)
    expected = set(contents) - {"manifest.json"}
    manifest = json.loads(contents["manifest.json"])
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "quant-risk.forecast-run/v1"
        or not isinstance(manifest.get("files"), dict)
        or set(manifest.get("files", {})) != expected
    ):
        raise ValueError("invalid risk validation manifest")
    for name in expected:
        if hashlib.sha256(contents[name]).hexdigest() != manifest["files"][name]:
            raise ValueError("risk validation evidence changed: " + name)
    forecasts, result = evaluate_risk_spec(json.loads(contents["input.json"]))
    if (
        json.loads(contents["forecasts.json"]) != forecasts
        or json.loads(contents["evaluation.json"]) != result
    ):
        raise ValueError("risk validation results do not reproduce")
    expected_csv = (
        pd.DataFrame(result["scores"]).to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    if contents["scores.csv"] != expected_csv:
        raise ValueError("risk validation scores do not reproduce")
    if _read_run_files(output) != contents:
        raise ValueError("risk validation evidence changed during verification")
    return result
