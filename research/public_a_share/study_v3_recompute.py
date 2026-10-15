"""Rebuild inputs, exposure history and all forecasts in an empty cache directory."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import build_panel
import exposure_v2_study as prior
from study_v3_archive import digest, write_json
from study_v3_inputs import freeze, prepare
from study_v3_risk import run as risk_run


def run(old, exposure, reference, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("independent rebuild requires an empty output/cache directory")
    out.mkdir(parents=True, exist_ok=True)
    build_panel.ROOT = old
    raw_prices = build_panel.read_prices()
    with (old / "clean/panel.pkl").open("rb") as stream:
        old_panel = pickle.load(stream)
    for name, expected in old_panel["prices"].items():
        expected = expected.copy()
        expected.index = expected.index.as_unit("ns")
        expected.columns = expected.columns.astype(object)
        actual = raw_prices[name].reindex(index=expected.index, columns=expected.columns)
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_names=False,
            check_freq=False,
            atol=1e-12,
            rtol=1e-12,
        )
    protocol = freeze(old, exposure, out)
    bundle = prepare(old, exposure, out, protocol)
    with (reference / "panel.pkl").open("rb") as stream:
        expected_bundle = pickle.load(stream)
    for section in ("prices", "descriptors"):
        for key, value in expected_bundle["panel"][section].items():
            pd.testing.assert_frame_equal(bundle["panel"][section][key], value)
    for key, value in expected_bundle["v2"]["descriptors"].items():
        pd.testing.assert_frame_equal(bundle["v2"]["descriptors"][key], value)
    print(
        json.dumps(
            {"phase": "raw_prices_and_descriptors_rebuilt", "assets": len(old_panel["assets"])}
        ),
        flush=True,
    )
    prior.VARIANTS = {name: prior.VARIANTS[name] for name in protocol["variants"]}
    prior.build_histories(
        bundle["panel"], bundle["v2"], out, {**protocol, "maximum_exposure_failure_fraction": 0.05}
    )
    write_json(
        out / "history-binding-v3.json",
        {name: digest(out / (name + ".pkl")) for name in ("panel", "histories")},
    )
    risk_run(out)
    actual = pd.read_parquet(out / "risk-forecasts.parquet")
    expected = pd.read_parquet(reference / "risk-forecasts.parquet")
    pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=1e-12)
    for name in ("risk-summary.json", "risk-calibration.json", "endpoint-issues.json"):
        if json.loads((out / name).read_text(encoding="utf8")) != json.loads(
            (reference / name).read_text(encoding="utf8")
        ):
            raise ValueError("independent result differs: " + name)
    # Independent dense arithmetic on a prespecified evenly spaced subset, only
    # in validation. Normal research/optimization never materializes asset covariance.
    with (out / "risk-models.pkl").open("rb") as stream:
        models = pickle.load(stream)
    selected = sorted(models)[::17]
    errors = []
    for day in selected:
        entry = models[day]
        for data in entry["models"].values():
            x = data["exposures"].to_numpy()
            w = entry["benchmark"].reindex(data["exposures"].index).to_numpy() * 0.8
            dense = x @ data["F_daily_rate"] @ x.T + np.diag(data["D_daily_rate"])
            e = x.T @ w
            direct = float(e @ data["F_daily_rate"] @ e + np.dot(w * w, data["D_daily_rate"]))
            errors.append(abs(float(w @ dense @ w) - direct))
    if max(errors) > 1e-12:
        raise ValueError("dense risk reference discrepancy")
    receipt = {
        "status": "passed",
        "assets": len(old_panel["assets"]),
        "forecast_rows_recomputed": len(actual),
        "fresh_exposure_cache": True,
        "raw_price_sources_reparsed": True,
        "financial_inputs": "archived extracted facts; original PDF hashes separately audited",
        "dense_reference_cases": len(errors),
        "maximum_dense_absolute_error": max(errors),
        "reference_forecasts_sha256": digest(reference / "risk-forecasts.parquet"),
        "recomputed_forecasts_sha256": digest(out / "risk-forecasts.parquet"),
    }
    write_json(out / "independent-recomputation.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("old", "exposure", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.old, args.exposure, args.reference, args.out)
