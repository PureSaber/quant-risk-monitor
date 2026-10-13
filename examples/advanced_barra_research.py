"""Deterministic SYNTHETIC exposure and two-period risk calibration study.

Run: python examples/advanced_barra_research.py <new-output-directory>
Then use build-exposures, calibrate-risk and verify-research on the written JSON.
No network, vendor data, forward evidence or market effectiveness is implied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk_monitor.risk_research import build_exposure_spec


def fixture():
    assets = [f"S{i:02d}" for i in range(12)]
    dates = pd.date_range("2025-01-01", periods=37, tz="UTC")
    source_rows, industry_rows = [], []
    for i, asset in enumerate(assets):
        identity = {
            "asset": asset,
            "effective_at": dates[0].isoformat(),
            "available_at": dates[0].isoformat(),
            "source": "synthetic-fixture",
        }
        source_rows.append(
            {
                **identity,
                "market_cap": float((i + 1) * 100),
                "log_size": float(np.log(i + 1)),
                "value": float([2, 7, 1, 9, 3, 8, 4, 12, 6, 10, 5, 15][i]),
            }
        )
        industry_rows.append({**identity, "industry": ["tech", "finance", "other"][i // 4]})
    exposure_study = {
        "schema": "quant-risk.exposure-study/v1",
        "evidence_kind": "synthetic",
        "assets": assets,
        "as_of": [dates[0].isoformat()],
        "descriptor_history": source_rows,
        "classification_history": industry_rows,
        "config": {
            "styles": {"size": {"log_size": 1}, "value": {"value": 1}},
            "industry_universe": ["tech", "finance", "other"],
            "orthogonalize": {"value": ["size"]},
            "winsor_method": "quantile",
            "winsor_limit": 0.05,
        },
    }
    exposures = build_exposure_spec(exposure_study)["exposures"]
    x = pd.DataFrame(exposures[0]["values"]).reindex(assets).to_numpy()
    rng = np.random.default_rng(117)
    daily = []
    returns = []
    for i in range(36):
        coefficients = rng.normal(0, 0.004 * (1 if i < 24 else 1.5), x.shape[1])
        values = x @ coefficients + rng.normal(0, 0.002, len(assets))
        daily.append(values)
        returns.append(
            {
                "period_start": dates[i].isoformat(),
                "period_end": dates[i + 1].isoformat(),
                "available_at": dates[i + 1].isoformat(),
                "source": "synthetic-daily",
                "values": dict(zip(assets, values.tolist(), strict=True)),
            }
        )
    benchmark = {asset: 1 / 12 for asset in assets}
    weights = {asset: (i + 1) / 78 for i, asset in enumerate(assets)}
    requests, outcomes = {}, []
    for label, starts in (("selection", (20, 22, 24)), ("holdout", (26, 28, 30))):
        requests[label] = []
        for start in starts:
            requests[label].append(
                {
                    "period_start": dates[start].isoformat(),
                    "period_end": dates[start + 2].isoformat(),
                    "horizon_periods": 2,
                    "weights_available_at": dates[start].isoformat(),
                    "source": "frozen-fixture-weights",
                    "weights": weights,
                    "benchmark_weights": benchmark,
                }
            )
            outcomes.append(
                {
                    "period_start": dates[start].isoformat(),
                    "period_end": dates[start + 2].isoformat(),
                    "available_at": dates[start + 2].isoformat(),
                    "source": "synthetic-two-period-arithmetic-sum",
                    "values": dict(
                        zip(assets, (daily[start] + daily[start + 1]).tolist(), strict=True)
                    ),
                }
            )
    baseline = {
        "model_kind": "fundamental_style",
        "min_periods": 12,
        "min_asset_observations": 12,
        "min_assets_per_period": 8,
        "industry_constraint": True,
        "risk_horizon": 2,
    }
    enhanced = {
        **baseline,
        "regression_weighting": "sqrt_market_cap",
        "factor_half_life": 12,
        "specific_half_life": 10,
        "newey_west_lags": 1,
        "specific_newey_west_lags": 1,
        "regime_window": 4,
        "regime_min_periods": 8,
        "regime_half_life": 3,
        "specific_target": "industry",
        "specific_shrinkage_method": "bayesian",
        "specific_prior_dof": 10,
    }
    calibration = {
        "schema": "quant-risk.calibration-study/v1",
        "evidence_kind": "synthetic",
        "frequency": "daily synthetic observation",
        "lookback": 18,
        "candidates": {"baseline": baseline, "enhanced": enhanced},
        "baseline_id": "baseline",
        "exposures": exposures,
        "returns": returns,
        "outcomes": outcomes,
        "selection_requests": requests["selection"],
        "holdout_requests": requests["holdout"],
        "selection_as_of": dates[26].isoformat(),
        "plan_available_at": dates[0].isoformat(),
        "evaluation_as_of": dates[36].isoformat(),
    }
    return exposure_study, calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    exposure, calibration = fixture()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("exposures.json", exposure), ("calibration.json", calibration)):
        (args.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
