"""Synthetic capacity check; never evidence of market forecast accuracy."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

import model_bootstrap  # noqa: F401

# isort: split

from quant_risk_monitor.exposure_builder import build_exposure_snapshot, public_equity_style_config
from quant_risk_monitor.exposure_cache import DailyExposureStore
from quant_risk_monitor.factor_model import BarraStyleRiskModel, FactorModelDiagnostics


def one(n, root):
    rng = np.random.default_rng(20261014)
    assets = [f"S{i:05}" for i in range(n)]
    cutoff = pd.Timestamp("2026-09-30T15:00:00+08:00")
    groups = [f"G{i}" for i in range(10)]
    config = public_equity_style_config(groups, definition_version="v2", min_assets=100)
    frame = pd.DataFrame(
        {
            "asset": assets,
            "effective_at": cutoff,
            "available_at": cutoff,
            "source": "synthetic-capacity-fixture",
            "market_cap": np.exp(rng.normal(23, 1, n)),
        }
    )
    for ds in config.styles.values():
        for name in ds:
            frame[name] = rng.normal(size=n)
    frame["size_log_mcap"] = np.log(frame.market_cap)
    classifications = frame[["asset", "effective_at", "available_at", "source"]].copy()
    classifications["industry"] = [groups[i % len(groups)] for i in range(n)]
    tracemalloc.start()
    start = time.perf_counter()
    snapshot = build_exposure_snapshot(
        descriptor_history=frame,
        classification_history=classifications,
        assets=assets,
        as_of=cutoff,
        config=config,
    )
    build_seconds = time.perf_counter() - start
    factors = list(snapshot.values.columns)
    k = len(factors)
    f = rng.normal(0, 0.002, (k, k))
    f = f @ f.T
    returns = pd.DataFrame(
        rng.normal(0, 0.002, (252, k)),
        index=pd.bdate_range("2025-09-01", periods=252),
        columns=factors,
    )
    diagnostic = FactorModelDiagnostics(
        model_kind="fundamental_style",
        as_of=cutoff,
        factors=tuple(factors),
        assets=tuple(assets),
        periods=252,
        first_period_start=returns.index[0].tz_localize("Asia/Shanghai"),
        last_period_end=returns.index[-1].tz_localize("Asia/Shanghai"),
        exposure_effective_at=cutoff,
        exposure_available_at=cutoff,
        cross_section_sizes=(n,) * 252,
        regression_ranks=(k - 1,) * 252,
        condition_numbers=(1.0,) * 252,
        residual_observations=dict.fromkeys(assets, 252),
        annualization=252,
        covariance_shrinkage=0.2,
        specific_variance_shrinkage=0.2,
    )
    model = BarraStyleRiskModel(
        model_kind="fundamental_style",
        as_of=cutoff,
        annualization=252,
        exposures=snapshot.values,
        factor_returns=returns,
        factor_covariance=pd.DataFrame(f, index=factors, columns=factors),
        specific_variances=pd.Series(0.03, index=assets),
        asset_covariance=None,
        diagnostics=diagnostic,
    )
    times = []
    for _ in range(10):
        held = rng.choice(assets, size=min(n, 100), replace=False)
        w = pd.Series(1 / len(held), index=held)
        start = time.perf_counter()
        model.analyze(w)
        times.append(time.perf_counter() - start)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    store = DailyExposureStore(root / f"cache-{n}")
    cache_args = dict(
        as_of=cutoff,
        configuration={"fixture": "synthetic-v2", "n": n},
        input_digests={"fixture": "a" * 64},
        build=lambda: snapshot,
    )
    store.get_or_build(**cache_args)
    start = time.perf_counter()
    restored, hit = store.get_or_build(**cache_args)
    read_seconds = time.perf_counter() - start
    np.testing.assert_allclose(snapshot.values, restored.values)
    return {
        "assets": n,
        "factors": k,
        "exposure_build_seconds": build_seconds,
        "portfolio_query_median_seconds": float(np.median(times)),
        "peak_tracemalloc_bytes": peak,
        "cache_read_seconds": read_seconds,
        "cache_hit": hit,
        "dense_covariance_allocated": model._asset_covariance is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for n in (200, 1000, 6000):
        record = one(n, args.out)
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.out / "capacity.json").write_text(
        json.dumps(
            {
                "evidence": "synthetic, one daily cross-section; excludes network/PDF parsing/historical backfill; tracemalloc is not whole-process RSS",
                "results": records,
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
