"""Run a deterministic four-asset Barra-style statistical proxy example."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_risk_monitor import (
    AssetReturnObservation,
    ExposureSnapshot,
    FactorModelConfig,
    fit_barra_style_risk_model,
)

exposures = pd.DataFrame(
    {
        "market": [1.0, 1.0, 1.0, 1.0],
        "momentum_proxy": [-1.0, -0.2, 0.3, 1.2],
    },
    index=["ETF-A", "ETF-B", "ETF-C", "ETF-D"],
)
snapshots = [
    ExposureSnapshot(
        effective_at="2025-01-01T00:00:00Z",
        available_at="2025-01-01T00:00:00Z",
        values=exposures,
        source="documented-statistical-proxy",
    )
]
rng = np.random.default_rng(7)
returns = []
for offset in range(25):
    start = pd.Timestamp("2025-01-02T00:00:00Z") + pd.Timedelta(days=offset)
    end = start + pd.Timedelta(days=1)
    factor_return = np.array([0.001 * (-1) ** offset, 0.0002 * (offset - 12)])
    realized = exposures.to_numpy() @ factor_return + rng.normal(0, 0.0005, 4)
    returns.append(
        AssetReturnObservation(
            period_start=start,
            period_end=end,
            available_at=end,
            values=pd.Series(realized, index=exposures.index),
            source="matured-close-to-close",
        )
    )

model = fit_barra_style_risk_model(
    exposure_snapshots=snapshots,
    return_observations=returns,
    as_of="2025-02-01T00:00:00Z",
    config=FactorModelConfig(
        model_kind="statistical_proxy",
        min_periods=20,
        min_assets_per_period=4,
        min_asset_observations=20,
        annualization=252,
    ),
)
report = model.analyze(
    {"ETF-A": 0.5, "ETF-B": 0.5},
    benchmark_weights={"ETF-A": 0.25, "ETF-B": 0.25, "ETF-C": 0.25, "ETF-D": 0.25},
)
print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
