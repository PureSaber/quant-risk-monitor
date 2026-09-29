"""Run a six-asset equity-style risk example. Parameters are not a vendor calibration."""

from __future__ import annotations

import json

import pandas as pd

from quant_risk_monitor import (
    AssetReturnObservation,
    EquityRiskConfig,
    IndustrySnapshot,
    MarketCapSnapshot,
    StyleExposureSnapshot,
    attribute_realized_return,
    fit_equity_style_risk_model,
)

assets = ["A1", "A2", "A3", "B1", "B2", "B3"]
when = "2024-01-01T00:00:00Z"
styles = [
    StyleExposureSnapshot(
        effective_at=when,
        available_at=when,
        values=pd.DataFrame({"value_style": [1.0, 0.2, -0.4, 0.5, -0.7, 0.1]}, index=assets),
        source="example-style",
    )
]
industries = [
    IndustrySnapshot(
        effective_at=when,
        available_at=when,
        labels=pd.Series(["Banks", "Banks", "Banks", "Energy", "Energy", "Energy"], index=assets),
        source="example-industry",
    )
]
caps = [
    MarketCapSnapshot(
        effective_at=when,
        available_at=when,
        values=pd.Series([10.0, 10.0, 10.0, 12.0, 12.0, 12.0], index=assets),
        source="example-cap",
    )
]
observations = []
for day in range(24):
    start = pd.Timestamp("2024-01-02T00:00:00Z") + pd.Timedelta(days=day)
    end = start + pd.Timedelta(days=1)
    sign = (-1) ** day
    values = []
    for asset, style, industry_sign in zip(
        assets,
        styles[0].values["value_style"],
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        strict=True,
    ):
        values.append(0.001 * sign + 0.002 * industry_sign * sign + 0.0004 * float(style))
    observations.append(
        AssetReturnObservation(
            period_start=start,
            period_end=end,
            available_at=end,
            values=pd.Series(values, index=assets),
            source="example-return",
        )
    )

model = fit_equity_style_risk_model(
    style_snapshots=styles,
    industry_snapshots=industries,
    market_cap_snapshots=caps,
    return_observations=observations,
    as_of="2024-01-27T00:00:00Z",
    config=EquityRiskConfig(
        model_kind="statistical_proxy",
        min_periods=20,
        min_assets_per_period=6,
        newey_west_lags=2,
        regime_lookback=8,
    ),
)
latest = model.factor_returns.iloc[-1]
industry_cap = {"Banks": 30.0, "Energy": 36.0}
constrained = sum(latest[name] * weight for name, weight in industry_cap.items()) / sum(
    industry_cap.values()
)
realized = pd.Series(0.01, index=model.exposures.index)
attribution = attribute_realized_return(model, realized, caps[0].values)
print(
    json.dumps(
        {
            "vendor_model": model.to_dict()["vendor_model"],
            "parameters_are_vendor_calibration": False,
            "factors": list(model.exposures.columns),
            "industry_cap_weighted_factor_return": constrained,
            "annualized_volatility": model.analyze(
                {asset: 1.0 / len(assets) for asset in assets}
            ).portfolio_risk.volatility,
            "attribution_residual_abs_sum": float(attribution.specific_return.abs().sum()),
        },
        indent=2,
    )
)
