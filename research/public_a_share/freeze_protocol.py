"""Freeze candidates and evaluation rules before model selection / holdout scoring."""

import hashlib
import json
from datetime import datetime, timezone

from public_sources import ROOT

if (ROOT / "protocol.json").exists():
    raise ValueError("Protocol already frozen; preserve the original design")
protocol = {
    "schema": "public-a-share-risk-study/v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "design": "retrospective historical reconstruction; no claim of historical preregistration",
    "excluded_sources": ["Huisheng", "Dataway"],
    "cohort_sha256": hashlib.sha256((ROOT / "cohort.json").read_bytes()).hexdigest(),
    "selection_year": 2023,
    "holdout_years": [2024, 2025],
    "lookback": 252,
    "horizon": 5,
    "prediction_time": "15:00 Asia/Shanghai; daily price/return observations are conservatively usable from the following calendar day; filings from the day after their dated public disclosure",
    "financial_values": "labeled amounts verified against original same-period CNINFO filings; unmatched fields are missing, never backdated from a current vendor table",
    "exposure_missing": "industry median, global median only with too few observed industry peers; max 40% missing per descriptor; no all-missing descriptor",
    "universe": "200 historical names frozen at 2021-11-10; forecast eligibility depends only on preceding data; include subsequent delistings; no hindsight survivor filter",
    "identification": "same cap-neutral industry identifying constraint in OLS and WLS, leaving OLS objective equal-weighted",
    "sector_mapping": {
        "A": "consumer",
        "B": "materials",
        "C": "manufacturing",
        "D": "utilities",
        "E": "industrials",
        "F": "consumer",
        "G": "transport",
        "H": "consumer",
        "I": "technology",
        "J": "financials",
        "K": "real_estate",
        "L": "services",
        "M": "services",
        "N": "utilities",
        "P": "services",
        "Q": "services",
        "R": "services",
        "S": "services",
    },
    "baseline": {
        "regression_weighting": "equal",
        "industry_constraint": True,
        "covariance_shrinkage": 0.2,
        "specific_variance_shrinkage": 0.2,
    },
    "candidate_grid": {
        "covariance_shrinkage": [0.1, 0.2, 0.4],
        "half_life_pairs": [[90, 60], [180, 90]],
        "specific_prior_dof": [10, 30],
    },
    "enhanced_fixed": {
        "regression_weighting": "sqrt_market_cap",
        "industry_constraint": True,
        "newey_west_lags": 4,
        "specific_newey_west_lags": 4,
        "regime_window": 20,
        "regime_min_periods": 60,
        "regime_half_life": 10,
        "specific_target": "industry",
        "specific_shrinkage_method": "bayesian",
        "specific_group_fallback": "global",
    },
    "score": "log(predicted variance)+squared five-session buy-and-hold return/predicted variance; mean within date, equal total weight on equal-weight, capitalization-weight and industry-portfolio families",
    "date_grid": "every fifth observed exchange session; discard any horizon crossing selection/holdout boundaries; at least five sessions separate final selection horizon from first holdout forecast",
    "evaluation_missing": "never rescale surviving holdings; publish missing-endpoint counts, common complete cases and a separate stale-mark sensitivity; delisting events retained and inspected",
    "minimum_forecast_assets": 100,
    "minimum_selection_dates": 40,
    "minimum_holdout_dates": 80,
    "bootstrap": {
        "unit": "forecast date, preserving all portfolios",
        "moving_block_length": 10,
        "replications": 2000,
        "seed": 20261013,
    },
    "promotion": "paired holdout QLIKE delta 95% interval entirely below zero; no unresolved data defects or material volatility bias deterioration; otherwise retain baseline",
    "production_parameter_changes": False,
}
target = ROOT / "protocol.json"
if target.exists():
    raise SystemExit(
        "Protocol already frozen; preserve the original and version any changes explicitly"
    )
target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print(hashlib.sha256(target.read_bytes()).hexdigest())
