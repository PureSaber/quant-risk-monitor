"""Causal public-data panel and daily factor-return history for the frozen study."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from model_bootstrap import (
    ExposureSnapshot,
    FactorModelConfig,
    build_exposure_snapshot,
    public_equity_style_config,
    solve_factor_returns,
)
from public_sources import ROOT, fetch


def read_prices():
    cohort = json.loads((ROOT / "cohort.json").read_text(encoding="utf-8"))["members"]
    frames = []
    for member in cohort:
        code = member["code"]
        content = json.loads((ROOT / "raw/prices" / (code + "-0.json")).read_text(encoding="utf-8"))
        data = content.get("data")
        if not data or not data.get("klines"):
            raise ValueError(f"Missing historical prices for preselected {code}")
        rows = [line.split(",") for line in data["klines"]]
        raw = pd.DataFrame(
            rows,
            columns=[
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "change",
                "turnover",
            ],
        )
        raw["date"] = pd.to_datetime(raw["date"])
        if raw.date.duplicated().any():
            raise ValueError("Duplicate daily price")
        raw = raw.set_index("date").apply(pd.to_numeric, errors="coerce")
        path = ROOT / "raw/bao-prices" / (code + ".csv")
        receipt = json.loads(path.with_suffix(".receipt.json").read_text(encoding="utf8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Adjusted price source changed")
        adjusted = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if adjusted.index.duplicated().any():
            raise ValueError("Duplicate adjusted daily price")
        if not raw.index.isin(adjusted.index).all():
            raise ValueError(f"Missing independent adjusted history for {code}")
        # EM's additive HFQ levels distort percentage returns. Use BaoStock's
        # multiplicative adjustment; preserve raw prices for capitalization.
        closes = adjusted.close.where(adjusted.tradestatus.eq(1)).reindex(raw.index)
        frames.append(raw.assign(adjusted_close=closes, asset=code).reset_index())
    long = pd.concat(frames, ignore_index=True)
    dates = pd.DatetimeIndex(sorted(long.date.unique()))
    result = {
        name: long.pivot(index="date", columns="asset", values=name).reindex(dates)
        for name in ("close", "adjusted_close", "volume", "turnover")
    }
    # No automatic zero return for a missing quote. Stale marks are evaluated separately.
    result["returns"] = result["adjusted_close"].pct_change(fill_method=None)
    return result


def calendar_availability(value):
    return pd.Timestamp(value).tz_convert("Asia/Shanghai").tz_localize(None)


def financial_daily(dates, assets):
    facts = pd.read_csv(
        ROOT / "clean/financial_facts.csv", dtype={"asset": str, "report_period": str}
    )
    facts["available_at"] = (
        pd.to_datetime(facts.available_at, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    matrices = {
        k: pd.DataFrame(np.nan, index=dates, columns=assets)
        for k in (
            "book_equity",
            "assets",
            "liabilities",
            "shares",
            "profit_ttm",
            "growth",
        )
    }
    for asset in assets:
        events = facts[facts.asset == asset].sort_values(["available_at", "report_period", "field"])
        rows = list(events.itertuples())
        cursor = 0
        known = defaultdict(dict)
        updated = {}
        for day in dates:
            while cursor < len(rows) and rows[cursor].available_at <= day:
                row = rows[cursor]
                known[row.field][row.report_period] = row.value
                updated[(row.field, row.report_period)] = row.available_at
                cursor += 1
            for field in ("book_equity", "assets", "liabilities", "shares"):
                if known[field]:
                    p = max(known[field])
                    if (day - pd.Timestamp(p)).days <= 550:
                        matrices[field].at[day, asset] = known[field][p]
            profits = known["profit_ytd"]
            if profits:
                p = max(profits)
                year = int(p[:4])
                suffix = p[4:]
                if (day - pd.Timestamp(p)).days <= 550:
                    prior = f"{year - 1}{suffix}"
                    if prior in profits and abs(profits[prior]) > 1:
                        matrices["growth"].at[day, asset] = (profits[p] - profits[prior]) / abs(
                            profits[prior]
                        )
                    if suffix == "1231":
                        matrices["profit_ttm"].at[day, asset] = profits[p]
                    elif prior in profits and f"{year - 1}1231" in profits:
                        matrices["profit_ttm"].at[day, asset] = (
                            profits[p] + profits[f"{year - 1}1231"] - profits[prior]
                        )
    return matrices


def capital_daily(dates, assets, financial_shares):
    result = financial_shares.copy()
    for asset in assets:
        path = ROOT / "raw/shares" / (asset + ".csv")
        if not path.exists():
            continue
        rows = pd.read_csv(path, dtype={"证券代码": str})
        rows["ann"] = pd.to_datetime(rows["公告日期"], errors="coerce") + pd.Timedelta(days=1)
        rows["effective"] = pd.to_datetime(rows["变动日期"], errors="coerce")
        rows["shares"] = pd.to_numeric(rows["总股本"], errors="coerce") * 10000
        rows = rows.dropna(subset=["ann", "effective", "shares"])
        rows = rows[rows.ann.dt.year >= 1990]
        rows = rows[rows.shares > 0]
        # Sort by when both conditions first hold; a late older report must not
        # displace a more recent effective capital change.
        rows["known_at"] = rows[["ann", "effective"]].max(axis=1)
        events = rows.sort_values(["known_at", "effective", "ann"])
        current_key = None
        changes = {}
        for event in events.itertuples():
            key = (event.effective, event.ann)
            if current_key is None or key >= current_key:
                current_key = key
                changes[event.known_at] = event.shares
        if changes:
            s = pd.Series(changes).sort_index()
            s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
            result[asset] = s.combine_first(result[asset])
    return result


def industry_daily(dates, assets, mapping):
    rows = pd.read_csv(ROOT / "clean/industry_history.csv", dtype={"asset": str})
    rows["available_at"] = (
        pd.to_datetime(rows.available_at, utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    result = pd.DataFrame(None, index=dates, columns=assets, dtype=object)
    for asset in assets:
        events = rows[rows.asset == asset].sort_values("available_at")
        for event in events.itertuples():
            result.loc[result.index >= event.available_at, asset] = mapping[event.sector]
    return result


def benchmark():
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "0",
        "secid": "1.000300",
        "beg": "20190101",
        "end": "20251231",
    }
    data = json.loads(
        fetch(
            "benchmark/csi300.json",
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params=params,
        )
    )["data"]
    return (
        pd.Series({pd.Timestamp(r.split(",")[0]): float(r.split(",")[2]) for r in data["klines"]})
        .sort_index()
        .pct_change(fill_method=None)
    )


def descriptors_from_history(px, financial, shares, market):
    dates = px["close"].index
    assets = px["close"].columns
    caps = px["close"].shift(1) * shares
    logcaps = np.log(caps.where(caps > 0))
    beta = pd.DataFrame(index=dates, columns=assets, dtype=float)
    residual_vol = beta.copy()
    for asset in assets:
        r = px["returns"][asset]
        b = r.rolling(252, min_periods=126).cov(market) / market.rolling(252, min_periods=126).var()
        beta[asset] = b.shift(1)
        residual_var = (
            r.rolling(252, min_periods=126).var()
            - 2 * b * r.rolling(252, min_periods=126).cov(market)
            + b * b * market.rolling(252, min_periods=126).var()
        )
        residual_vol[asset] = np.sqrt(residual_var.clip(lower=0)).shift(1)
    return caps, {
        "size_log_mcap": logcaps,
        "nonlinear_size": logcaps**3,
        "beta_252d": beta,
        "momentum_252_21": np.log(px["adjusted_close"].shift(22) / px["adjusted_close"].shift(253)),
        "residual_vol_252d": residual_vol,
        "liquidity_log_turnover_20d": np.log(
            px["turnover"].rolling(20, min_periods=15).mean().shift(1).where(lambda x: x > 0) / 100
        ),
        "book_to_price": financial["book_equity"] / caps,
        "earnings_yield_ttm": financial["profit_ttm"] / caps,
        "earnings_growth_yoy": financial["growth"],
        "book_leverage": financial["liabilities"] / financial["book_equity"].where(lambda x: x > 0),
    }


def prepare():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    px = read_prices()
    dates = px["close"].index
    assets = list(px["close"].columns)
    market = benchmark().reindex(dates)
    financial = financial_daily(dates, assets)
    shares = capital_daily(dates, assets, financial["shares"])
    groups = industry_daily(dates, assets, protocol["sector_mapping"])
    caps, descriptors = descriptors_from_history(px, financial, shares, market)
    output = ROOT / "clean"
    output.mkdir(exist_ok=True)
    for name, frame in {
        **px,
        "market_cap": caps,
        "industry": groups,
        **descriptors,
    }.items():
        frame.to_parquet(output / (name + ".parquet"))
    panel = {
        "dates": dates,
        "assets": assets,
        "prices": px,
        "caps": caps,
        "groups": groups,
        "descriptors": descriptors,
    }
    with (output / "panel.pkl").open("wb") as f:
        pickle.dump(panel, f, protocol=5)
    print(
        json.dumps(
            {
                "dates": len(dates),
                "assets": len(assets),
                "range": [str(dates[0]), str(dates[-1])],
            }
        ),
        flush=True,
    )


def snapshot(panel, day, config):
    caps = panel["caps"].loc[day]
    valid = caps[caps.gt(0) & np.isfinite(caps) & panel["groups"].loc[day].notna()].index
    # Eligibility uses only the price observation preceding this cutoff.
    values = pd.DataFrame(
        {name: frame.loc[day, valid] for name, frame in panel["descriptors"].items()}
    )
    finite = values.replace([np.inf, -np.inf], np.nan)
    # Price-derived descriptors require their own observed trailing histories.
    valid = finite.dropna(
        subset=[
            "size_log_mcap",
            "beta_252d",
            "momentum_252_21",
            "residual_vol_252d",
            "liquidity_log_turnover_20d",
        ]
    ).index
    finite = finite.loc[valid]
    cutoff = pd.Timestamp(day).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    frame = finite.assign(
        asset=valid,
        market_cap=caps.loc[valid],
        effective_at=cutoff,
        available_at=cutoff,
        source="verified public PIT panel",
    ).reset_index(drop=True)
    classification = pd.DataFrame(
        {
            "asset": valid,
            "industry": panel["groups"].loc[day, valid].to_numpy(),
            "effective_at": cutoff,
            "available_at": cutoff,
            "source": "dated CSRC/CAPCO original releases",
        }
    )
    return build_exposure_snapshot(
        descriptor_history=frame,
        classification_history=classification,
        assets=list(valid),
        as_of=cutoff,
        config=config,
    )


def regress():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    with (ROOT / "clean/panel.pkl").open("rb") as f:
        panel = pickle.load(f)
    config = public_equity_style_config(
        sorted(set(protocol["sector_mapping"].values())),
        missing_policy="industry_median",
        missing_group_fallback="global",
        max_missing_fraction=0.4,
        min_assets=100,
    )
    snapshots = {}
    failures = []
    dates = panel["dates"]
    for day in dates[dates >= "2021-11-11"]:
        try:
            snapshots[day] = snapshot(panel, day, config)
        except ValueError as exc:
            failures.append({"date": str(day.date()), "stage": "exposure", "error": str(exc)})
    histories = {
        name: {"factors": [], "residuals": [], "dates": []}
        for name in ("equal", "sqrt_market_cap", "legacy_ols")
    }
    for i, day in enumerate(dates):
        if i == 0 or dates[i - 1] not in snapshots:
            continue
        prior = snapshots[dates[i - 1]]
        r = panel["prices"]["returns"].loc[day].dropna()
        assets = prior.values.index.intersection(r.index)
        if len(assets) < 100:
            continue
        selected = ExposureSnapshot(
            effective_at=prior.effective_at,
            available_at=prior.available_at,
            values=prior.values.loc[assets],
            source=prior.source,
            market_caps=prior.market_caps.loc[assets],
            industries=prior.industries.loc[assets],
            industry_factors=prior.industry_factors,
        )
        for method in histories:
            chosen = selected
            if method == "legacy_ols":
                # Original full-rank OLS uses industry intercepts without a
                # redundant market column; no identifying constraint is added.
                chosen = ExposureSnapshot(
                    effective_at=selected.effective_at,
                    available_at=selected.available_at,
                    values=selected.values.drop(columns="market"),
                    source=selected.source,
                    market_caps=selected.market_caps,
                    industries=selected.industries,
                    industry_factors=selected.industry_factors,
                )
            cfg = FactorModelConfig(
                model_kind="fundamental_style",
                regression_weighting="equal" if method == "legacy_ols" else method,
                industry_constraint=method != "legacy_ols",
            )
            try:
                factor, residual, rank, condition, diagnostic = solve_factor_returns(
                    chosen, r.loc[assets], cfg
                )
            except ValueError as exc:
                failures.append({"date": str(day.date()), "stage": method, "error": str(exc)})
                continue
            histories[method]["factors"].append(pd.Series(factor, index=chosen.values.columns))
            histories[method]["residuals"].append(pd.Series(residual, index=assets))
            histories[method]["dates"].append(day)
    for method, data in histories.items():
        data["factors"] = pd.DataFrame(data["factors"], index=pd.DatetimeIndex(data["dates"]))
        data["residuals"] = pd.DataFrame(
            data["residuals"], index=pd.DatetimeIndex(data["dates"])
        ).reindex(columns=panel["assets"])
        data["factors"].to_parquet(ROOT / "clean" / (method + "-factors.parquet"))
        data["residuals"].to_parquet(ROOT / "clean" / (method + "-residuals.parquet"))
    with (ROOT / "clean/regressions.pkl").open("wb") as f:
        pickle.dump({"snapshots": snapshots, "histories": histories}, f, protocol=5)
    (ROOT / "clean/regression-failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "snapshots": len(snapshots),
                "regression_dates": {k: len(v["dates"]) for k, v in histories.items()},
                "failures": len(failures),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"prepare": prepare, "regress": regress}[sys.argv[1]]()
