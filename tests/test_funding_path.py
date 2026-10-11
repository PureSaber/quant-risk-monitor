import copy
import json
from decimal import Decimal as D

import pytest

from quant_risk_monitor.funding_path import evaluate_funding_path, evaluate_joint_funding, main


def mark(name, value="100", initial="0", maintenance="0", cash="0", capacity="1", lag=0):
    return {
        "id": name,
        "mark_value": value,
        "eligible_collateral": "0",
        "initial_margin": initial,
        "maintenance_margin": maintenance,
        "cash_pnl": cash,
        "liquidation_notional": "100",
        "liquidation_cost_rate": "0",
        "max_liquidation_fraction": capacity,
        "settlement_lag_steps": lag,
    }


def funding():
    return {
        "schema": "quant-risk-monitor.funding-path/v1",
        "base_currency": "USD",
        "collateral_pool": "broker-one",
        "as_of": "2025-01-01T00:00:00Z",
        "snapshot_known_at": "2025-01-01T00:00:00Z",
        "source_ref": "synthetic:external-native-valuation",
        "opening_cash": "30",
        "cash_reserve": "5",
        "liquidation_order": ["future", "equity"],
        "positions": [
            {"id": "future", "opening_mark_value": "0"},
            {"id": "equity", "opening_mark_value": "100"},
        ],
        "steps": [
            {
                "time": "2025-01-02T00:00:00Z",
                "external_cash_flow": "0",
                "financing_cash_flow": "0",
                "positions": [
                    mark("future", "0", "40", "30", "-20", ".25"),
                    mark("equity", "80", capacity=".10"),
                ],
            },
            {
                "time": "2025-01-03T00:00:00Z",
                "external_cash_flow": "0",
                "financing_cash_flow": "-1",
                "positions": [mark("future", "0", "40", "30", "-8"), mark("equity", "70")],
            },
        ],
    }


def test_path_margin_is_encumbrance_and_independent_conservation():
    result = evaluate_funding_path(funding())
    a, b = result["steps"]
    # 30 - 20 = 10 cash. Close .25 futures -> margin 30; sell .1 equity -> cash 18.
    assert a["remaining_fractions"] == {"future": "0.75", "equity": "0.90"}
    assert D(a["after"]["cash"]) == 18
    assert D(a["after"]["initial_margin"]) == 30
    assert D(a["after"]["maintenance_margin"]) == D("22.5")
    assert D(a["after"]["maintenance_gap"]) == D("9.5")
    assert a["status"] == "funding_gap"
    assert D(a["equity"]) == 18 + D(".9") * 80 == 90
    # Day 2 VM only on remaining .75 futures: -6. Equity loss .9*(-10)=-9; fee -1.
    assert D(b["equity"]) == 90 - 6 - 9 - 1 == 74
    assert D(b["cash_pnl"]) == -6
    assert D(b["mark_pnl"]) == -9
    assert b["status"] == "funded"
    assert all(abs(D(row["conservation_residual"])) < D("1e-28") for row in result["steps"])


def test_liquidation_order_settlement_and_collateral_are_material():
    spec = funding()
    reversed_order = copy.deepcopy(spec)
    reversed_order["liquidation_order"].reverse()
    assert evaluate_funding_path(reversed_order)["steps"][0]["liquidations"][0]["id"] == "equity"
    delayed = copy.deepcopy(spec)
    delayed["steps"][0]["positions"][1]["settlement_lag_steps"] = 1
    delayed["steps"][0]["positions"][0]["cash_pnl"] = "-35"
    report = evaluate_funding_path(delayed)
    assert report["steps"][0]["after"]["cash"] == "-5.00"
    assert D(report["steps"][0]["pending_settlement_value"]) == 8
    assert D(report["steps"][1]["settled_cash"]) == 8
    assert report["steps"][0]["status"] == "funding_gap"
    assert report["unsettled"] == []
    # Collateral relief must subtract the collateral that a sale removes.
    collateral = funding()
    collateral["steps"][0]["positions"][1]["eligible_collateral"] = "40"
    assert not evaluate_funding_path(collateral)["steps"][0]["triggered"]


def test_cost_fee_external_flows_pending_and_bankruptcy():
    spec = funding()
    spec["liquidation_order"] = ["equity", "future"]
    spec["steps"][0]["positions"][1]["liquidation_cost_rate"] = ".01"
    first = evaluate_funding_path(spec)["steps"][0]
    assert D(first["liquidation_cost"]) == D(".10")
    assert D(first["equity"]) == D("89.90")
    spec = funding()
    spec["steps"][0]["positions"][0]["cash_pnl"] = "-200"
    spec["steps"][0]["positions"][1]["settlement_lag_steps"] = 5
    spec["steps"][0]["external_cash_flow"] = "10"
    result = evaluate_funding_path(spec)
    assert result["steps"][0]["status"] == "insolvent"
    assert result["unsettled"][0]["due"] == 5


def test_joint_pools_do_not_net_unavailable_collateral():
    a, b = funding(), funding()
    b.update(collateral_pool="broker-two", opening_cash="1000")
    spec = {"schema": "quant-risk-monitor.joint-funding/v1", "pools": [a, b]}
    out = evaluate_joint_funding(spec)
    assert D(out["aggregate"][0]["required_external_cash"]) == D("9.5")
    assert out["aggregate"][0]["pool_statuses"]["broker-two"] == "funded"
    for bad in [dict(spec, schema="bad"), dict(spec, pools=[]), dict(spec, pools=[a, a])]:
        with pytest.raises(ValueError):
            evaluate_joint_funding(bad)
    b["steps"][0]["time"] = "2025-01-02T01:00:00Z"
    with pytest.raises(ValueError, match="times"):
        evaluate_joint_funding(spec)
    b["steps"][0]["time"] = a["steps"][0]["time"]
    b["base_currency"] = "EUR"
    with pytest.raises(ValueError, match="currency"):
        evaluate_joint_funding(spec)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(schema="wrong"),
        lambda s: s.update(snapshot_known_at="2026-01-01T00:00:00Z"),
        lambda s: s.update(as_of="2025-01-01"),
        lambda s: s.update(source_ref=""),
        lambda s: s.update(opening_cash="NaN"),
        lambda s: s.update(opening_cash="bad"),
        lambda s: s.update(opening_cash=30),
        lambda s: s.update(opening_cash="-500"),
        lambda s: s.update(cash_reserve="-1"),
        lambda s: s.update(liquidation_order=["equity"]),
        lambda s: s["positions"].append(s["positions"][0]),
        lambda s: s["steps"][0].update(time="2025-01-01T00:00:00Z"),
        lambda s: s["steps"][0]["positions"].pop(),
        lambda s: s["steps"][0]["positions"][0].update(eligible_collateral="1"),
        lambda s: s["steps"][0]["positions"][0].update(maintenance_margin="50"),
        lambda s: s["steps"][0]["positions"][0].update(max_liquidation_fraction="1.01"),
        lambda s: s["steps"][0]["positions"][0].update(settlement_lag_steps=True),
        lambda s: s.update(steps=[]),
    ],
)
def test_reject_invalid(mutation):
    spec = funding()
    mutation(spec)
    with pytest.raises((ValueError, TypeError)):
        evaluate_funding_path(spec)


def test_skip_zero_capacity_and_initial_buffer_and_cli(tmp_path):
    spec = funding()
    spec["steps"][0]["positions"][0]["max_liquidation_fraction"] = "0"
    assert evaluate_funding_path(spec)["steps"][0]["liquidations"][0]["id"] == "equity"
    spec = funding()
    spec["steps"][0]["positions"][0]["cash_pnl"] = "10"
    assert evaluate_funding_path(spec)["steps"][0]["status"] == "below_initial_buffer"
    source, target = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(funding()), encoding="utf-8")
    main(["--input", str(source), "--out", str(target)])
    assert json.loads(target.read_text())["input_sha256"]
    with pytest.raises(FileExistsError):
        main(["--input", str(source), "--out", str(target)])
