import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from quant_risk_monitor.cli import main
from quant_risk_monitor.lookthrough import evaluate_portfolio_overlap, run_portfolio_overlap
from quant_risk_monitor.risk_validation import run_risk_validation


def position(instrument, weight, *, asset_type="security", currency="CNY"):
    return {
        "instrument_id": instrument,
        "asset_type": asset_type,
        "currency": currency,
        "weight": weight,
        "source": "synthetic opening book",
    }


def disclosure(
    fund,
    instrument,
    weight,
    *,
    asset_type="security",
    currency="CNY",
    holding_date="2025-01-01",
    available_at="2025-01-02T00:00:00Z",
):
    return {
        "disclosure_id": f"{fund}-{holding_date}",
        "fund_id": fund,
        "holding_date": holding_date,
        "available_at": available_at,
        "instrument_id": instrument,
        "asset_type": asset_type,
        "currency": currency,
        "weight": weight,
        "source": "synthetic disclosure",
        "evidence_id": f"synthetic-{fund}-{holding_date}",
    }


def strategy(strategy_id, positions):
    return {
        "strategy_id": strategy_id,
        "effective_at": "2025-02-01T00:00:00Z",
        "available_at": "2025-02-01T00:00:00Z",
        "source": "synthetic strategy snapshot",
        "positions": positions,
    }


def study():
    return {
        "schema": "quant-risk.portfolio-exposure-study/v1",
        "evidence_kind": "synthetic",
        "business_at": "2025-02-01T00:00:00Z",
        "knowledge_at": "2025-02-01T00:00:00Z",
        "base_currency": "CNY",
        "position_weight_basis": "fraction_of_strategy_nav_in_base_currency",
        "max_depth": 8,
        "max_age_days": 60,
        "strategies": [
            strategy(
                "alpha",
                [
                    position("A", 0.2),
                    position("ETF", 0.6, asset_type="fund"),
                    position("CASH:CNY", 0.2, asset_type="cash"),
                ],
            ),
            strategy(
                "beta",
                [
                    position("A", 0.25),
                    position("B", 0.25),
                    position("CASH:CNY", 0.5, asset_type="cash"),
                ],
            ),
        ],
        "disclosures": [
            disclosure("ETF", "A", 0.5),
            disclosure("ETF", "B", 0.4),
            disclosure("ETF", "CASH:CNY", 0.1, asset_type="cash"),
        ],
    }


def strategy_exposures(report, index):
    return {row["instrument_id"]: row["weight"] for row in report["strategies"][index]["exposures"]}


def test_hand_calculated_direct_and_etf_cross_strategy_overlap():
    report = evaluate_portfolio_overlap(study())
    assert strategy_exposures(report, 0) == {"A": 0.5, "B": 0.24, "CASH:CNY": 0.26}
    alpha = report["strategies"][0]
    assert alpha["declared"] == {
        "net_weight": 1.0,
        "gross_weight": 1.0,
        "position_rows": 3,
        "aggregated_positions": 3,
    }
    assert alpha["coverage_gross_weight"]["covered"] == 1.0
    assert alpha["resolved"]["path_gross_weight"] == 1.0
    pair = report["pairwise"][0]
    assert pair["status"] == "available"
    assert pair["shared_known_gross_weight"] == pytest.approx(0.49)
    assert pair["known_security_gross"]["smaller_strategy_denominator"] == 0.5
    assert pair["overlap_ratio_of_smaller_gross"] == pytest.approx(0.98)
    assert {row["instrument_id"] for row in pair["common_exposures"]} == {"A", "B"}
    assert report["valuation_contract"]["certification"].startswith("this report does not")


def test_published_schemas_accept_the_example_and_report():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "examples/portfolio_overlap_study.json").read_text())
    input_schema = json.loads(
        (root / "schemas/portfolio-exposure-study-v1.schema.json").read_text()
    )
    output_schema = json.loads(
        (root / "schemas/portfolio-exposure-report-v1.schema.json").read_text()
    )
    Draft202012Validator(input_schema).validate(value)
    Draft202012Validator(output_schema).validate(evaluate_portfolio_overlap(value))


def test_duplicate_rows_and_direct_plus_etf_paths_are_aggregated_once():
    value = study()
    value["strategies"][0]["positions"] = [
        position("A", 0.1),
        position("A", 0.1),
        position("ETF", 0.3, asset_type="fund"),
        position("ETF", 0.3, asset_type="fund"),
        position("CASH:CNY", 0.2, asset_type="cash"),
    ]
    report = evaluate_portfolio_overlap(value)
    alpha = report["strategies"][0]
    assert alpha["declared"]["position_rows"] == 5
    assert alpha["declared"]["aggregated_positions"] == 3
    assert strategy_exposures(report, 0) == {"A": 0.5, "B": 0.24, "CASH:CNY": 0.26}
    assert alpha["resolved"]["path_gross_weight"] == alpha["declared"]["gross_weight"]


def test_partial_unknown_retained_and_known_only_is_not_complete_comparison():
    value = study()
    value["disclosures"] = [disclosure("ETF", "A", 0.6)]
    report = evaluate_portfolio_overlap(value)
    alpha = report["strategies"][0]
    assert alpha["coverage_gross_weight"]["covered"] == pytest.approx(0.76)
    assert alpha["coverage_gross_weight"]["unknown"] == pytest.approx(0.24)
    assert alpha["coverage_ratio"]["unknown"] == pytest.approx(0.24)
    pair = report["pairwise"][0]
    assert pair["status"] == "unavailable"
    assert pair["overlap_ratio_of_smaller_gross"] is None
    assert pair["diagnostic_known_only_ratio"] is not None


@pytest.mark.parametrize(
    "mutate,bucket",
    [
        (
            lambda value: value["disclosures"].__setitem__(
                slice(None),
                [
                    disclosure(
                        "ETF",
                        "A",
                        1,
                        available_at="2025-02-02T00:00:00Z",
                    )
                ],
            ),
            "future",
        ),
        (
            lambda value: value.update(
                disclosures=[
                    disclosure(
                        "ETF",
                        "A",
                        1,
                        holding_date="2024-01-01",
                        available_at="2024-01-02T00:00:00Z",
                    )
                ],
                max_age_days=30,
            ),
            "stale",
        ),
    ],
)
def test_future_and_stale_disclosures_are_never_used(mutate, bucket):
    value = study()
    mutate(value)
    report = evaluate_portfolio_overlap(value)
    alpha = report["strategies"][0]
    assert alpha["coverage_gross_weight"][bucket] == 0.6
    assert alpha["comparison_eligible"] is False
    assert all(row["instrument_id"] != "A" or row["weight"] == 0.2 for row in alpha["exposures"])
    if bucket == "future":
        future_path = next(
            row for row in alpha["unavailable_paths"] if row["coverage_bucket"] == "future"
        )
        assert len(future_path["source_sha256"]) == 2


def test_cycle_and_depth_limit_stop_recursion_without_double_counting():
    value = study()
    value["strategies"][0]["positions"] = [position("F", 1, asset_type="fund")]
    value["disclosures"] = [
        disclosure("F", "G", 1, asset_type="fund"),
        disclosure("G", "F", 1, asset_type="fund"),
    ]
    report = evaluate_portfolio_overlap(value)
    alpha = report["strategies"][0]
    assert alpha["coverage_gross_weight"]["cycle"] == 1
    assert alpha["coverage_gross_weight"]["denominator"] == 1
    value["max_depth"] = 1
    report = evaluate_portfolio_overlap(value)
    assert report["strategies"][0]["coverage_gross_weight"]["depth_limit"] == 1


def test_negative_weights_leverage_cash_and_internal_netting_remain_distinct():
    value = study()
    value["strategies"][0]["positions"] = [
        position("A", 1.2),
        position("ETF", -0.4, asset_type="fund"),
        position("CASH:CNY", 0.2, asset_type="cash"),
    ]
    value["disclosures"] = [
        disclosure("ETF", "A", 0.5),
        disclosure("ETF", "B", 0.5),
    ]
    report = evaluate_portfolio_overlap(value)
    alpha = report["strategies"][0]
    assert alpha["declared"]["net_weight"] == 1
    assert alpha["declared"]["gross_weight"] == 1.8
    assert alpha["resolved"]["path_gross_weight"] == 1.8
    assert alpha["resolved"]["aggregated_known_gross_weight"] == 1.4
    assert alpha["resolved"]["internal_netting_weight"] == pytest.approx(0.4)
    assert strategy_exposures(report, 0) == {"A": 1.0, "B": -0.2, "CASH:CNY": 0.2}
    common = {row["instrument_id"]: row for row in report["pairwise"][0]["common_exposures"]}
    assert common["B"]["direction"] == "opposing"


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda value: value.update(knowledge_at="2025-02-02T00:00:00Z"), "knowledge_at"),
        (
            lambda value: value["strategies"][0].update(available_at="2025-02-02T00:00:00Z"),
            "future strategy",
        ),
        (
            lambda value: value["strategies"][0]["positions"][0].update(amount=100),
            "position requires exactly",
        ),
        (
            lambda value: value["strategies"][0]["positions"][0].pop("currency"),
            "position requires exactly",
        ),
        (lambda value: value.update(position_weight_basis="shares"), "strategy NAV"),
        (
            lambda value: value["strategies"][1]["positions"].append(
                position("ETF", 0.1, asset_type="fund", currency="USD")
            ),
            "conflicting asset type or currency",
        ),
    ],
)
def test_time_denominator_fx_and_schema_gaps_fail_instead_of_filling_zero(change, match):
    value = study()
    change(value)
    with pytest.raises((ValueError, TypeError), match=match):
        evaluate_portfolio_overlap(value)


def test_cli_writes_consumer_report_and_verified_risk_summary(tmp_path, capsys):
    source = tmp_path / "exposure.json"
    source.write_text(json.dumps(study()), encoding="utf-8")
    risk_run = tmp_path / "risk"
    run_risk_validation(Path("examples/risk_forecast_study.json"), risk_run)
    output = tmp_path / "report.json"
    main(
        [
            "portfolio-overlap",
            "--input",
            str(source),
            "--out",
            str(output),
            "--risk-run",
            str(risk_run),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text())
    assert printed == {
        "status": "complete",
        "strategies": 2,
        "pairwise_available": 1,
        "pairwise_unavailable": 0,
        "risk_forecast_attached": True,
    }
    assert report["risk_forecast"]["verification_status"].startswith("verified_by_hash")
    assert report["risk_forecast"]["calibration_status"] == "not_asserted"
    assert set(report["risk_forecast"]["series"]) == {"active", "portfolio"}
    assert len(report["provenance"]["input_file_sha256"]) == 64


def test_rehashed_risk_score_tampering_blocks_overlap_output(tmp_path):
    source = tmp_path / "exposure.json"
    source.write_text(json.dumps(study()), encoding="utf-8")
    risk_run = tmp_path / "risk"
    run_risk_validation(Path("examples/risk_forecast_study.json"), risk_run)
    (risk_run / "scores.csv").write_text("fake\n1\n", encoding="utf-8")
    manifest_path = risk_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["scores.csv"] = hashlib.sha256(
        (risk_run / "scores.csv").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="scores do not reproduce"):
        run_portfolio_overlap(source, output, risk_run=risk_run)
    assert not output.exists()


def test_input_and_report_are_immutable_boundaries(tmp_path):
    source = tmp_path / "exposure.json"
    source.write_text(json.dumps(study()), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot replace"):
        run_portfolio_overlap(source, source)
    output = tmp_path / "report.json"
    run_portfolio_overlap(source, output)
    with pytest.raises(FileExistsError, match="already exists"):
        run_portfolio_overlap(source, output)
    damaged = deepcopy(study())
    damaged["strategies"][0]["positions"][0]["weight"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_portfolio_overlap(damaged)
