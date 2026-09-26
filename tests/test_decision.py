from decimal import Decimal

import pytest
from quant_data_kit.exceptions import ValidationError

from quant_risk_monitor import DecisionPortfolioLimits, check_decision_portfolio


def test_healthy_target_reports_portfolio_metrics() -> None:
    result = check_decision_portfolio(
        target_weights={"000001.SZ": "0.25", "000333.SZ": "0.30"},
        current_weights={"000001.SZ": "0.20"},
        classifications={"000001.SZ": "bank", "000333.SZ": "appliance"},
        limits=DecisionPortfolioLimits(
            max_single_weight="0.35",
            max_gross_weight="0.90",
            min_cash_weight="0.10",
            max_industry_weight="0.40",
            max_turnover="0.40",
            max_positions=4,
        ),
    )

    assert not result.has_critical
    assert result.metrics == {
        "positions": 2,
        "gross_weight": 0.55,
        "cash_weight": 0.45,
        "turnover": 0.35,
        "industry_weights": {"appliance": 0.3, "bank": 0.25},
        "estimated_cost_rate": None,
    }
    assert result.to_dict()["has_critical"] is False


def test_all_enabled_portfolio_limits_fail_closed() -> None:
    result = check_decision_portfolio(
        target_weights={"BANK-A": "0.55", "BANK-B": "0.40", "UNKNOWN": "0.05"},
        current_weights={"OLD": "0.50"},
        classifications={"BANK-A": "bank", "BANK-B": "bank"},
        limits=DecisionPortfolioLimits(
            max_single_weight="0.50",
            max_gross_weight="0.90",
            min_cash_weight="0.10",
            max_industry_weight="0.60",
            max_turnover="0.80",
            max_estimated_cost_rate="0.01",
            max_positions=2,
        ),
        estimated_cost_rate="0.02",
    )

    assert result.has_critical
    assert {alert.rule_id for alert in result.alerts} == {
        "portfolio.max_positions",
        "portfolio.max_single_weight",
        "portfolio.max_gross_weight",
        "portfolio.min_cash_weight",
        "portfolio.max_turnover",
        "portfolio.missing_classification",
        "portfolio.max_industry_weight",
        "portfolio.max_estimated_cost_rate",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_positions": 0}, "positive integer"),
        ({"max_turnover": Decimal("NaN")}, "finite"),
        ({"min_cash_weight": 1.1}, "between 0 and 1"),
    ],
)
def test_limits_reject_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DecisionPortfolioLimits(**kwargs)


def test_enabled_cost_limit_rejects_missing_estimate() -> None:
    result = check_decision_portfolio(
        target_weights={"A": "0.5"},
        limits=DecisionPortfolioLimits(max_estimated_cost_rate="0.001"),
    )

    assert result.has_critical
    assert [alert.rule_id for alert in result.alerts] == ["portfolio.missing_estimated_cost_rate"]
    assert result.metrics["estimated_cost_rate"] is None
