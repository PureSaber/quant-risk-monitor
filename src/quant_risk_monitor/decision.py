"""Deterministic portfolio checks for target-weight decision workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from quant_data_kit.exceptions import ValidationError

from quant_risk_monitor.models import Alert, CheckResult, Severity


def _decimal(value: object, name: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValidationError(f"{name} must be decimal-compatible") from exc
    if not parsed.is_finite():
        raise ValidationError(f"{name} must be finite")
    return parsed


def _optional_ratio(
    value: object | None, name: str, *, maximum: Decimal = Decimal(1)
) -> Decimal | None:
    if value is None:
        return None
    parsed = _decimal(value, name)
    if not Decimal(0) <= parsed <= maximum:
        raise ValidationError(f"{name} must be between 0 and {maximum}")
    return parsed


def _optional_count(value: object | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or parsed != _decimal(value, name):
        raise ValidationError(f"{name} must be a positive integer")
    return parsed


def _weights(values: Mapping[str, object], name: str) -> dict[str, Decimal]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    parsed: dict[str, Decimal] = {}
    for symbol, value in values.items():
        clean = str(symbol).strip()
        if not clean:
            raise ValidationError(f"{name} contains an empty symbol")
        if clean in parsed:
            raise ValidationError(f"{name} contains duplicate normalized symbols")
        weight = _decimal(value, f"{name}[{clean!r}]")
        if weight < 0:
            raise ValidationError(f"{name}[{clean!r}] must be non-negative")
        parsed[clean] = weight
    return parsed


@dataclass(frozen=True, kw_only=True, slots=True)
class DecisionPortfolioLimits:
    """Long-only portfolio limits expressed as fractions of net asset value."""

    max_single_weight: Decimal | str | float | None = None
    max_gross_weight: Decimal | str | float | None = None
    min_cash_weight: Decimal | str | float | None = None
    max_industry_weight: Decimal | str | float | None = None
    max_turnover: Decimal | str | float | None = None
    max_estimated_cost_rate: Decimal | str | float | None = None
    max_positions: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_single_weight",
            "max_gross_weight",
            "min_cash_weight",
            "max_industry_weight",
            "max_estimated_cost_rate",
        ):
            object.__setattr__(self, name, _optional_ratio(getattr(self, name), name))
        # Turnover is sum(abs(target-current)), including both legs of a rotation.
        object.__setattr__(
            self,
            "max_turnover",
            _optional_ratio(self.max_turnover, "max_turnover", maximum=Decimal(2)),
        )
        object.__setattr__(
            self, "max_positions", _optional_count(self.max_positions, "max_positions")
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> DecisionPortfolioLimits:
        if not isinstance(values, Mapping):
            raise ValidationError("portfolio limits must be a mapping")
        supported = {
            "max_single_weight",
            "max_gross_weight",
            "min_cash_weight",
            "max_industry_weight",
            "max_turnover",
            "max_estimated_cost_rate",
            "max_positions",
        }
        return cls(**{key: values[key] for key in supported if key in values})


def check_decision_portfolio(
    *,
    target_weights: Mapping[str, object],
    current_weights: Mapping[str, object] | None = None,
    classifications: Mapping[str, str] | None = None,
    limits: DecisionPortfolioLimits,
    estimated_cost_rate: Decimal | str | float | None = None,
) -> CheckResult:
    """Check a proposed long-only target before any order intent is emitted."""

    if not isinstance(limits, DecisionPortfolioLimits):
        raise ValidationError("limits must be DecisionPortfolioLimits")
    target = _weights(target_weights, "target_weights")
    current = _weights(current_weights or {}, "current_weights")
    industries = dict(classifications or {})
    alerts: list[Alert] = []
    gross = sum(target.values(), Decimal(0))
    cash = Decimal(1) - gross
    turnover = sum(
        abs(target.get(symbol, Decimal(0)) - current.get(symbol, Decimal(0)))
        for symbol in set(target) | set(current)
    )
    active = {symbol: weight for symbol, weight in target.items() if weight > 0}
    industry_weights: dict[str, Decimal] = defaultdict(Decimal)
    missing_classifications: list[str] = []
    for symbol, weight in active.items():
        industry = str(industries.get(symbol, "")).strip()
        if industry:
            industry_weights[industry] += weight
        elif limits.max_industry_weight is not None:
            missing_classifications.append(symbol)

    def breach(rule_id: str, message: str, **details: object) -> None:
        alerts.append(Alert(rule_id, Severity.CRITICAL, message, details))

    if limits.max_positions is not None and len(active) > limits.max_positions:
        breach(
            "portfolio.max_positions",
            "target portfolio has too many positions",
            actual=len(active),
            limit=limits.max_positions,
        )
    if limits.max_single_weight is not None:
        for symbol, weight in sorted(active.items()):
            if weight > limits.max_single_weight:
                breach(
                    "portfolio.max_single_weight",
                    f"{symbol} exceeds the single-position limit",
                    symbol=symbol,
                    actual=float(weight),
                    limit=float(limits.max_single_weight),
                )
    if limits.max_gross_weight is not None and gross > limits.max_gross_weight:
        breach(
            "portfolio.max_gross_weight",
            "target invested weight exceeds the portfolio limit",
            actual=float(gross),
            limit=float(limits.max_gross_weight),
        )
    if limits.min_cash_weight is not None and cash < limits.min_cash_weight:
        breach(
            "portfolio.min_cash_weight",
            "target cash reserve is below the minimum",
            actual=float(cash),
            limit=float(limits.min_cash_weight),
        )
    if limits.max_turnover is not None and turnover > limits.max_turnover:
        breach(
            "portfolio.max_turnover",
            "proposed traded value exceeds the turnover limit",
            actual=float(turnover),
            limit=float(limits.max_turnover),
        )
    if missing_classifications:
        breach(
            "portfolio.missing_classification",
            "industry classification is required for every target position",
            symbols=sorted(missing_classifications),
        )
    if limits.max_industry_weight is not None:
        for industry, weight in sorted(industry_weights.items()):
            if weight > limits.max_industry_weight:
                breach(
                    "portfolio.max_industry_weight",
                    f"{industry} exceeds the industry concentration limit",
                    industry=industry,
                    actual=float(weight),
                    limit=float(limits.max_industry_weight),
                )
    cost = None
    if limits.max_estimated_cost_rate is not None and estimated_cost_rate is None:
        breach(
            "portfolio.missing_estimated_cost_rate",
            "estimated trading cost is required when the cost limit is enabled",
            limit=float(limits.max_estimated_cost_rate),
        )
    elif estimated_cost_rate is not None:
        cost = _decimal(estimated_cost_rate, "estimated_cost_rate")
        if cost < 0:
            raise ValidationError("estimated_cost_rate must be non-negative")
        if limits.max_estimated_cost_rate is not None and cost > limits.max_estimated_cost_rate:
            breach(
                "portfolio.max_estimated_cost_rate",
                "estimated trading cost exceeds the portfolio limit",
                actual=float(cost),
                limit=float(limits.max_estimated_cost_rate),
            )

    return CheckResult(
        alerts=alerts,
        metrics={
            "positions": len(active),
            "gross_weight": float(gross),
            "cash_weight": float(cash),
            "turnover": float(turnover),
            "industry_weights": {
                industry: float(weight) for industry, weight in sorted(industry_weights.items())
            },
            "estimated_cost_rate": None if cost is None else float(cost),
        },
    )
