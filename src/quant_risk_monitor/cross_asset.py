"""Pure, point-in-time cross-asset policy for quant-execution v0.5.1."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import ClassVar, TypeVar

from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, ensure_utc_datetime
from quant_data_kit.exceptions import ValidationError
from quant_execution import OrderIntent, RiskCheckContext, RiskDecision, Side

_ACCEPTED = RiskDecision(True, "ACCEPTED")
_DERIVATIVE_CLASSES = frozenset({AssetClass.FUTURE, AssetClass.OPTION})
_T = TypeVar("_T")


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _decimal(value: Decimal | str | float, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal normalizes many invalid inputs
        raise ValidationError(f"{name} must be decimal-compatible") from exc
    if not result.is_finite():
        raise ValidationError(f"{name} must be finite")
    return result


def _optional_non_negative(
    value: Decimal | str | float | None,
    name: str,
) -> Decimal | None:
    if value is None:
        return None
    result = _decimal(value, name)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative")
    return result


def _positive_ratio(value: Decimal | str | float, name: str) -> Decimal:
    result = _decimal(value, name)
    if result <= 0:
        raise ValidationError(f"{name} must be positive")
    return result


def _fixed_map(
    values: Mapping[str, FixedPoint],
    name: str,
) -> Mapping[str, FixedPoint]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    frozen: dict[str, FixedPoint] = {}
    for key, value in values.items():
        clean_key = _required_text(key, f"{name} key")
        if clean_key in frozen:
            raise ValidationError(f"{name} contains duplicate normalized keys")
        if not isinstance(value, FixedPoint):
            raise ValidationError(f"{name}[{clean_key!r}] must be a FixedPoint")
        frozen[clean_key] = value
    return MappingProxyType(frozen)


def _decimal_map(
    values: Mapping[object, Decimal | str | float],
    name: str,
    *,
    key_type: type | None = None,
) -> Mapping[object, Decimal]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    frozen: dict[object, Decimal] = {}
    for key, value in values.items():
        if key_type is not None and not isinstance(key, key_type):
            raise ValidationError(f"{name} keys must be {key_type.__name__} values")
        if key_type is str:
            key = _required_text(key, f"{name} key")
        if key in frozen:
            raise ValidationError(f"{name} contains duplicate normalized keys")
        parsed = _optional_non_negative(value, f"{name}[{key!r}]")
        assert parsed is not None
        frozen[key] = parsed
    return MappingProxyType(frozen)


def _pit_times(
    observed_at: datetime,
    available_at: datetime,
) -> tuple[datetime, datetime]:
    observed = ensure_utc_datetime(observed_at, field="observed_at")
    available = ensure_utc_datetime(available_at, field="available_at")
    if available < observed:
        raise ValidationError("available_at must not be earlier than observed_at")
    return observed, available


@dataclass(frozen=True, kw_only=True, slots=True)
class PriceObservation:
    instrument_id: str
    price: FixedPoint
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instrument_id", _required_text(self.instrument_id, "instrument_id")
        )
        if not isinstance(self.price, FixedPoint) or not self.price.is_positive():
            raise ValidationError("price must be a positive FixedPoint")
        observed, available = _pit_times(self.observed_at, self.available_at)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, kw_only=True, slots=True)
class FxRateObservation:
    currency: str
    base_currency: str
    rate_to_base: FixedPoint
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _required_text(self.currency, "currency"))
        object.__setattr__(
            self, "base_currency", _required_text(self.base_currency, "base_currency")
        )
        if not isinstance(self.rate_to_base, FixedPoint) or not self.rate_to_base.is_positive():
            raise ValidationError("rate_to_base must be a positive FixedPoint")
        observed, available = _pit_times(self.observed_at, self.available_at)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, kw_only=True, slots=True)
class LiquidityObservation:
    instrument_id: str
    average_daily_value_base: FixedPoint
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instrument_id", _required_text(self.instrument_id, "instrument_id")
        )
        if (
            not isinstance(self.average_daily_value_base, FixedPoint)
            or not self.average_daily_value_base.is_positive()
        ):
            raise ValidationError("average_daily_value_base must be a positive FixedPoint")
        observed, available = _pit_times(self.observed_at, self.available_at)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, kw_only=True, slots=True)
class StrategyExposureSnapshot:
    strategy_id: str
    gross_exposure_base: FixedPoint
    instrument_notionals_base: Mapping[str, FixedPoint] = field(default_factory=dict)
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", _required_text(self.strategy_id, "strategy_id"))
        if (
            not isinstance(self.gross_exposure_base, FixedPoint)
            or self.gross_exposure_base.units < 0
        ):
            raise ValidationError("gross_exposure_base must be a non-negative FixedPoint")
        notionals = _fixed_map(self.instrument_notionals_base, "instrument_notionals_base")
        if sum(abs(value.to_decimal()) for value in notionals.values()) > (
            self.gross_exposure_base.to_decimal()
        ):
            raise ValidationError("strategy instrument notionals exceed gross exposure")
        object.__setattr__(self, "instrument_notionals_base", notionals)
        observed, available = _pit_times(self.observed_at, self.available_at)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, kw_only=True, slots=True)
class AnalyticsRiskSnapshot:
    historical_var: FixedPoint
    historical_cvar: FixedPoint
    parametric_var: FixedPoint
    parametric_cvar: FixedPoint
    factor_drift_z: Mapping[str, FixedPoint] = field(default_factory=dict)
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "historical_var",
            "historical_cvar",
            "parametric_var",
            "parametric_cvar",
        ):
            value = getattr(self, name)
            if not isinstance(value, FixedPoint) or value.units < 0:
                raise ValidationError(f"{name} must be a non-negative FixedPoint")
        object.__setattr__(
            self, "factor_drift_z", _fixed_map(self.factor_drift_z, "factor_drift_z")
        )
        observed, available = _pit_times(self.observed_at, self.available_at)
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)


@dataclass(frozen=True, kw_only=True, slots=True)
class PITRiskInputs:
    prices: tuple[PriceObservation, ...] = ()
    fx_rates: tuple[FxRateObservation, ...] = ()
    liquidity: tuple[LiquidityObservation, ...] = ()
    strategy_exposures: tuple[StrategyExposureSnapshot, ...] = ()
    analytics: tuple[AnalyticsRiskSnapshot, ...] = ()

    def __post_init__(self) -> None:
        for name, expected, identity_fields in (
            (
                "prices",
                PriceObservation,
                ("instrument_id", "observed_at", "available_at"),
            ),
            (
                "fx_rates",
                FxRateObservation,
                ("currency", "base_currency", "observed_at", "available_at"),
            ),
            (
                "liquidity",
                LiquidityObservation,
                ("instrument_id", "observed_at", "available_at"),
            ),
            (
                "strategy_exposures",
                StrategyExposureSnapshot,
                ("strategy_id", "observed_at", "available_at"),
            ),
            ("analytics", AnalyticsRiskSnapshot, ("observed_at", "available_at")),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, expected) for value in values):
                raise ValidationError(f"{name} contains an invalid observation")
            identities = [
                tuple(getattr(value, field_name) for field_name in identity_fields)
                for value in values
            ]
            if len(identities) != len(set(identities)):
                raise ValidationError(f"{name} contains duplicate PIT observations")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, kw_only=True, slots=True)
class StressScenario:
    name: str
    instrument_shocks: Mapping[str, Decimal | str | float] = field(default_factory=dict)
    asset_class_shocks: Mapping[AssetClass, Decimal | str | float] = field(default_factory=dict)
    currency_shocks: Mapping[str, Decimal | str | float] = field(default_factory=dict)
    venue_shocks: Mapping[str, Decimal | str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(
            self,
            "instrument_shocks",
            _shock_map(self.instrument_shocks, "instrument_shocks", key_type=str),
        )
        object.__setattr__(
            self,
            "asset_class_shocks",
            _shock_map(self.asset_class_shocks, "asset_class_shocks", key_type=AssetClass),
        )
        object.__setattr__(
            self,
            "currency_shocks",
            _shock_map(self.currency_shocks, "currency_shocks", key_type=str),
        )
        object.__setattr__(
            self,
            "venue_shocks",
            _shock_map(self.venue_shocks, "venue_shocks", key_type=str),
        )
        if not any(
            (
                self.instrument_shocks,
                self.asset_class_shocks,
                self.currency_shocks,
                self.venue_shocks,
            )
        ):
            raise ValidationError("stress scenario must define at least one shock")


def _shock_map(
    values: Mapping[object, Decimal | str | float],
    name: str,
    *,
    key_type: type,
) -> Mapping[object, Decimal]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    frozen: dict[object, Decimal] = {}
    for key, value in values.items():
        if not isinstance(key, key_type):
            raise ValidationError(f"{name} contains an invalid key")
        if key_type is str:
            key = _required_text(key, f"{name} key")
        if key in frozen:
            raise ValidationError(f"{name} contains duplicate normalized keys")
        frozen[key] = _decimal(value, f"{name}[{key!r}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True, kw_only=True, slots=True)
class CrossAssetRiskLimits:
    max_gross_leverage: Decimal | str | float | None = None
    max_abs_net_leverage: Decimal | str | float | None = None
    max_instrument_concentration: Decimal | str | float | None = None
    max_asset_class_concentration: Mapping[AssetClass, Decimal | str | float] = field(
        default_factory=dict
    )
    max_currency_concentration: Mapping[str, Decimal | str | float] = field(default_factory=dict)
    max_venue_concentration: Mapping[str, Decimal | str | float] = field(default_factory=dict)
    max_strategy_concentration: Decimal | str | float | None = None
    max_initial_margin_base: Decimal | str | float | None = None
    max_maintenance_margin_base: Decimal | str | float | None = None
    max_margin_utilization: Decimal | str | float | None = None
    max_adv_participation: Decimal | str | float | None = None
    max_days_to_liquidate: Decimal | str | float | None = None
    liquidation_participation_rate: Decimal | str | float = Decimal("0.10")
    max_stress_loss_ratio: Decimal | str | float | None = None
    max_historical_var: Decimal | str | float | None = None
    max_historical_cvar: Decimal | str | float | None = None
    max_parametric_var: Decimal | str | float | None = None
    max_parametric_cvar: Decimal | str | float | None = None
    max_factor_drift_z: Decimal | str | float | None = None

    def __post_init__(self) -> None:
        optional_fields = (
            "max_gross_leverage",
            "max_abs_net_leverage",
            "max_instrument_concentration",
            "max_strategy_concentration",
            "max_initial_margin_base",
            "max_maintenance_margin_base",
            "max_margin_utilization",
            "max_adv_participation",
            "max_days_to_liquidate",
            "max_stress_loss_ratio",
            "max_historical_var",
            "max_historical_cvar",
            "max_parametric_var",
            "max_parametric_cvar",
            "max_factor_drift_z",
        )
        for name in optional_fields:
            object.__setattr__(self, name, _optional_non_negative(getattr(self, name), name))
        participation = _positive_ratio(
            self.liquidation_participation_rate, "liquidation_participation_rate"
        )
        if participation > 1:
            raise ValidationError("liquidation_participation_rate must not exceed 1")
        object.__setattr__(self, "liquidation_participation_rate", participation)
        object.__setattr__(
            self,
            "max_asset_class_concentration",
            _decimal_map(
                self.max_asset_class_concentration,
                "max_asset_class_concentration",
                key_type=AssetClass,
            ),
        )
        object.__setattr__(
            self,
            "max_currency_concentration",
            _decimal_map(
                self.max_currency_concentration,
                "max_currency_concentration",
                key_type=str,
            ),
        )
        object.__setattr__(
            self,
            "max_venue_concentration",
            _decimal_map(
                self.max_venue_concentration,
                "max_venue_concentration",
                key_type=str,
            ),
        )


@dataclass(slots=True)
class _ExposureState:
    as_of: datetime
    base_currency: str
    nav: Decimal
    notionals: dict[str, Decimal]
    specs: dict[str, InstrumentSpec]
    initial_margin_by_instrument: dict[str, Decimal]
    maintenance_margin_by_instrument: dict[str, Decimal]
    initial_margin: Decimal
    maintenance_margin: Decimal

    @property
    def gross(self) -> Decimal:
        return sum((abs(value) for value in self.notionals.values()), Decimal(0))

    @property
    def net(self) -> Decimal:
        return sum(self.notionals.values(), Decimal(0))


@dataclass(frozen=True, slots=True, init=False)
class CrossAssetRiskPolicy:
    """Immutable policy implementing quant_execution.PortfolioRiskPolicy."""

    sends_live_orders: ClassVar[bool] = False
    instruments: Mapping[str, InstrumentSpec]
    limits: CrossAssetRiskLimits
    inputs: PITRiskInputs
    stress_scenarios: tuple[StressScenario, ...]

    def __init__(
        self,
        *,
        instruments: Mapping[str, InstrumentSpec],
        limits: CrossAssetRiskLimits,
        inputs: PITRiskInputs,
        stress_scenarios: Sequence[StressScenario] = (),
    ) -> None:
        if not isinstance(limits, CrossAssetRiskLimits):
            raise ValidationError("limits must be CrossAssetRiskLimits")
        if not isinstance(inputs, PITRiskInputs):
            raise ValidationError("inputs must be PITRiskInputs")
        registry = dict(instruments)
        if any(not isinstance(spec, InstrumentSpec) for spec in registry.values()):
            raise ValidationError("instruments must contain InstrumentSpec values")
        if any(key != spec.instrument_id for key, spec in registry.items()):
            raise ValidationError(
                "instrument registry keys must match InstrumentSpec.instrument_id"
            )
        scenarios = tuple(stress_scenarios)
        if any(not isinstance(scenario, StressScenario) for scenario in scenarios):
            raise ValidationError("stress_scenarios contains an invalid scenario")
        if limits.max_stress_loss_ratio is not None and not scenarios:
            raise ValidationError("stress scenarios are required when the stress limit is enabled")
        object.__setattr__(self, "instruments", MappingProxyType(registry))
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "stress_scenarios", scenarios)

    def check_order(
        self,
        order_intent: OrderIntent,
        context: RiskCheckContext,
    ) -> RiskDecision:
        state_or_decision = self._state(context)
        if isinstance(state_or_decision, RiskDecision):
            return state_or_decision
        state = state_or_decision
        spec_or_decision = self._order_spec(order_intent, context, state.as_of)
        if isinstance(spec_or_decision, RiskDecision):
            return spec_or_decision
        spec = spec_or_decision
        input_decision = self._validate_order_reference(order_intent, context, spec, state)
        if input_decision is not None:
            return input_decision

        delta = context.projected_notional_base.to_decimal()
        current = state.notionals.get(order_intent.instrument_id, Decimal(0))
        projected = _project_position(current, delta, reduce_only=order_intent.reduce_only)
        margin_decision = self._replace_position(
            state, spec, current, projected, order_intent.reduce_only
        )
        if margin_decision is not None:
            return margin_decision

        liquidity = self._check_liquidity(
            state,
            instrument_id=order_intent.instrument_id,
            order_notional=abs(delta),
        )
        if liquidity is not None:
            return liquidity
        return self._evaluate(state, order_intent=order_intent, order_delta=delta)

    def runtime_check(self, context: RiskCheckContext) -> RiskDecision:
        state_or_decision = self._state(context)
        if isinstance(state_or_decision, RiskDecision):
            return state_or_decision
        state = state_or_decision
        liquidity = self._check_liquidity(state)
        if liquidity is not None:
            return liquidity
        return self._evaluate(state)

    def _state(self, context: RiskCheckContext) -> _ExposureState | RiskDecision:
        portfolio = context.portfolio_snapshot
        state = _ExposureState(
            as_of=portfolio.event_time,
            base_currency=portfolio.base_currency,
            nav=portfolio.nav.to_decimal(),
            notionals={},
            specs={},
            initial_margin_by_instrument={},
            maintenance_margin_by_instrument={},
            initial_margin=portfolio.initial_margin.to_decimal(),
            maintenance_margin=portfolio.maintenance_margin.to_decimal(),
        )
        for position in portfolio.positions:
            spec_or_decision = self._causal_spec(position.instrument_id, state.as_of)
            if isinstance(spec_or_decision, RiskDecision):
                return spec_or_decision
            spec = spec_or_decision
            if (
                position.asset_class is not spec.asset_class
                or position.venue != spec.venue
                or position.settlement_currency != spec.settlement_currency
            ):
                return _reject(
                    "CLASSIFICATION_MISMATCH",
                    f"risk snapshot classification differs for {position.instrument_id}",
                )
            reference = self._price(position.instrument_id, state.as_of)
            if isinstance(reference, RiskDecision):
                return reference
            if reference.price.to_decimal() != position.mark_price.to_decimal():
                return _reject(
                    "PIT_PRICE_MISMATCH",
                    f"PIT price differs from QExec mark for {position.instrument_id}",
                )
            fx_or_decision = self._fx(spec.settlement_currency, state.base_currency, state.as_of)
            if isinstance(fx_or_decision, RiskDecision):
                return fx_or_decision
            expected = (
                position.quantity.to_decimal()
                * position.mark_price.to_decimal()
                * spec.contract_multiplier.to_decimal()
                * fx_or_decision
            )
            if not _fixed_equal(expected, position.base_notional):
                return _reject(
                    "PIT_FX_MISMATCH",
                    f"PIT FX differs from QExec base notional for {position.instrument_id}",
                )
            state.notionals[position.instrument_id] = position.base_notional.to_decimal()
            state.specs[position.instrument_id] = spec
            state.initial_margin_by_instrument[position.instrument_id] = (
                position.initial_margin.to_decimal()
            )
            state.maintenance_margin_by_instrument[position.instrument_id] = (
                position.maintenance_margin.to_decimal()
            )
        if (
            state.gross != portfolio.gross_exposure.to_decimal()
            or state.net != portfolio.net_exposure.to_decimal()
            or sum(state.initial_margin_by_instrument.values(), Decimal(0)) != state.initial_margin
            or sum(state.maintenance_margin_by_instrument.values(), Decimal(0))
            != state.maintenance_margin
        ):
            return _reject("RISK_SNAPSHOT_INCONSISTENT", "QExec risk snapshot totals disagree")
        return state

    def _causal_spec(self, instrument_id: str, as_of: datetime) -> InstrumentSpec | RiskDecision:
        spec = self.instruments.get(instrument_id)
        if spec is None or spec.asset_class in {AssetClass.CASH, AssetClass.OTHER}:
            return _reject(
                "MISSING_CLASSIFICATION", f"classification is missing for {instrument_id}"
            )
        if spec.available_at > as_of:
            return _reject(
                "PIT_CLASSIFICATION_NOT_AVAILABLE",
                f"classification is not available for {instrument_id}",
            )
        if spec.effective_from > as_of or (
            spec.effective_to is not None and spec.effective_to <= as_of
        ):
            return _reject(
                "INSTRUMENT_NOT_EFFECTIVE",
                f"instrument is not effective for {instrument_id}",
            )
        if spec.inverse:
            return _reject(
                "UNSUPPORTED_INVERSE_CONTRACT",
                f"inverse contract is outside the v2.0 linear risk contract: {instrument_id}",
            )
        return spec

    def _order_spec(
        self,
        order_intent: OrderIntent,
        context: RiskCheckContext,
        as_of: datetime,
    ) -> InstrumentSpec | RiskDecision:
        spec_or_decision = self._causal_spec(order_intent.instrument_id, as_of)
        if isinstance(spec_or_decision, RiskDecision):
            return spec_or_decision
        context_spec = context.instrument_spec
        if context_spec is None or context_spec != spec_or_decision:
            return _reject(
                "CLASSIFICATION_MISMATCH",
                f"QExec InstrumentSpec differs for {order_intent.instrument_id}",
            )
        return spec_or_decision

    def _validate_order_reference(
        self,
        order_intent: OrderIntent,
        context: RiskCheckContext,
        spec: InstrumentSpec,
        state: _ExposureState,
    ) -> RiskDecision | None:
        reference = self._price(order_intent.instrument_id, state.as_of)
        if isinstance(reference, RiskDecision):
            return reference
        if context.reference_price is None or (
            reference.price.to_decimal() != context.reference_price.to_decimal()
        ):
            return _reject(
                "PIT_PRICE_MISMATCH",
                f"PIT price differs from QExec reference for {order_intent.instrument_id}",
            )
        fx_or_decision = self._fx(spec.settlement_currency, state.base_currency, state.as_of)
        if isinstance(fx_or_decision, RiskDecision):
            return fx_or_decision
        signed = Decimal(1) if order_intent.side is Side.BUY else Decimal(-1)
        expected = (
            order_intent.quantity.to_decimal()
            * context.reference_price.to_decimal()
            * spec.contract_multiplier.to_decimal()
            * fx_or_decision
            * signed
        )
        projected = context.projected_notional_base
        if projected is None or not _fixed_equal(expected, projected):
            return _reject(
                "PIT_FX_MISMATCH",
                f"PIT FX differs from QExec projected notional for {order_intent.instrument_id}",
            )
        return None

    def _price(self, instrument_id: str, as_of: datetime) -> PriceObservation | RiskDecision:
        value, status = _latest(
            self.inputs.prices,
            lambda item: item.instrument_id == instrument_id,
            as_of,
        )
        if status == "missing":
            return _reject("MISSING_PIT_PRICE", f"price is missing for {instrument_id}")
        if status == "future":
            return _reject("PIT_PRICE_NOT_AVAILABLE", f"price is not available for {instrument_id}")
        return value

    def _fx(
        self,
        currency: str,
        base_currency: str,
        as_of: datetime,
    ) -> Decimal | RiskDecision:
        if currency == base_currency:
            return Decimal(1)
        value, status = _latest(
            self.inputs.fx_rates,
            lambda item: item.currency == currency and item.base_currency == base_currency,
            as_of,
        )
        if status == "missing":
            return _reject("MISSING_PIT_FX", f"FX rate is missing for {currency}/{base_currency}")
        if status == "future":
            return _reject(
                "PIT_FX_NOT_AVAILABLE",
                f"FX rate is not available for {currency}/{base_currency}",
            )
        return value.rate_to_base.to_decimal()

    def _replace_position(
        self,
        state: _ExposureState,
        spec: InstrumentSpec,
        current: Decimal,
        projected: Decimal,
        reduce_only: bool,
    ) -> RiskDecision | None:
        instrument_id = spec.instrument_id
        current_initial = state.initial_margin_by_instrument.get(instrument_id, Decimal(0))
        current_maintenance = state.maintenance_margin_by_instrument.get(instrument_id, Decimal(0))
        projected_margin = self._projected_margin(
            spec,
            current=current,
            projected=projected,
            current_initial=current_initial,
            current_maintenance=current_maintenance,
            reduce_only=reduce_only,
        )
        if isinstance(projected_margin, RiskDecision):
            return projected_margin
        projected_initial, projected_maintenance = projected_margin
        state.initial_margin += projected_initial - current_initial
        state.maintenance_margin += projected_maintenance - current_maintenance
        state.initial_margin_by_instrument[instrument_id] = projected_initial
        state.maintenance_margin_by_instrument[instrument_id] = projected_maintenance
        state.notionals[instrument_id] = projected
        state.specs[instrument_id] = spec
        return None

    @staticmethod
    def _projected_margin(
        spec: InstrumentSpec,
        *,
        current: Decimal,
        projected: Decimal,
        current_initial: Decimal,
        current_maintenance: Decimal,
        reduce_only: bool,
    ) -> tuple[Decimal, Decimal] | RiskDecision:
        if not _is_derivative(spec):
            return Decimal(0), Decimal(0)
        same_direction = not projected or current * projected > 0
        if reduce_only or (current and same_direction and abs(projected) <= abs(current)):
            ratio = abs(projected) / abs(current) if current else Decimal(0)
            return current_initial * ratio, current_maintenance * ratio
        missing = [
            name
            for name in ("initial_margin_rate", "maintenance_margin_rate")
            if name not in spec.metadata
        ]
        if missing:
            return _reject(
                "MISSING_MARGIN_RATE",
                f"derivative margin rate is missing: {','.join(missing)}",
            )
        try:
            initial_rate = _positive_ratio(
                spec.metadata["initial_margin_rate"], "initial_margin_rate"
            )
            maintenance_rate = _positive_ratio(
                spec.metadata["maintenance_margin_rate"], "maintenance_margin_rate"
            )
        except ValidationError as exc:
            return _reject("INVALID_MARGIN_RATE", str(exc))
        if maintenance_rate > initial_rate:
            return _reject(
                "INVALID_MARGIN_RATE",
                "maintenance_margin_rate cannot exceed initial_margin_rate",
            )
        return abs(projected) * initial_rate, abs(projected) * maintenance_rate

    def _check_liquidity(
        self,
        state: _ExposureState,
        *,
        instrument_id: str | None = None,
        order_notional: Decimal | None = None,
    ) -> RiskDecision | None:
        limits = self.limits
        enabled = (
            limits.max_adv_participation is not None or limits.max_days_to_liquidate is not None
        )
        if not enabled:
            return None
        targets = (
            (instrument_id,)
            if instrument_id is not None
            else tuple(sorted(key for key, value in state.notionals.items() if value))
        )
        for target in targets:
            observation, status = _latest(
                self.inputs.liquidity,
                lambda item, target=target: item.instrument_id == target,
                state.as_of,
            )
            if status == "missing":
                return _reject("MISSING_PIT_ADV", f"ADV is missing for {target}")
            if status == "future":
                return _reject("PIT_ADV_NOT_AVAILABLE", f"ADV is not available for {target}")
            adv = observation.average_daily_value_base.to_decimal()
            if (
                order_notional is not None
                and limits.max_adv_participation is not None
                and order_notional / adv > limits.max_adv_participation
            ):
                return _reject(
                    "ADV_PARTICIPATION_LIMIT",
                    f"order ADV participation exceeds limit for {target}",
                )
            if limits.max_days_to_liquidate is not None:
                days = abs(state.notionals.get(target, Decimal(0))) / (
                    adv * limits.liquidation_participation_rate
                )
                if days > limits.max_days_to_liquidate:
                    return _reject(
                        "DAYS_TO_LIQUIDATE_LIMIT",
                        f"days-to-liquidate exceeds limit for {target}",
                    )
        return None

    def _evaluate(
        self,
        state: _ExposureState,
        *,
        order_intent: OrderIntent | None = None,
        order_delta: Decimal = Decimal(0),
    ) -> RiskDecision:
        if state.nav <= 0:
            return _reject("NAV_NOT_POSITIVE", "portfolio NAV must be positive")
        limits = self.limits
        if (
            limits.max_gross_leverage is not None
            and state.gross / state.nav > limits.max_gross_leverage
        ):
            return _reject("GROSS_LEVERAGE_LIMIT", "gross leverage exceeds limit")
        if (
            limits.max_abs_net_leverage is not None
            and abs(state.net) / state.nav > limits.max_abs_net_leverage
        ):
            return _reject("NET_LEVERAGE_LIMIT", "absolute net leverage exceeds limit")
        if limits.max_instrument_concentration is not None:
            for instrument_id in sorted(state.notionals):
                if (
                    abs(state.notionals[instrument_id]) / state.nav
                    > limits.max_instrument_concentration
                ):
                    return _reject(
                        "INSTRUMENT_CONCENTRATION_LIMIT",
                        f"instrument concentration exceeds limit for {instrument_id}",
                    )
        grouped_checks = (
            (
                "ASSET_CLASS_CONCENTRATION_LIMIT",
                limits.max_asset_class_concentration,
                lambda spec: spec.asset_class,
            ),
            (
                "CURRENCY_CONCENTRATION_LIMIT",
                limits.max_currency_concentration,
                lambda spec: spec.settlement_currency,
            ),
            (
                "VENUE_CONCENTRATION_LIMIT",
                limits.max_venue_concentration,
                lambda spec: spec.venue,
            ),
        )
        for code, configured, classifier in grouped_checks:
            exposures: dict[object, Decimal] = {}
            for instrument_id, notional in state.notionals.items():
                key = classifier(state.specs[instrument_id])
                exposures[key] = exposures.get(key, Decimal(0)) + abs(notional)
            for key in sorted(configured, key=str):
                if exposures.get(key, Decimal(0)) / state.nav > configured[key]:
                    return _reject(code, f"{key} concentration exceeds limit")
        strategy_decision = self._check_strategy(
            state,
            order_intent=order_intent,
            order_delta=order_delta,
        )
        if strategy_decision is not None:
            return strategy_decision
        if (
            limits.max_initial_margin_base is not None
            and state.initial_margin > limits.max_initial_margin_base
        ):
            return _reject("INITIAL_MARGIN_LIMIT", "initial margin exceeds limit")
        if (
            limits.max_maintenance_margin_base is not None
            and state.maintenance_margin > limits.max_maintenance_margin_base
        ):
            return _reject("MAINTENANCE_MARGIN_LIMIT", "maintenance margin exceeds limit")
        if (
            limits.max_margin_utilization is not None
            and state.initial_margin / state.nav > limits.max_margin_utilization
        ):
            return _reject("MARGIN_UTILIZATION_LIMIT", "margin utilization exceeds limit")
        stress = self._check_stress(state)
        if stress is not None:
            return stress
        analytics = self._check_analytics(state.as_of)
        if analytics is not None:
            return analytics
        return _ACCEPTED

    def _check_strategy(
        self,
        state: _ExposureState,
        *,
        order_intent: OrderIntent | None,
        order_delta: Decimal,
    ) -> RiskDecision | None:
        limit = self.limits.max_strategy_concentration
        if limit is None:
            return None
        latest_by_strategy: dict[str, StrategyExposureSnapshot] = {}
        for item in self.inputs.strategy_exposures:
            if item.available_at <= state.as_of:
                prior = latest_by_strategy.get(item.strategy_id)
                if prior is None or (item.available_at, item.observed_at) > (
                    prior.available_at,
                    prior.observed_at,
                ):
                    latest_by_strategy[item.strategy_id] = item
        if order_intent is not None:
            item = latest_by_strategy.get(order_intent.strategy_id)
            if item is None:
                return _reject(
                    "MISSING_STRATEGY_EXPOSURE",
                    f"strategy exposure is missing for {order_intent.strategy_id}",
                )
            current = item.instrument_notionals_base.get(
                order_intent.instrument_id, FixedPoint(0, 0)
            ).to_decimal()
            projected = _project_position(current, order_delta, order_intent.reduce_only)
            gross = item.gross_exposure_base.to_decimal() - abs(current) + abs(projected)
            if gross / state.nav > limit:
                return _reject(
                    "STRATEGY_CONCENTRATION_LIMIT",
                    f"strategy concentration exceeds limit for {order_intent.strategy_id}",
                )
            return None
        if not latest_by_strategy:
            return _reject("MISSING_STRATEGY_EXPOSURE", "strategy exposure is missing")
        for strategy_id in sorted(latest_by_strategy):
            if latest_by_strategy[strategy_id].gross_exposure_base.to_decimal() / state.nav > limit:
                return _reject(
                    "STRATEGY_CONCENTRATION_LIMIT",
                    f"strategy concentration exceeds limit for {strategy_id}",
                )
        return None

    def _check_stress(self, state: _ExposureState) -> RiskDecision | None:
        limit = self.limits.max_stress_loss_ratio
        if limit is None:
            return None
        for scenario in self.stress_scenarios:
            pnl = Decimal(0)
            for instrument_id, notional in state.notionals.items():
                spec = state.specs[instrument_id]
                shock = (
                    scenario.instrument_shocks.get(instrument_id, Decimal(0))
                    + scenario.asset_class_shocks.get(spec.asset_class, Decimal(0))
                    + scenario.currency_shocks.get(spec.settlement_currency, Decimal(0))
                    + scenario.venue_shocks.get(spec.venue, Decimal(0))
                )
                pnl += notional * shock
            loss = max(-pnl, Decimal(0))
            if loss / state.nav > limit:
                return _reject(
                    "STRESS_LOSS_LIMIT", f"stress loss exceeds limit for {scenario.name}"
                )
        return None

    def _check_analytics(self, as_of: datetime) -> RiskDecision | None:
        limits = self.limits
        configured = (
            ("HISTORICAL_VAR_LIMIT", limits.max_historical_var, "historical_var"),
            ("HISTORICAL_CVAR_LIMIT", limits.max_historical_cvar, "historical_cvar"),
            ("PARAMETRIC_VAR_LIMIT", limits.max_parametric_var, "parametric_var"),
            ("PARAMETRIC_CVAR_LIMIT", limits.max_parametric_cvar, "parametric_cvar"),
        )
        if not any(limit is not None for _, limit, _ in configured) and (
            limits.max_factor_drift_z is None
        ):
            return None
        snapshot, status = _latest(self.inputs.analytics, lambda _: True, as_of)
        if status == "missing":
            return _reject("MISSING_ANALYTICS_RISK", "analytics risk snapshot is missing")
        if status == "future":
            return _reject(
                "PIT_ANALYTICS_NOT_AVAILABLE", "analytics risk snapshot is not available"
            )
        for code, limit, field_name in configured:
            if limit is not None and getattr(snapshot, field_name).to_decimal() > limit:
                return _reject(code, f"{field_name} exceeds limit")
        if limits.max_factor_drift_z is not None:
            for factor in sorted(snapshot.factor_drift_z):
                if abs(snapshot.factor_drift_z[factor].to_decimal()) > limits.max_factor_drift_z:
                    return _reject("FACTOR_DRIFT_LIMIT", f"factor drift exceeds limit for {factor}")
        return None


def _latest(
    values: Sequence[_T],
    predicate: Callable[[_T], bool],
    as_of: datetime,
) -> tuple[_T | None, str]:
    matches = [value for value in values if predicate(value)]
    if not matches:
        return None, "missing"
    causal = [value for value in matches if value.available_at <= as_of]
    if not causal:
        return None, "future"
    return max(causal, key=lambda value: (value.available_at, value.observed_at)), "ok"


def _project_position(current: Decimal, delta: Decimal, reduce_only: bool) -> Decimal:
    if not reduce_only:
        return current + delta
    if not current or current * delta >= 0:
        return current
    remaining = max(abs(current) - abs(delta), Decimal(0))
    return remaining.copy_sign(current)


def _is_derivative(spec: InstrumentSpec) -> bool:
    return spec.asset_class in _DERIVATIVE_CLASSES or spec.product_type in {
        "linear_perpetual",
        "inverse_perpetual",
    }


def _fixed_equal(expected: Decimal, actual: FixedPoint) -> bool:
    return expected == actual.to_decimal()


def _reject(code: str, message: str) -> RiskDecision:
    return RiskDecision(False, code, message)


__all__ = [
    "AnalyticsRiskSnapshot",
    "CrossAssetRiskLimits",
    "CrossAssetRiskPolicy",
    "FxRateObservation",
    "LiquidityObservation",
    "PITRiskInputs",
    "PriceObservation",
    "StrategyExposureSnapshot",
    "StressScenario",
]
