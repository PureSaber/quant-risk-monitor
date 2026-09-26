from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType

import pytest
from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, MarginMode, TradeEvent
from quant_data_kit.exceptions import ValidationError
from quant_execution import (
    AccountSnapshot,
    ExactAccountLedger,
    OrderIntent,
    OrderType,
    PortfolioRiskSnapshot,
    PositionRiskSnapshot,
    RiskCheckContext,
    RuleBookRiskGate,
    Side,
    TimeInForce,
)

from quant_risk_monitor.cross_asset import (
    AnalyticsRiskSnapshot,
    CrossAssetRiskLimits,
    CrossAssetRiskPolicy,
    FxRateObservation,
    LiquidityObservation,
    PITRiskInputs,
    PriceObservation,
    StrategyExposureSnapshot,
    StressScenario,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 2, 1, 0, tzinfo=UTC)
STOCK = "equity:sse:600000"
FUTURE = "future:cffex:IF2603"
PERP = "crypto:binance:BTCUSDT-PERP"


def fp(value: str | int, scale: int = 2) -> FixedPoint:
    return FixedPoint.from_decimal(Decimal(str(value)), scale)


def stock_spec() -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=STOCK,
        asset_class=AssetClass.EQUITY,
        product_type="a_share",
        venue="SSE",
        native_symbol="600000",
        settlement_currency="CNY",
        price_tick=fp("0.01"),
        quantity_step=fp("100", 0),
        contract_multiplier=fp("1", 0),
        calendar_id="SSE",
        margin_mode=MarginMode.NONE,
        effective_from=T0 - timedelta(days=365),
        available_at=T0 - timedelta(days=365),
        metadata={"lot_size": "100"},
    )


def future_spec(*, metadata: dict[str, str] | None = None) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=FUTURE,
        asset_class=AssetClass.FUTURE,
        product_type="index_future",
        venue="CFFEX",
        native_symbol="IF2603",
        settlement_currency="CNY",
        price_tick=fp("0.2", 1),
        quantity_step=fp("1", 0),
        contract_multiplier=fp("300", 0),
        calendar_id="CFFEX",
        margin_mode=MarginMode.CROSS,
        effective_from=T0 - timedelta(days=365),
        available_at=T0 - timedelta(days=365),
        metadata=metadata
        if metadata is not None
        else {"initial_margin_rate": "0.10", "maintenance_margin_rate": "0.08"},
    )


def perp_spec() -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=PERP,
        asset_class=AssetClass.CRYPTO,
        product_type="linear_perpetual",
        venue="BINANCE",
        native_symbol="BTCUSDT",
        base_currency="BTC",
        quote_currency="USDT",
        settlement_currency="USDT",
        price_tick=fp("0.01"),
        quantity_step=fp("0.001", 3),
        contract_multiplier=fp("1", 0),
        calendar_id="CRYPTO-24X7",
        margin_mode=MarginMode.CROSS,
        effective_from=T0 - timedelta(days=365),
        available_at=T0 - timedelta(days=365),
        metadata={
            "min_quantity": "0.001",
            "initial_margin_rate": "0.10",
            "maintenance_margin_rate": "0.05",
            "maker_fee_rate": "0.0002",
            "taker_fee_rate": "0.0005",
        },
    )


def trade() -> TradeEvent:
    return TradeEvent(
        event_id="trade:stock",
        instrument_id=STOCK,
        event_time=T0,
        received_at=T0,
        available_at=T0,
        source="fixture",
        trading_day=date(2026, 1, 2),
        session_id="sse:2026-01-02",
        sequence=1,
        price=fp("10"),
        quantity=fp("10000", 0),
    )


def intent(quantity: str) -> OrderIntent:
    return OrderIntent(
        idempotency_key=f"order:{quantity}",
        account_id="account",
        strategy_id="strategy",
        instrument_id=STOCK,
        side=Side.BUY,
        quantity=fp(quantity, 0),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        created_at=T0,
        limit_price=fp("10"),
    )


def asset_intent(
    instrument_id: str,
    quantity: str,
    price: str,
    *,
    side: Side = Side.BUY,
    reduce_only: bool = False,
) -> OrderIntent:
    quantity_scale = len(quantity.split(".")[1]) if "." in quantity else 0
    return OrderIntent(
        idempotency_key=f"order:{instrument_id}:{side.value}:{quantity}:{reduce_only}",
        account_id="account",
        strategy_id="strategy",
        instrument_id=instrument_id,
        side=side,
        quantity=fp(quantity, quantity_scale),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        created_at=T0,
        limit_price=fp(price, len(price.split(".")[1]) if "." in price else 0),
        reduce_only=reduce_only,
    )


def price(instrument_id: str, value: str, *, available_at: datetime = T0) -> PriceObservation:
    scale = len(value.split(".")[1]) if "." in value else 0
    return PriceObservation(
        instrument_id=instrument_id,
        price=fp(value, scale),
        observed_at=min(T0, available_at),
        available_at=available_at,
    )


def fx(
    currency: str,
    base_currency: str,
    value: str,
    *,
    available_at: datetime = T0,
) -> FxRateObservation:
    scale = len(value.split(".")[1]) if "." in value else 0
    return FxRateObservation(
        currency=currency,
        base_currency=base_currency,
        rate_to_base=fp(value, scale),
        observed_at=min(T0, available_at),
        available_at=available_at,
    )


def liquidity(
    instrument_id: str,
    adv: str,
    *,
    available_at: datetime = T0,
) -> LiquidityObservation:
    return LiquidityObservation(
        instrument_id=instrument_id,
        average_daily_value_base=fp(adv),
        observed_at=min(T0, available_at),
        available_at=available_at,
    )


def empty_runtime_context(*, nav: str = "100000", base_currency: str = "CNY") -> RiskCheckContext:
    account = AccountSnapshot(
        account_id="account",
        event_time=T0,
        base_currency=base_currency,
        cash_balances={base_currency: fp(nav)},
        nav=fp(nav),
    )
    portfolio = PortfolioRiskSnapshot(
        account_id="account",
        event_time=T0,
        base_currency=base_currency,
        nav=fp(nav),
        cash_value=fp(nav),
        gross_exposure=fp("0"),
        net_exposure=fp("0"),
        initial_margin=fp("0"),
        maintenance_margin=fp("0"),
    )
    return RiskCheckContext(account_snapshot=account, portfolio_snapshot=portfolio)


def single_position_context(
    spec: InstrumentSpec,
    *,
    quantity: FixedPoint,
    mark_price: FixedPoint,
    base_notional: FixedPoint,
    nav: FixedPoint,
    initial_margin: FixedPoint,
    maintenance_margin: FixedPoint,
    order_spec: InstrumentSpec | None = None,
    reference_price: FixedPoint | None = None,
    projected_notional_base: FixedPoint | None = None,
) -> RiskCheckContext:
    position = PositionRiskSnapshot(
        instrument_id=spec.instrument_id,
        asset_class=spec.asset_class,
        venue=spec.venue,
        settlement_currency=spec.settlement_currency,
        quantity=quantity,
        mark_price=mark_price,
        base_notional=base_notional,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
    )
    account = AccountSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="CNY",
        positions={spec.instrument_id: quantity},
        nav=nav,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
    )
    portfolio = PortfolioRiskSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="CNY",
        nav=nav,
        cash_value=fp("0"),
        gross_exposure=FixedPoint.from_decimal(
            abs(base_notional.to_decimal()), base_notional.scale
        ),
        net_exposure=base_notional,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        positions=(position,),
    )
    return RiskCheckContext(
        account_snapshot=account,
        portfolio_snapshot=portfolio,
        instrument_spec=order_spec,
        reference_price=reference_price,
        projected_notional_base=projected_notional_base,
    )


def two_stock_context(
    first: InstrumentSpec,
    second: InstrumentSpec,
    *,
    first_notional: str,
    second_notional: str,
    projected_notional_base: str,
) -> RiskCheckContext:
    notionals = {
        first.instrument_id: Decimal(first_notional),
        second.instrument_id: Decimal(second_notional),
    }
    specs = {first.instrument_id: first, second.instrument_id: second}
    positions = tuple(
        PositionRiskSnapshot(
            instrument_id=instrument_id,
            asset_class=specs[instrument_id].asset_class,
            venue=specs[instrument_id].venue,
            settlement_currency=specs[instrument_id].settlement_currency,
            quantity=FixedPoint.from_decimal(notional / Decimal(10), 0),
            mark_price=fp("10"),
            base_notional=FixedPoint.from_decimal(notional, 0),
            initial_margin=fp("0"),
            maintenance_margin=fp("0"),
        )
        for instrument_id, notional in notionals.items()
    )
    account = AccountSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="CNY",
        positions={position.instrument_id: position.quantity for position in positions},
        nav=fp("100000"),
        initial_margin=fp("0"),
        maintenance_margin=fp("0"),
    )
    portfolio = PortfolioRiskSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="CNY",
        nav=fp("100000"),
        cash_value=fp("0"),
        gross_exposure=FixedPoint.from_decimal(sum(abs(value) for value in notionals.values()), 0),
        net_exposure=FixedPoint.from_decimal(sum(notionals.values()), 0),
        initial_margin=fp("0"),
        maintenance_margin=fp("0"),
        positions=positions,
    )
    return RiskCheckContext(
        account_snapshot=account,
        portfolio_snapshot=portfolio,
        instrument_spec=first,
        reference_price=fp("10"),
        projected_notional_base=fp(projected_notional_base, 0),
    )


def gate_for(
    spec: InstrumentSpec,
    policy: CrossAssetRiskPolicy,
    *,
    price_value: str,
    base_currency: str,
    cash: str,
    fx_to_base: dict[str, FixedPoint] | None = None,
) -> tuple[RuleBookRiskGate, ExactAccountLedger]:
    ledger = ExactAccountLedger(
        account_id="account",
        base_currency=base_currency,
        instruments={spec.instrument_id: spec},
        initial_cash={spec.settlement_currency: fp(cash)},
        fx_to_base=fx_to_base or {},
    )
    gate = RuleBookRiskGate(
        instruments={spec.instrument_id: spec},
        ledger=ledger,
        policies=[policy],
    )
    event = TradeEvent(
        event_id=f"trade:{spec.instrument_id}",
        instrument_id=spec.instrument_id,
        event_time=T0,
        received_at=T0,
        available_at=T0,
        source="fixture",
        trading_day=date(2026, 1, 2),
        session_id="fixture:2026-01-02",
        sequence=1,
        price=fp(
            price_value,
            len(price_value.split(".")[1]) if "." in price_value else 0,
        ),
        quantity=fp("1000000"),
    )
    gate.observe(event)
    ledger.observe_market(event)
    return gate, ledger


def test_real_rulebook_uses_cross_asset_policy_on_projected_order_risk() -> None:
    spec = stock_spec()
    ledger = ExactAccountLedger(
        account_id="account",
        base_currency="CNY",
        instruments={STOCK: spec},
        initial_cash={"CNY": fp("100000")},
    )
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="0.10"),
        inputs=PITRiskInputs(
            prices=(
                PriceObservation(
                    instrument_id=STOCK,
                    price=fp("10"),
                    observed_at=T0,
                    available_at=T0,
                ),
            )
        ),
    )
    gate = RuleBookRiskGate(instruments={STOCK: spec}, ledger=ledger, policies=[policy])
    event = trade()
    gate.observe(event)
    ledger.observe_market(event)

    assert gate.check_current(intent("1000"), event_time=T0).accepted
    rejected = gate.check_current(intent("1100"), event_time=T0)
    assert rejected.code == "GROSS_LEVERAGE_LIMIT"
    assert not rejected.accepted


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (CrossAssetRiskLimits(max_gross_leverage="0.10"), "GROSS_LEVERAGE_LIMIT"),
        (CrossAssetRiskLimits(max_abs_net_leverage="0.10"), "NET_LEVERAGE_LIMIT"),
        (
            CrossAssetRiskLimits(max_instrument_concentration="0.10"),
            "INSTRUMENT_CONCENTRATION_LIMIT",
        ),
        (
            CrossAssetRiskLimits(max_asset_class_concentration={AssetClass.EQUITY: "0.10"}),
            "ASSET_CLASS_CONCENTRATION_LIMIT",
        ),
        (
            CrossAssetRiskLimits(max_currency_concentration={"CNY": "0.10"}),
            "CURRENCY_CONCENTRATION_LIMIT",
        ),
        (
            CrossAssetRiskLimits(max_venue_concentration={"SSE": "0.10"}),
            "VENUE_CONCENTRATION_LIMIT",
        ),
        (
            CrossAssetRiskLimits(max_strategy_concentration="0.10"),
            "STRATEGY_CONCENTRATION_LIMIT",
        ),
    ],
)
def test_exposure_limits_accept_boundary_and_reject_excess(
    limits: CrossAssetRiskLimits,
    code: str,
) -> None:
    spec = stock_spec()
    strategy_inputs = (
        StrategyExposureSnapshot(
            strategy_id="strategy",
            gross_exposure_base=fp("0"),
            observed_at=T0,
            available_at=T0,
        ),
    )
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=limits,
        inputs=PITRiskInputs(
            prices=(price(STOCK, "10"),),
            strategy_exposures=strategy_inputs,
        ),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )

    assert gate.check_current(intent("1000"), event_time=T0).accepted
    decision = gate.check_current(intent("1100"), event_time=T0)
    assert decision.code == code


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (CrossAssetRiskLimits(max_initial_margin_base="120000"), "INITIAL_MARGIN_LIMIT"),
        (
            CrossAssetRiskLimits(max_maintenance_margin_base="96000"),
            "MAINTENANCE_MARGIN_LIMIT",
        ),
        (CrossAssetRiskLimits(max_margin_utilization="0.12"), "MARGIN_UTILIZATION_LIMIT"),
    ],
)
def test_future_margin_limits_accept_boundary_and_reject_excess(
    limits: CrossAssetRiskLimits,
    code: str,
) -> None:
    spec = future_spec()
    policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=limits,
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="4000",
        base_currency="CNY",
        cash="1000000",
    )

    boundary = asset_intent(FUTURE, "1", "4000")
    assert gate.check_current(boundary, event_time=T0).accepted
    decision = gate.check_current(asset_intent(FUTURE, "2", "4000"), event_time=T0)
    assert decision.code == code


def test_derivative_crossing_position_uses_current_margin_rates() -> None:
    spec = future_spec(metadata={"initial_margin_rate": "0.20", "maintenance_margin_rate": "0.15"})
    policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=CrossAssetRiskLimits(max_initial_margin_base="150000"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    context = single_position_context(
        spec,
        quantity=fp("-1", 0),
        mark_price=fp("4000", 0),
        base_notional=fp("-1200000"),
        nav=fp("1000000"),
        initial_margin=fp("120000"),
        maintenance_margin=fp("96000"),
        order_spec=spec,
        reference_price=fp("4000", 0),
        projected_notional_base=fp("2400000"),
    )

    decision = policy.check_order(asset_intent(FUTURE, "2", "4000"), context)

    assert decision.code == "INITIAL_MARGIN_LIMIT"


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (CrossAssetRiskLimits(max_adv_participation="0.10"), "ADV_PARTICIPATION_LIMIT"),
        (CrossAssetRiskLimits(max_days_to_liquidate="1"), "DAYS_TO_LIQUIDATE_LIMIT"),
    ],
)
def test_liquidity_limits_accept_boundary_and_reject_excess(
    limits: CrossAssetRiskLimits,
    code: str,
) -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=limits,
        inputs=PITRiskInputs(
            prices=(price(STOCK, "10"),),
            liquidity=(liquidity(STOCK, "100000"),),
        ),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )

    assert gate.check_current(intent("1000"), event_time=T0).accepted
    decision = gate.check_current(intent("1100"), event_time=T0)
    assert decision.code == code


def test_configured_stress_accepts_boundary_and_rejects_excess() -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_stress_loss_ratio="0.01"),
        inputs=PITRiskInputs(prices=(price(STOCK, "10"),)),
        stress_scenarios=(
            StressScenario(name="equity-crash", asset_class_shocks={AssetClass.EQUITY: "-0.10"}),
        ),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )

    assert gate.check_current(intent("1000"), event_time=T0).accepted
    decision = gate.check_current(intent("1100"), event_time=T0)
    assert decision.code == "STRESS_LOSS_LIMIT"


def test_reduce_only_decreases_projected_risk_without_mutating_policy() -> None:
    spec = future_spec()
    policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="0.50"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    base_context = single_position_context(
        spec,
        quantity=fp("1", 0),
        mark_price=fp("4000", 0),
        base_notional=fp("1200000"),
        nav=fp("1000000"),
        initial_margin=fp("120000"),
        maintenance_margin=fp("96000"),
        order_spec=spec,
        reference_price=fp("4000", 0),
        projected_notional_base=fp("-1200000"),
    )
    reducing = asset_intent(FUTURE, "1", "4000", side=Side.SELL, reduce_only=True)

    decisions = [policy.check_order(reducing, base_context) for _ in range(3)]
    assert decisions == [decisions[0]] * 3
    assert decisions[0].accepted
    assert policy.instruments == {FUTURE: spec}
    with pytest.raises(FrozenInstanceError):
        policy.limits = CrossAssetRiskLimits()

    increasing_context = replace(
        base_context,
        projected_notional_base=fp("1200000"),
    )
    increasing = asset_intent(FUTURE, "1", "4000")
    assert policy.check_order(increasing, increasing_context).code == "GROSS_LEVERAGE_LIMIT"


def test_reduce_only_allows_monotonic_reduction_of_existing_multi_limit_breaches() -> None:
    first = stock_spec()
    second = replace(
        first,
        instrument_id="equity:sse:600001",
        native_symbol="600001",
    )
    policy = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(
            max_gross_leverage="1.0",
            max_abs_net_leverage="1.0",
            max_instrument_concentration="0.5",
            max_asset_class_concentration={AssetClass.EQUITY: "0.9"},
            max_currency_concentration={"CNY": "0.9"},
            max_venue_concentration={"SSE": "0.9"},
            max_strategy_concentration="1.0",
            max_adv_participation="0.2",
            max_days_to_liquidate="2",
            max_stress_loss_ratio="0.05",
        ),
        inputs=PITRiskInputs(
            prices=(price(first.instrument_id, "10"), price(second.instrument_id, "10")),
            liquidity=(
                liquidity(first.instrument_id, "100000"),
                liquidity(second.instrument_id, "100000"),
            ),
            strategy_exposures=(
                StrategyExposureSnapshot(
                    strategy_id="strategy",
                    gross_exposure_base=fp("120000", 0),
                    instrument_notionals_base={
                        first.instrument_id: fp("60000", 0),
                        second.instrument_id: fp("60000", 0),
                    },
                    observed_at=T0,
                    available_at=T0,
                ),
            ),
        ),
        stress_scenarios=(
            StressScenario(name="equity-down", asset_class_shocks={AssetClass.EQUITY: "-0.10"}),
        ),
    )
    context = two_stock_context(
        first,
        second,
        first_notional="60000",
        second_notional="60000",
        projected_notional_base="-10000",
    )
    reducing = asset_intent(
        first.instrument_id,
        "1000",
        "10",
        side=Side.SELL,
        reduce_only=True,
    )

    decision = policy.check_order(reducing, context)

    assert decision.accepted


def test_reduce_only_still_requires_all_pit_inputs_and_analytics() -> None:
    first = stock_spec()
    second = replace(
        first,
        instrument_id="equity:sse:600001",
        native_symbol="600001",
    )
    context = two_stock_context(
        first,
        second,
        first_notional="60000",
        second_notional="60000",
        projected_notional_base="-10000",
    )
    reducing = asset_intent(
        first.instrument_id,
        "1000",
        "10",
        side=Side.SELL,
        reduce_only=True,
    )
    missing_price = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(prices=(price(first.instrument_id, "10"),)),
    )
    assert missing_price.check_order(reducing, context).code == "MISSING_PIT_PRICE"

    analytics_breach = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(
            max_gross_leverage="1",
            max_historical_var="0.05",
        ),
        inputs=PITRiskInputs(
            prices=(price(first.instrument_id, "10"), price(second.instrument_id, "10")),
            analytics=(
                AnalyticsRiskSnapshot(
                    historical_var=fp("0.06"),
                    historical_cvar=fp("0.06"),
                    parametric_var=fp("0.06"),
                    parametric_cvar=fp("0.06"),
                    observed_at=T0,
                    available_at=T0,
                ),
            ),
        ),
    )
    assert analytics_breach.check_order(reducing, context).code == "HISTORICAL_VAR_LIMIT"


def test_reduce_only_rejects_noop_and_any_controlled_risk_increase() -> None:
    first = stock_spec()
    second = replace(
        first,
        instrument_id="equity:sse:600001",
        native_symbol="600001",
    )
    inputs = PITRiskInputs(
        prices=(price(first.instrument_id, "10"), price(second.instrument_id, "10"))
    )
    no_op_policy = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(max_gross_leverage="2"),
        inputs=inputs,
    )
    long_context = two_stock_context(
        first,
        second,
        first_notional="60000",
        second_notional="60000",
        projected_notional_base="10000",
    )
    no_op = asset_intent(first.instrument_id, "1000", "10", reduce_only=True)
    assert no_op_policy.check_order(no_op, long_context).code == "REDUCE_ONLY_NOT_REDUCING"

    hedged_context = two_stock_context(
        first,
        second,
        first_notional="60000",
        second_notional="-60000",
        projected_notional_base="-10000",
    )
    net_policy = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(
            max_gross_leverage="2",
            max_abs_net_leverage="1",
        ),
        inputs=inputs,
    )
    reducing = asset_intent(
        first.instrument_id,
        "1000",
        "10",
        side=Side.SELL,
        reduce_only=True,
    )
    assert net_policy.check_order(reducing, hedged_context).code == "NET_LEVERAGE_LIMIT"

    stress_policy = CrossAssetRiskPolicy(
        instruments={first.instrument_id: first, second.instrument_id: second},
        limits=CrossAssetRiskLimits(max_stress_loss_ratio="1"),
        inputs=inputs,
        stress_scenarios=(
            StressScenario(
                name="same-direction-shock",
                instrument_shocks={
                    first.instrument_id: "0.10",
                    second.instrument_id: "0.10",
                },
            ),
        ),
    )
    assert stress_policy.check_order(reducing, hedged_context).code == "STRESS_LOSS_LIMIT"


def test_runtime_accepts_a_share_future_crypto_multicurrency_portfolio() -> None:
    specs = {spec.instrument_id: spec for spec in (stock_spec(), future_spec(), perp_spec())}
    positions = (
        PositionRiskSnapshot(
            instrument_id=PERP,
            asset_class=AssetClass.CRYPTO,
            venue="BINANCE",
            settlement_currency="USDT",
            quantity=fp("1.000", 3),
            mark_price=fp("20000", 0),
            base_notional=fp("10000"),
            initial_margin=fp("1000"),
            maintenance_margin=fp("500"),
        ),
        PositionRiskSnapshot(
            instrument_id=STOCK,
            asset_class=AssetClass.EQUITY,
            venue="SSE",
            settlement_currency="CNY",
            quantity=fp("100", 0),
            mark_price=fp("10", 0),
            base_notional=fp("140"),
            initial_margin=fp("0"),
            maintenance_margin=fp("0"),
        ),
        PositionRiskSnapshot(
            instrument_id=FUTURE,
            asset_class=AssetClass.FUTURE,
            venue="CFFEX",
            settlement_currency="CNY",
            quantity=fp("1", 0),
            mark_price=fp("4000", 0),
            base_notional=fp("168000"),
            initial_margin=fp("16800"),
            maintenance_margin=fp("13440"),
        ),
    )
    account = AccountSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="USD",
        positions={position.instrument_id: position.quantity for position in positions},
        nav=fp("500000"),
        initial_margin=fp("17800"),
        maintenance_margin=fp("13940"),
    )
    portfolio = PortfolioRiskSnapshot(
        account_id="account",
        event_time=T0,
        base_currency="USD",
        nav=fp("500000"),
        cash_value=fp("321860"),
        gross_exposure=fp("178140"),
        net_exposure=fp("178140"),
        initial_margin=fp("17800"),
        maintenance_margin=fp("13940"),
        positions=positions,
    )
    context = RiskCheckContext(account_snapshot=account, portfolio_snapshot=portfolio)
    policy = CrossAssetRiskPolicy(
        instruments=specs,
        limits=CrossAssetRiskLimits(
            max_gross_leverage="0.40",
            max_abs_net_leverage="0.40",
            max_instrument_concentration="0.34",
            max_asset_class_concentration={
                AssetClass.EQUITY: "0.01",
                AssetClass.FUTURE: "0.34",
                AssetClass.CRYPTO: "0.02",
            },
            max_currency_concentration={"CNY": "0.34", "USDT": "0.02"},
            max_venue_concentration={"SSE": "0.01", "CFFEX": "0.34", "BINANCE": "0.02"},
            max_initial_margin_base="17800",
            max_maintenance_margin_base="13940",
            max_margin_utilization="0.0356",
            max_days_to_liquidate="2",
            max_strategy_concentration="0.40",
            max_stress_loss_ratio="0.04",
        ),
        inputs=PITRiskInputs(
            prices=(price(STOCK, "10"), price(FUTURE, "4000"), price(PERP, "20000")),
            fx_rates=(fx("CNY", "USD", "0.14"), fx("USDT", "USD", "0.5")),
            liquidity=(
                liquidity(STOCK, "10000"),
                liquidity(FUTURE, "1000000"),
                liquidity(PERP, "100000"),
            ),
            strategy_exposures=(
                StrategyExposureSnapshot(
                    strategy_id="strategy",
                    gross_exposure_base=fp("178140"),
                    instrument_notionals_base={
                        position.instrument_id: position.base_notional for position in positions
                    },
                    observed_at=T0,
                    available_at=T0,
                ),
            ),
        ),
        stress_scenarios=(
            StressScenario(
                name="cross-asset-selloff",
                asset_class_shocks={
                    AssetClass.EQUITY: "-0.10",
                    AssetClass.FUTURE: "-0.10",
                    AssetClass.CRYPTO: "-0.20",
                },
            ),
        ),
    )

    assert policy.runtime_check(context).accepted


@pytest.mark.parametrize(
    ("limit_name", "field_name", "code"),
    [
        ("max_historical_var", "historical_var", "HISTORICAL_VAR_LIMIT"),
        ("max_historical_cvar", "historical_cvar", "HISTORICAL_CVAR_LIMIT"),
        ("max_parametric_var", "parametric_var", "PARAMETRIC_VAR_LIMIT"),
        ("max_parametric_cvar", "parametric_cvar", "PARAMETRIC_CVAR_LIMIT"),
    ],
)
def test_tail_risk_limits_accept_boundary_and_reject_excess(
    limit_name: str,
    field_name: str,
    code: str,
) -> None:
    limits = CrossAssetRiskLimits(**{limit_name: "0.05"})
    boundary = AnalyticsRiskSnapshot(
        historical_var=fp("0.05"),
        historical_cvar=fp("0.05"),
        parametric_var=fp("0.05"),
        parametric_cvar=fp("0.05"),
        observed_at=T0,
        available_at=T0,
    )
    accepted = CrossAssetRiskPolicy(
        instruments={},
        limits=limits,
        inputs=PITRiskInputs(analytics=(boundary,)),
    )
    assert accepted.runtime_check(empty_runtime_context()).accepted

    exceeded = replace(boundary, **{field_name: fp("0.06")})
    rejected = CrossAssetRiskPolicy(
        instruments={},
        limits=limits,
        inputs=PITRiskInputs(analytics=(exceeded,)),
    )
    assert rejected.runtime_check(empty_runtime_context()).code == code


def test_factor_drift_limit_accepts_boundary_and_rejects_excess() -> None:
    limits = CrossAssetRiskLimits(max_factor_drift_z="2")
    boundary = AnalyticsRiskSnapshot(
        historical_var=fp("0"),
        historical_cvar=fp("0"),
        parametric_var=fp("0"),
        parametric_cvar=fp("0"),
        factor_drift_z={"momentum": fp("2", 0)},
        observed_at=T0,
        available_at=T0,
    )
    policy = CrossAssetRiskPolicy(
        instruments={},
        limits=limits,
        inputs=PITRiskInputs(analytics=(boundary,)),
    )
    assert policy.runtime_check(empty_runtime_context()).accepted
    exceeded = replace(boundary, factor_drift_z={"momentum": fp("-2.1", 1)})
    rejected = CrossAssetRiskPolicy(
        instruments={},
        limits=limits,
        inputs=PITRiskInputs(analytics=(exceeded,)),
    )
    assert rejected.runtime_check(empty_runtime_context()).code == "FACTOR_DRIFT_LIMIT"


def test_factor_drift_limit_fails_closed_on_empty_or_incomplete_factor_universe() -> None:
    empty = AnalyticsRiskSnapshot(
        historical_var=fp("0"),
        historical_cvar=fp("0"),
        parametric_var=fp("0"),
        parametric_cvar=fp("0"),
        factor_drift_z={},
        observed_at=T0,
        available_at=T0,
    )
    empty_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_factor_drift_z="2"),
        inputs=PITRiskInputs(analytics=(empty,)),
    )
    assert empty_policy.runtime_check(empty_runtime_context()).code == "MISSING_FACTOR_DRIFT"

    incomplete = replace(empty, factor_drift_z={"momentum": fp("1")})
    required_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(
            max_factor_drift_z="2",
            required_factor_drift_factors=("momentum", "value"),
        ),
        inputs=PITRiskInputs(analytics=(incomplete,)),
    )
    assert required_policy.runtime_check(empty_runtime_context()).code == "MISSING_FACTOR_DRIFT"

    complete = replace(incomplete, factor_drift_z={"momentum": fp("1"), "value": fp("2")})
    complete_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=required_policy.limits,
        inputs=PITRiskInputs(analytics=(complete,)),
    )
    assert complete_policy.runtime_check(empty_runtime_context()).accepted


def test_pit_selection_prefers_newer_observation_over_late_stale_data() -> None:
    recent_breach = AnalyticsRiskSnapshot(
        historical_var=fp("0.06"),
        historical_cvar=fp("0"),
        parametric_var=fp("0"),
        parametric_cvar=fp("0"),
        observed_at=T0 - timedelta(minutes=1),
        available_at=T0 - timedelta(minutes=1),
    )
    late_stale = replace(
        recent_breach,
        historical_var=fp("0.01"),
        observed_at=T0 - timedelta(days=1),
        available_at=T0,
    )
    analytics_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_historical_var="0.05"),
        inputs=PITRiskInputs(analytics=(recent_breach, late_stale)),
    )
    assert analytics_policy.runtime_check(empty_runtime_context()).code == "HISTORICAL_VAR_LIMIT"

    recent_strategy_breach = StrategyExposureSnapshot(
        strategy_id="strategy",
        gross_exposure_base=fp("20000", 0),
        observed_at=T0 - timedelta(minutes=1),
        available_at=T0 - timedelta(minutes=1),
    )
    late_stale_strategy = replace(
        recent_strategy_breach,
        gross_exposure_base=fp("0"),
        observed_at=T0 - timedelta(days=1),
        available_at=T0,
    )
    strategy_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_strategy_concentration="0.10"),
        inputs=PITRiskInputs(
            strategy_exposures=(recent_strategy_breach, late_stale_strategy),
        ),
    )
    assert strategy_policy.runtime_check(empty_runtime_context()).code == (
        "STRATEGY_CONCENTRATION_LIMIT"
    )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (PITRiskInputs(), "MISSING_PIT_PRICE"),
        (
            PITRiskInputs(prices=(price(STOCK, "10", available_at=T0 + timedelta(seconds=1)),)),
            "PIT_PRICE_NOT_AVAILABLE",
        ),
    ],
)
def test_missing_or_future_price_fails_closed(inputs: PITRiskInputs, expected: str) -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=inputs,
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )
    assert gate.check_current(intent("100"), event_time=T0).code == expected


@pytest.mark.parametrize(
    ("fx_inputs", "expected"),
    [
        ((), "MISSING_PIT_FX"),
        (
            (fx("USDT", "USD", "0.5", available_at=T0 + timedelta(seconds=1)),),
            "PIT_FX_NOT_AVAILABLE",
        ),
        ((fx("USDT", "USD", "0.4"),), "PIT_FX_MISMATCH"),
    ],
)
def test_missing_future_or_mismatched_fx_fails_closed(
    fx_inputs: tuple[FxRateObservation, ...],
    expected: str,
) -> None:
    spec = perp_spec()
    policy = CrossAssetRiskPolicy(
        instruments={PERP: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="10"),
        inputs=PITRiskInputs(
            prices=(price(PERP, "20000"),),
            fx_rates=fx_inputs,
        ),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="20000",
        base_currency="USD",
        cash="100000",
        fx_to_base={"USDT": fp("0.5", 1)},
    )
    order = asset_intent(PERP, "1.000", "20000")
    assert gate.check_current(order, event_time=T0).code == expected


@pytest.mark.parametrize(
    ("liquidity_inputs", "expected"),
    [
        ((), "MISSING_PIT_ADV"),
        (
            (liquidity(STOCK, "100000", available_at=T0 + timedelta(seconds=1)),),
            "PIT_ADV_NOT_AVAILABLE",
        ),
    ],
)
def test_missing_or_future_adv_fails_closed(
    liquidity_inputs: tuple[LiquidityObservation, ...],
    expected: str,
) -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_adv_participation="0.10"),
        inputs=PITRiskInputs(
            prices=(price(STOCK, "10"),),
            liquidity=liquidity_inputs,
        ),
    )
    gate, _ = gate_for(
        spec,
        policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )
    assert gate.check_current(intent("100"), event_time=T0).code == expected


def test_missing_strategy_and_analytics_inputs_fail_closed() -> None:
    strategy_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_strategy_concentration="1"),
        inputs=PITRiskInputs(),
    )
    assert (
        strategy_policy.runtime_check(empty_runtime_context()).code == "MISSING_STRATEGY_EXPOSURE"
    )
    analytics_policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_historical_var="0.05"),
        inputs=PITRiskInputs(),
    )
    assert analytics_policy.runtime_check(empty_runtime_context()).code == "MISSING_ANALYTICS_RISK"
    future_analytics = AnalyticsRiskSnapshot(
        historical_var=fp("0"),
        historical_cvar=fp("0"),
        parametric_var=fp("0"),
        parametric_cvar=fp("0"),
        observed_at=T0,
        available_at=T0 + timedelta(seconds=1),
    )
    policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_historical_var="0.05"),
        inputs=PITRiskInputs(analytics=(future_analytics,)),
    )
    assert policy.runtime_check(empty_runtime_context()).code == "PIT_ANALYTICS_NOT_AVAILABLE"


def test_classification_and_margin_configuration_fail_closed_with_stable_codes() -> None:
    order = asset_intent(FUTURE, "1", "4000")
    empty = empty_runtime_context(nav="1000000")
    spec = future_spec()
    context = RiskCheckContext(
        account_snapshot=empty.account_snapshot,
        portfolio_snapshot=empty.portfolio_snapshot,
        instrument_spec=spec,
        reference_price=fp("4000", 0),
        projected_notional_base=fp("1200000"),
    )
    missing = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_gross_leverage="2"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    assert missing.check_order(order, context).code == "MISSING_CLASSIFICATION"

    future_classification = replace(spec, available_at=T0 + timedelta(seconds=1))
    future_policy = CrossAssetRiskPolicy(
        instruments={FUTURE: future_classification},
        limits=CrossAssetRiskLimits(max_gross_leverage="2"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    future_context = replace(context, instrument_spec=future_classification)
    assert (
        future_policy.check_order(order, future_context).code == "PIT_CLASSIFICATION_NOT_AVAILABLE"
    )

    for unavailable_spec in (
        replace(spec, effective_from=T0 + timedelta(seconds=1)),
        replace(spec, effective_to=T0),
    ):
        unavailable_policy = CrossAssetRiskPolicy(
            instruments={FUTURE: unavailable_spec},
            limits=CrossAssetRiskLimits(max_gross_leverage="2"),
            inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
        )
        unavailable_context = replace(context, instrument_spec=unavailable_spec)
        assert (
            unavailable_policy.check_order(order, unavailable_context).code
            == "INSTRUMENT_NOT_EFFECTIVE"
        )

    inverse_spec = replace(spec, inverse=True)
    inverse_policy = CrossAssetRiskPolicy(
        instruments={FUTURE: inverse_spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="2"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    assert (
        inverse_policy.check_order(order, replace(context, instrument_spec=inverse_spec)).code
        == "UNSUPPORTED_INVERSE_CONTRACT"
    )

    mismatch_policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="2"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )
    mismatched_context = replace(context, instrument_spec=replace(spec, venue="OTHER"))
    assert mismatch_policy.check_order(order, mismatched_context).code == "CLASSIFICATION_MISMATCH"

    for metadata, expected in (
        ({"initial_margin_rate": "0.1"}, "MISSING_MARGIN_RATE"),
        (
            {"initial_margin_rate": "invalid", "maintenance_margin_rate": "0.08"},
            "INVALID_MARGIN_RATE",
        ),
        (
            {"initial_margin_rate": "0.10", "maintenance_margin_rate": "0.20"},
            "INVALID_MARGIN_RATE",
        ),
    ):
        broken = future_spec(metadata=metadata)
        broken_context = replace(context, instrument_spec=broken)
        policy = CrossAssetRiskPolicy(
            instruments={FUTURE: broken},
            limits=CrossAssetRiskLimits(max_initial_margin_base="999999"),
            inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
        )
        assert policy.check_order(order, broken_context).code == expected


def test_runtime_snapshot_validation_fails_closed() -> None:
    spec = stock_spec()
    context = single_position_context(
        spec,
        quantity=fp("100", 0),
        mark_price=fp("10", 0),
        base_notional=fp("1000"),
        nav=fp("100000"),
        initial_margin=fp("0"),
        maintenance_margin=fp("0"),
    )
    no_price = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(),
    )
    assert no_price.runtime_check(context).code == "MISSING_PIT_PRICE"

    wrong_price = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(prices=(price(STOCK, "9"),)),
    )
    assert wrong_price.runtime_check(context).code == "PIT_PRICE_MISMATCH"

    mismatched_position = replace(
        context.portfolio_snapshot.positions[0],
        venue="OTHER",
    )
    mismatched_context = replace(
        context,
        portfolio_snapshot=replace(
            context.portfolio_snapshot,
            positions=(mismatched_position,),
        ),
    )
    valid_price = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(prices=(price(STOCK, "10"),)),
    )
    assert valid_price.runtime_check(mismatched_context).code == "CLASSIFICATION_MISMATCH"

    inconsistent = replace(
        context,
        portfolio_snapshot=replace(context.portfolio_snapshot, gross_exposure=fp("999")),
    )
    assert valid_price.runtime_check(inconsistent).code == "RISK_SNAPSHOT_INCONSISTENT"


def test_immutable_input_contract_validation() -> None:
    limits = CrossAssetRiskLimits(max_currency_concentration={"CNY": "1"})
    assert isinstance(limits.max_currency_concentration, MappingProxyType)
    with pytest.raises(TypeError):
        limits.max_currency_concentration["USD"] = Decimal(1)
    with pytest.raises(ValidationError, match="duplicate normalized keys"):
        CrossAssetRiskLimits(max_currency_concentration={"CNY": "1", " CNY": "0"})

    observation = price(STOCK, "10")
    with pytest.raises(ValidationError, match="duplicate PIT observations"):
        PITRiskInputs(prices=(observation, observation))

    with pytest.raises(ValidationError, match="available_at"):
        PriceObservation(
            instrument_id=STOCK,
            price=fp("10"),
            observed_at=T0,
            available_at=T0 - timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="positive FixedPoint"):
        FxRateObservation(
            currency="CNY",
            base_currency="USD",
            rate_to_base=fp("0"),
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="positive FixedPoint"):
        LiquidityObservation(
            instrument_id=STOCK,
            average_daily_value_base=fp("0"),
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="exceed gross"):
        StrategyExposureSnapshot(
            strategy_id="strategy",
            gross_exposure_base=fp("1"),
            instrument_notionals_base={STOCK: fp("2")},
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="non-negative"):
        AnalyticsRiskSnapshot(
            historical_var=fp("-0.01"),
            historical_cvar=fp("0"),
            parametric_var=fp("0"),
            parametric_cvar=fp("0"),
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="at least one shock"):
        StressScenario(name="empty")
    with pytest.raises(ValidationError, match="must not exceed 1"):
        CrossAssetRiskLimits(liquidation_participation_rate="1.1")
    with pytest.raises(ValidationError, match="stress scenarios"):
        CrossAssetRiskPolicy(
            instruments={},
            limits=CrossAssetRiskLimits(max_stress_loss_ratio="0.1"),
            inputs=PITRiskInputs(),
        )


def test_invalid_policy_configuration_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        PriceObservation(
            instrument_id="",
            price=fp("10"),
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="finite"):
        CrossAssetRiskLimits(max_gross_leverage="NaN")
    with pytest.raises(ValidationError, match="non-negative"):
        CrossAssetRiskLimits(max_gross_leverage="-1")
    with pytest.raises(ValidationError, match="positive"):
        CrossAssetRiskLimits(liquidation_participation_rate="0")
    with pytest.raises(ValidationError, match="requires max_factor_drift_z"):
        CrossAssetRiskLimits(required_factor_drift_factors=("momentum",))
    with pytest.raises(ValidationError, match="duplicate normalized"):
        CrossAssetRiskLimits(
            max_factor_drift_z="2",
            required_factor_drift_factors=("momentum", " momentum "),
        )
    with pytest.raises(ValidationError, match="must be a mapping"):
        StrategyExposureSnapshot(
            strategy_id="strategy",
            gross_exposure_base=fp("0"),
            instrument_notionals_base=[],
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="must be a FixedPoint"):
        StrategyExposureSnapshot(
            strategy_id="strategy",
            gross_exposure_base=fp("0"),
            instrument_notionals_base={STOCK: "invalid"},
            observed_at=T0,
            available_at=T0,
        )
    with pytest.raises(ValidationError, match="must be a mapping"):
        CrossAssetRiskLimits(max_currency_concentration=[])
    with pytest.raises(ValidationError, match="keys must be AssetClass"):
        CrossAssetRiskLimits(max_asset_class_concentration={"equity": "1"})
    with pytest.raises(ValidationError, match="invalid observation"):
        PITRiskInputs(prices=("invalid",))
    with pytest.raises(ValidationError, match="must be a mapping"):
        StressScenario(name="invalid", instrument_shocks=[])
    with pytest.raises(ValidationError, match="invalid key"):
        StressScenario(name="invalid", asset_class_shocks={"equity": "-0.1"})
    with pytest.raises(ValidationError, match="limits"):
        CrossAssetRiskPolicy(instruments={}, limits="invalid", inputs=PITRiskInputs())
    with pytest.raises(ValidationError, match="inputs"):
        CrossAssetRiskPolicy(
            instruments={},
            limits=CrossAssetRiskLimits(),
            inputs="invalid",
        )
    with pytest.raises(ValidationError, match="InstrumentSpec values"):
        CrossAssetRiskPolicy(
            instruments={STOCK: "invalid"},
            limits=CrossAssetRiskLimits(),
            inputs=PITRiskInputs(),
        )
    with pytest.raises(ValidationError, match="registry keys"):
        CrossAssetRiskPolicy(
            instruments={"wrong": stock_spec()},
            limits=CrossAssetRiskLimits(),
            inputs=PITRiskInputs(),
        )
    with pytest.raises(ValidationError, match="stress_scenarios"):
        CrossAssetRiskPolicy(
            instruments={},
            limits=CrossAssetRiskLimits(),
            inputs=PITRiskInputs(),
            stress_scenarios=("invalid",),
        )


def test_non_positive_nav_and_price_mismatch_are_rejected() -> None:
    policy = CrossAssetRiskPolicy(
        instruments={},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(),
    )
    assert policy.runtime_check(empty_runtime_context(nav="0")).code == "NAV_NOT_POSITIVE"

    spec = stock_spec()
    price_policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="1"),
        inputs=PITRiskInputs(prices=(price(STOCK, "9"),)),
    )
    gate, _ = gate_for(
        spec,
        price_policy,
        price_value="10",
        base_currency="CNY",
        cash="100000",
    )
    assert gate.check_current(intent("100"), event_time=T0).code == "PIT_PRICE_MISMATCH"


def test_absolute_net_leverage_rejects_short_exposure() -> None:
    spec = future_spec()
    context = single_position_context(
        spec,
        quantity=fp("-1", 0),
        mark_price=fp("4000", 0),
        base_notional=fp("-1200000"),
        nav=fp("1000000"),
        initial_margin=fp("120000"),
        maintenance_margin=fp("96000"),
    )
    policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=CrossAssetRiskLimits(max_abs_net_leverage="1"),
        inputs=PITRiskInputs(prices=(price(FUTURE, "4000"),)),
    )

    assert policy.runtime_check(context).code == "NET_LEVERAGE_LIMIT"
    assert not policy.sends_live_orders
