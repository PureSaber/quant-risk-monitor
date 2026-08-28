from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from quant_data_kit import (
    AssetClass,
    BarEvent,
    FixedPoint,
    FundingRateEvent,
    InstrumentSpec,
    MarginMode,
    MarkPriceEvent,
    StatusEvent,
)
from quant_execution import (
    BarMatchingModel,
    DeterministicBroker,
    DeterministicRunEngine,
    ExactAccountLedger,
    LedgerEventType,
    OrderIntent,
    OrderStatus,
    OrderType,
    RuleBookRiskGate,
    Side,
    TimeInForce,
)

from quant_risk_monitor.cross_asset import (
    AnalyticsRiskSnapshot,
    CrossAssetRiskLimits,
    CrossAssetRiskPolicy,
    PITRiskInputs,
    PriceObservation,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 2, 1, 0, tzinfo=UTC)
STOCK = "equity:sse:600000"
FUTURE = "future:cffex:IF2603"
PERP = "crypto:binance:BTCUSDT-PERP"


def fp(value: str | int, scale: int | None = None) -> FixedPoint:
    text = str(value)
    resolved = len(text.split(".")[1]) if scale is None and "." in text else (scale or 0)
    return FixedPoint.from_decimal(Decimal(text), resolved)


def instrument(
    instrument_id: str,
    *,
    asset_class: AssetClass,
    product_type: str,
    venue: str,
    settlement_currency: str,
    multiplier: str = "1",
    quantity_step: str = "1",
    margin_mode: MarginMode = MarginMode.NONE,
    metadata: dict[str, str] | None = None,
) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=instrument_id,
        asset_class=asset_class,
        product_type=product_type,
        venue=venue,
        native_symbol=instrument_id,
        settlement_currency=settlement_currency,
        price_tick=fp("0.01"),
        quantity_step=fp(quantity_step),
        contract_multiplier=fp(multiplier),
        calendar_id=f"{venue}-CALENDAR",
        margin_mode=margin_mode,
        effective_from=T0 - timedelta(days=365),
        available_at=T0 - timedelta(days=365),
        metadata=metadata or {},
    )


def stock_spec() -> InstrumentSpec:
    return instrument(
        STOCK,
        asset_class=AssetClass.EQUITY,
        product_type="a_share",
        venue="SSE",
        settlement_currency="CNY",
        metadata={"lot_size": "100", "commission_rate": "0.0003"},
    )


def future_spec() -> InstrumentSpec:
    return instrument(
        FUTURE,
        asset_class=AssetClass.FUTURE,
        product_type="index_future",
        venue="CFFEX",
        settlement_currency="CNY",
        multiplier="300",
        margin_mode=MarginMode.CROSS,
        metadata={
            "initial_margin_rate": "0.10",
            "maintenance_margin_rate": "0.08",
            "fee_rate": "0",
            "close_today_fee_rate": "0",
        },
    )


def perp_spec() -> InstrumentSpec:
    return instrument(
        PERP,
        asset_class=AssetClass.CRYPTO,
        product_type="linear_perpetual",
        venue="BINANCE",
        settlement_currency="USDT",
        quantity_step="0.001",
        margin_mode=MarginMode.CROSS,
        metadata={
            "min_quantity": "0.001",
            "initial_margin_rate": "0.10",
            "maintenance_margin_rate": "0.05",
            "maker_fee_rate": "0",
            "taker_fee_rate": "0",
        },
    )


def event_fields(event_id: str, instrument_id: str, seconds: int) -> dict[str, object]:
    at = T0 + timedelta(seconds=seconds)
    return {
        "event_id": event_id,
        "instrument_id": instrument_id,
        "event_time": at,
        "received_at": at,
        "available_at": at,
        "source": "fixture",
        "trading_day": date(2026, 1, 2),
        "session_id": "fixture:2026-01-02",
        "sequence": seconds,
    }


def bar(event_id: str, instrument_id: str, seconds: int, value: str) -> BarEvent:
    at = T0 + timedelta(seconds=seconds)
    price = fp(value)
    return BarEvent(
        **event_fields(event_id, instrument_id, seconds),
        bar_start=at - timedelta(minutes=1),
        bar_end=at,
        open_price=price,
        high_price=price,
        low_price=price,
        close_price=price,
        volume=fp("1000000"),
        is_complete=True,
    )


def price_observation(
    instrument_id: str,
    value: str,
    seconds: int,
) -> PriceObservation:
    at = T0 + timedelta(seconds=seconds)
    return PriceObservation(
        instrument_id=instrument_id,
        price=fp(value),
        observed_at=at,
        available_at=at,
    )


def analytics(value: str, seconds: int) -> AnalyticsRiskSnapshot:
    at = T0 + timedelta(seconds=seconds)
    risk = fp(value)
    return AnalyticsRiskSnapshot(
        historical_var=risk,
        historical_cvar=risk,
        parametric_var=risk,
        parametric_cvar=risk,
        observed_at=at,
        available_at=at,
    )


@dataclass(frozen=True)
class Signal:
    instrument_id: str
    quantity: FixedPoint
    price: FixedPoint


class FixtureStrategy:
    def __init__(self, signals: dict[str, Signal]) -> None:
        self.signals = dict(signals)

    def on_event(self, context, event):
        signal = self.signals.get(event.event_id)
        if signal is None:
            return ()
        return (
            OrderIntent(
                idempotency_key=f"{context.run_id}:{event.event_id}",
                account_id=context.account_id,
                strategy_id=context.strategy_id,
                instrument_id=signal.instrument_id,
                side=Side.BUY,
                quantity=signal.quantity,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                created_at=event.available_at,
                limit_price=signal.price,
            ),
        )


def engine_for(
    spec: InstrumentSpec,
    policy: CrossAssetRiskPolicy,
    strategy: FixtureStrategy,
    *,
    initial_cash: str,
) -> DeterministicRunEngine:
    ledger = ExactAccountLedger(
        account_id="account",
        base_currency=spec.settlement_currency,
        instruments={spec.instrument_id: spec},
        initial_cash={spec.settlement_currency: fp(initial_cash)},
    )
    return DeterministicRunEngine(
        run_id=f"risk:{spec.instrument_id}",
        account_id="account",
        strategy_id="strategy",
        strategy=strategy,
        broker=DeterministicBroker(),
        risk_gate=RuleBookRiskGate(
            instruments={spec.instrument_id: spec},
            ledger=ledger,
            policies=[policy],
        ),
        matching_model=BarMatchingModel({spec.instrument_id: spec}, participation_rate="1"),
        ledger=ledger,
    )


def test_engine_emits_rejected_order_event_and_risk_event() -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="0.10"),
        inputs=PITRiskInputs(prices=(price_observation(STOCK, "10", 60),)),
    )
    strategy = FixtureStrategy({"signal": Signal(STOCK, fp("1100"), fp("10"))})
    engine = engine_for(spec, policy, strategy, initial_cash="100000")

    result = engine.replay([bar("signal", STOCK, 60, "10")], seed=7)

    assert result.order_count == 1
    assert engine.artifacts is not None
    assert engine.artifacts.order_events[-1].to_status is OrderStatus.REJECTED
    assert "GROSS_LEVERAGE_LIMIT" in engine.artifacts.order_events[-1].reason
    assert engine.artifacts.risk_events == (
        "risk:equity:sse:600000:signal:GROSS_LEVERAGE_LIMIT:gross leverage exceeds limit",
    )


def test_engine_boundary_acceptance_and_three_replays_are_deterministic() -> None:
    spec = stock_spec()
    policy = CrossAssetRiskPolicy(
        instruments={STOCK: spec},
        limits=CrossAssetRiskLimits(max_gross_leverage="0.10"),
        inputs=PITRiskInputs(prices=(price_observation(STOCK, "10", 60),)),
    )
    strategy = FixtureStrategy({"signal": Signal(STOCK, fp("1000"), fp("10"))})
    engine = engine_for(spec, policy, strategy, initial_cash="100000")

    results = []
    for _ in range(3):
        result = engine.replay([bar("signal", STOCK, 60, "10")], seed=7)
        assert engine.artifacts is not None
        results.append(
            (
                result.result_sha256,
                result.event_sha256,
                result.ledger_sha256,
                engine.artifacts.risk_events,
                engine.artifacts.order_events[-1].to_status,
            )
        )
    assert results[0] == results[1] == results[2]
    assert results[0][-1] is OrderStatus.ACCEPTED


def test_funding_triggers_runtime_policy_through_real_engine() -> None:
    spec = perp_spec()
    policy = CrossAssetRiskPolicy(
        instruments={PERP: spec},
        limits=CrossAssetRiskLimits(max_historical_var="0.05"),
        inputs=PITRiskInputs(
            prices=(
                price_observation(PERP, "100", 60),
                price_observation(PERP, "100", 120),
            ),
            analytics=(analytics("0.01", 60), analytics("0.06", 180)),
        ),
    )
    strategy = FixtureStrategy({"signal": Signal(PERP, fp("1.000"), fp("100"))})
    engine = engine_for(spec, policy, strategy, initial_cash="100000")
    funding = FundingRateEvent(
        **event_fields("funding", PERP, 180),
        rate=0.01,
        interval_start=T0,
        interval_end=T0 + timedelta(hours=8),
    )

    engine.replay(
        [bar("signal", PERP, 60, "100"), bar("fill", PERP, 120, "100"), funding],
        seed=11,
    )

    assert engine.artifacts is not None
    assert any(
        transaction.event_type is LedgerEventType.FUNDING
        for transaction in engine.artifacts.ledger_transactions
    )
    assert any(
        event.startswith("funding:HISTORICAL_VAR_LIMIT") for event in engine.artifacts.risk_events
    )


def test_mark_and_settlement_trigger_runtime_policy_through_real_engine() -> None:
    spec = future_spec()
    policy = CrossAssetRiskPolicy(
        instruments={FUTURE: spec},
        limits=CrossAssetRiskLimits(max_historical_var="0.05"),
        inputs=PITRiskInputs(
            prices=(
                price_observation(FUTURE, "4000", 60),
                price_observation(FUTURE, "4000", 120),
                price_observation(FUTURE, "4010", 180),
            ),
            analytics=(
                analytics("0.01", 60),
                analytics("0.06", 180),
                analytics("0.01", 181),
                analytics("0.06", 240),
            ),
        ),
    )
    strategy = FixtureStrategy({"signal": Signal(FUTURE, fp("1"), fp("4000"))})
    engine = engine_for(spec, policy, strategy, initial_cash="1000000")
    mark = MarkPriceEvent(**event_fields("mark", FUTURE, 180), price=fp("4010"))
    settlement = StatusEvent(
        **event_fields("settlement", FUTURE, 240),
        status="daily_settlement",
        reason="fixture close",
    )

    engine.replay(
        [
            bar("signal", FUTURE, 60, "4000"),
            bar("fill", FUTURE, 120, "4000"),
            mark,
            settlement,
        ],
        seed=13,
    )

    assert engine.artifacts is not None
    assert len(engine.artifacts.settlements) == 1
    assert any(
        event.startswith("mark:HISTORICAL_VAR_LIMIT") for event in engine.artifacts.risk_events
    )
    assert any(
        event.startswith("settlement:HISTORICAL_VAR_LIMIT")
        for event in engine.artifacts.risk_events
    )
