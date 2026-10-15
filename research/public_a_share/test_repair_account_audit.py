from decimal import Decimal

import pandas as pd

from repair_account_audit import exact_quote


def test_independent_quote_keeps_floor_commission_tax_change_and_tick_rounding():
    assert exact_quote(0, Decimal("10"), pd.Timestamp("2024-01-01")) == (0, 0)
    movement, cost = exact_quote(100, Decimal("10"), pd.Timestamp("2024-01-01"))
    assert movement == Decimal("-1006") and cost == Decimal("6")
    before, _ = exact_quote(-135, Decimal("10"), pd.Timestamp("2023-08-25"))
    after, _ = exact_quote(-135, Decimal("10"), pd.Timestamp("2023-08-28"))
    assert after - before == Decimal(".674325")
