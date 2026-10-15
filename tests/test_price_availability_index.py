from bisect import bisect_right
from datetime import datetime, timedelta, timezone

import numpy as np
from quant_data_kit import FixedPoint

from quant_risk_monitor.cross_asset import PriceObservation, _latest, _price_availability_index


def test_index_matches_full_scan_with_late_revisions_ties_and_future_data():
    rng = np.random.default_rng(500)
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    observations = []
    for asset in ("A", "B"):
        for day in range(50):
            observed = base + timedelta(days=day)
            for lag in (0, int(rng.integers(1, 20))):
                observations.append(
                    PriceObservation(
                        instrument_id=asset,
                        price=FixedPoint(100 + day + lag, 2),
                        observed_at=observed,
                        available_at=observed + timedelta(days=lag),
                    )
                )
    rng.shuffle(observations)
    indexed = _price_availability_index(observations)
    for asset in ("A", "B"):
        times, winners = indexed[asset]
        for day in range(-1, 72):
            at = base + timedelta(days=day)
            expected, status = _latest(
                observations, lambda item, selected=asset: item.instrument_id == selected, at
            )
            i = bisect_right(times, at) - 1
            assert (None if i < 0 else winners[i]) == expected
            assert ("future" if i < 0 else "ok") == status
    past = base + timedelta(days=20)
    causal_only = _price_availability_index([x for x in observations if x.available_at <= past])
    for asset in ("A", "B"):
        times, values = indexed[asset]
        assert values[bisect_right(times, past) - 1] == causal_only[asset][1][-1]
