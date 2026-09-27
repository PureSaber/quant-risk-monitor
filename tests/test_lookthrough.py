import pandas as pd

from quant_risk_monitor.lookthrough import check_fund_concentration


def test_unknown_holdings_cannot_be_reported_as_safe_diversification():
    frame = pd.DataFrame(
        [
            {
                "disclosure_id": "F",
                "fund_id": "F",
                "holding_date": "2024-01-01",
                "available_at": "2024-01-02T00:00:00Z",
                "instrument_id": "A",
                "asset_type": "security",
                "currency": "CNY",
                "weight": ".6",
                "source": "synthetic",
                "evidence_id": "F",
            }
        ]
    )
    result = check_fund_concentration({"F": 1}, frame, "2024-02-01T00:00:00Z")
    assert not result["allowed"]
    assert {x["code"] for x in result["breaches"]} == {
        "LOOKTHROUGH_UNKNOWN",
        "LOOKTHROUGH_CONCENTRATION",
    }
