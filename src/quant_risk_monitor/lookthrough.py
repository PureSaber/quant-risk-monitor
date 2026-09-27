"""Fail-closed concentration checks on disclosed and unknown fund exposures."""

from quant_data_kit.financial.common import number
from quant_data_kit.financial.holdings import exposure_summary, look_through


def check_fund_concentration(
    weights, disclosures, at, *, max_security_weight="0.1", max_unknown_weight="0", **coverage
):
    limit = number(max_security_weight, nonnegative=True)
    unknown_limit = number(max_unknown_weight, nonnegative=True)
    if limit > 1 or unknown_limit > 1:
        raise ValueError("weight limits cannot exceed one")
    summary = exposure_summary(look_through(disclosures, weights, at, **coverage))
    breaches = [
        {**x, "code": "LOOKTHROUGH_CONCENTRATION"}
        for x in summary["exposures"]
        if x["weight"] > limit and not x["instrument_id"].startswith("CASH:")
    ]
    if summary["unknown_weight"] > unknown_limit:
        breaches.append({"code": "LOOKTHROUGH_UNKNOWN", "weight": summary["unknown_weight"]})
    return {
        "allowed": not breaches,
        "breaches": breaches,
        "coverage": summary,
        "scope": "disclosed_holdings_not_realtime_positions",
    }
