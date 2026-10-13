import pytest
from quant_data_kit.exceptions import ValidationError

from quant_risk_monitor.decision import DecisionPortfolioLimits


@pytest.mark.parametrize("mapping", [{"max_single_weigth": 0.1}, {"max_positions": 3, "typo": 2}])
def test_unknown_limits_never_disable_a_requested_rule(mapping):
    with pytest.raises(ValidationError, match="Unknown portfolio limit"):
        DecisionPortfolioLimits.from_mapping(mapping)
