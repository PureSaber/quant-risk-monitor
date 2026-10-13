"""Load the project's exact numerical modules without unrelated service startup."""

import importlib
import os
import sys
import types
from pathlib import Path

candidate = Path(__file__).resolve().parents[2] / "src/quant_risk_monitor"
fallback = (
    Path(__file__).resolve().parent.parent
    / "barra-upgrade-20261013/quant-risk-monitor/src/quant_risk_monitor"
)
SOURCE = Path(
    os.environ.get("QUANT_RISK_SOURCE", candidate if candidate.exists() else fallback)
).resolve()
if not (SOURCE / "factor_model.py").is_file():
    raise FileNotFoundError("Set QUANT_RISK_SOURCE to the project numerical module directory")
if "quant_risk_monitor" not in sys.modules:
    package = types.ModuleType("quant_risk_monitor")
    package.__path__ = [str(SOURCE)]
    package.__file__ = str(SOURCE / "__init__.py")
    sys.modules["quant_risk_monitor"] = package

factor_model = importlib.import_module("quant_risk_monitor.factor_model")
exposure_builder = importlib.import_module("quant_risk_monitor.exposure_builder")
risk_estimators = importlib.import_module("quant_risk_monitor.risk_estimators")
FactorModelConfig = factor_model.FactorModelConfig
AssetReturnObservation = factor_model.AssetReturnObservation
fit_barra_style_risk_model = factor_model.fit_barra_style_risk_model
ExposureSnapshot = factor_model.ExposureSnapshot
public_equity_style_config = exposure_builder.public_equity_style_config
build_exposure_snapshot = exposure_builder.build_exposure_snapshot
solve_factor_returns = risk_estimators.solve_factor_returns
estimate_covariance = risk_estimators.estimate_covariance
estimate_specific_risk = risk_estimators.estimate_specific_risk
regime_multiplier = risk_estimators.regime_multiplier
