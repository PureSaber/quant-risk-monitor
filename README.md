# quant-risk-monitor

Rule-based portfolio risk checker for backtest outputs.

## Commands

```bash
pip install -e ".[dev]"
quant-risk check --config configs/default.yaml --out state/alerts.json
pytest -q
```

## Rules

- `max_drawdown`: peak-to-trough drawdown on NAV curve
- `daily_loss`: single-day return threshold
- `single_name_weight`: max weight in holdings file

Exit code 1 when any critical alert fires.
