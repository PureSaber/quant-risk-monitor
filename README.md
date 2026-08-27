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
- historical VaR/CVaR with configurable limits
- shrinkage covariance and component risk contribution
- named stress scenarios and worst loss
- liquidity days-to-exit at a maximum participation rate
- portfolio factor exposures

`advanced.positions` supplies the latest standard position snapshot. Optional `advanced.returns`,
`asset_returns`, `scenarios`, `liquidity`, and `factor_exposures` CSV inputs add machine-readable
metrics to the same alert JSON. See `configs/advanced.example.yaml`.

Exit code 1 when any critical alert fires.
