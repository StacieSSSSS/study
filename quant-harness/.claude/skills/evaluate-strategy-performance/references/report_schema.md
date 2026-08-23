# Performance report schema

- `reports/<strategy>/backtests/<run_id>/metrics.json`: holdout/full-run Sharpe, annualized return, max drawdown, and turnover.
- `reports/<strategy>/backtests/<run_id>/metrics_by_instrument.csv`: the same metrics by tradable instrument.
- `reports/<strategy>/walk_forward/<run_id>/windows.csv`: one row per signal variant and OOS window.
- `reports/<strategy>/walk_forward/<run_id>/summary.csv`: cross-window mean/worst statistics and signal-availability status.
- `reports/<strategy>/walk_forward/<run_id>/oos_daily_returns.csv.gz`: daily OOS return and gross exposure.
- `reports/<strategy>/parameters/`: immutable YAML config snapshots and factor parameter tables.
- `reports/<strategy>/data_audits/<run_id>/`: workbook schema, price quality, macro date reconciliation, and missing-input audit.

Interpretation order: data completeness and point-in-time status; number of OOS windows; worst window; average OOS; turnover/cost sensitivity; factor coverage; gate result. Do not rank a factor with zero active days as a valid zero-return strategy.
