# Performance report schema

- `reports/<strategy>/backtests/<run_id>/metrics.json`: holdout/full-run Sharpe, annualized return, max drawdown, and turnover.
- `reports/<strategy>/backtests/<run_id>/metrics_by_instrument.csv`: the same metrics by tradable instrument.
- `reports/<strategy>/backtests/<run_id>/metrics_by_instrument_factor.csv`: standalone full-holdout metrics for every instrument/factor pair.
- `reports/<strategy>/walk_forward/<run_id>/windows_by_instrument_factor.csv`: one row per instrument, signal variant and OOS window.
- `reports/<strategy>/walk_forward/<run_id>/factor_effectiveness.csv`: cross-window statistics, effective flag and explicit failure reason.
- `reports/<strategy>/walk_forward/<run_id>/oos_daily_returns_by_instrument_factor.csv.gz`: standalone daily OOS returns and positions.
- `reports/<strategy>/walk_forward/<run_id>/instrument_data_coverage.csv`: available and missing configured instruments.
- `reports/<strategy>/walk_forward/<run_id>/harness_status.json`: machine-readable effective pairs, gaps, open improvements and artifact paths.
- `reports/<strategy>/visualizations/<run_id>/`: one net-value/drawdown chart per instrument plus the effectiveness heatmap.
- `reports/<strategy>/parameters/`: immutable YAML config snapshots and factor parameter tables.
- `reports/<strategy>/data_audits/<run_id>/`: workbook schema, price quality, macro date reconciliation, and missing-input audit.

Interpretation order: data completeness and point-in-time status; number of OOS windows; worst window; average OOS; turnover/cost sensitivity; factor coverage; gate result. Do not rank a factor with zero active days as a valid zero-return strategy.
