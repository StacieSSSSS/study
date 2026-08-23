# Parameter invariants

- Factor windows are positive integers and `fast_window < medium_window <= zscore_window`.
- FX, UST and IRS factor weights are non-negative and each asset-class block sums to 1 within `1e-6`.
- `signal_clip`, volatility target, leverage, gross limit, costs, and annualization are positive.
- `train_size`, `test_size`, and `step` are positive; the available dataset must produce at least one complete walk-forward window.
- Effectiveness is judged independently for every `instrument × factor`; changing these thresholds requires a new run id and a `DECISIONS.md` entry.
- `max_drawdown` is a negative floor; `min_sharpe` and `max_turnover` use their natural directions.
- A gate may be relaxed only when the user explicitly authorizes it and `DECISIONS.md` records the rationale.
- A changed parameter set gets a new run id. Reusing the same OOS windows for repeated tuning turns them into validation data; reserve a later untouched holdout before calling the result out-of-sample.
- For manual Excel mode, never map a Treasury yield to an IRS instrument. Missing carry/rate inputs remain unavailable until the exact series is supplied.
- Rate instruments must use `return_model: yield_duration`; positive weight means long duration/receive fixed and approximate return is `-duration × Δyield / 100` when yields are stored in percent.
