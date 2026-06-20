# _template

Copy this folder to `strategies/<your_strategy_name>/` to start a new strategy. Then:

1. Edit `config.yaml`: universe, walk-forward window, and gate thresholds.
2. `data/loader.py`: implement `load_raw()` to pull/clean your source data into a
   `DatetimeIndex`-ed DataFrame.
3. `signals/signal.py`: implement `generate_signal(pit, as_of)`. It must only read
   data through the `PointInTimeFrame` passed in — never the raw DataFrame — so the
   bias-check and the structural design agree on what "no look-ahead" means.
4. `backtest/run.py`: wire `load_data()`, `evaluate_window()` (used by walk-forward),
   and `main()` (full-history backtest that writes `reports/<name>/metrics.json`,
   used by the perf-gate).

Then run:

```
make bias-check STRATEGY=<your_strategy_name>
make walk-forward STRATEGY=<your_strategy_name>
python -m strategies.<your_strategy_name>.backtest.run   # writes reports/<name>/metrics.json
make perf-gate STRATEGY=<your_strategy_name>
```

or just `make verify-full STRATEGY=<your_strategy_name>` once metrics exist.
