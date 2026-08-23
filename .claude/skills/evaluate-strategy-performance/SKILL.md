---
name: evaluate-strategy-performance
description: Run and interpret quant-harness data audits, backtests, per-factor walk-forward validation, look-ahead checks, and performance gates. Use when the user asks to backtest a strategy, refresh performance output, compare indicators, assess out-of-sample robustness, or decide whether a strategy is ready for further research or live promotion.
---

# Evaluate Strategy Performance

Use this workflow from the `quant-harness` repository root.

1. Read `CLAUDE.md`, the strategy `README.md`, `config.yaml`, and [report_schema.md](references/report_schema.md).
2. For manual Excel mode, run `python -m strategies.wind_macro_daily.data.profile_workbook --workbook <xlsx> --run-id <id>` first. Stop on schema mismatch; do not silently remap columns.
3. Run the backtest with the same run id, then run the strategy-specific walk-forward validator. For `wind_macro_daily`:

```bash
python -m strategies.wind_macro_daily.backtest.run \
  --data-mode manual_excel --workbook <xlsx> --run-id <id>
python -m strategies.wind_macro_daily.validation.run \
  --data-mode manual_excel --workbook <xlsx> --run-id <id>
```

4. Run bias scanning and the performance gate. Prefer `make bias-check STRATEGY=<name>` and `make perf-gate STRATEGY=<name>`. If `make` is unavailable, run `python -m core.validation.bias_check strategies/<name>` and `python -m core.reporting.gate --strategy=<name>`.
5. Run `python .claude/skills/evaluate-strategy-performance/scripts/summarize_results.py --strategy <name> --run-id <id>` to create a compact evidence summary.
6. Judge robustness from OOS windows, not the full-sample curve: report average and worst-window Sharpe, positive-window fraction, worst drawdown, turnover, active days, factor availability, and gate outcome.
7. Separate three conclusions: code/data validation, statistical OOS evidence, and production readiness. A failed gate is a research finding, not a code bug; never alter the gate to force a pass.

For manual macro data, always repeat that release timing is guarded but revision leakage remains until historical vintages are supplied. Do not present research-only macro performance as live-ready.
