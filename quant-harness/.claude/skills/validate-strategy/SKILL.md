---
description: Run the full quant-harness validation chain for one strategy — bias-check, walk-forward, backtest run, perf-gate — and report a single consolidated pass/fail summary. Use when the user wants to validate, check, or test a strategy end-to-end (e.g. "validate duration_rotation", "is X ready", "run the full check on Y").
argument-hint: <strategy_name>
allowed-tools: Bash(make bias-check:*), Bash(make walk-forward:*), Bash(make perf-gate:*), Bash(python3 -m strategies.*.backtest.run), Read
---

Run the complete validation chain for strategy `$1`, in this exact order, from the `quant-harness/` root. Stop and report at the first failing stage rather than continuing — each stage assumes the previous one passed.

1. **Bias check**: `make bias-check STRATEGY=$1`. If it fails, report the exact findings (file:line:message) and stop — do not proceed to walk-forward on code with a known look-ahead bias finding.
2. **Walk-forward validation**: `make walk-forward STRATEGY=$1`. Report the number of windows and average out-of-sample Sharpe. If it fails (e.g. "no windows produced"), report the likely cause (data length vs. `train_size`/`test_size`/`step` in `config.yaml`) and stop.
3. **Run the full-history backtest** to refresh `reports/$1/metrics.json`: `python3 -m strategies.$1.backtest.run`. This is a required step before perf-gate — `perf-gate` only reads whatever `metrics.json` already contains, so skipping this would gate on stale numbers.
4. **Performance gate**: `make perf-gate STRATEGY=$1`. Report the metrics and the configured thresholds (from `strategies/$1/config.yaml`'s `gate:` section) side by side.

End with one consolidated summary table: stage → pass/fail → key number. If everything passes, say so plainly — don't add hedging caveats beyond what the numbers show.

**Do not loosen `gate:` thresholds in `config.yaml` to make a failing perf-gate pass.** A perf-gate failure almost always means the strategy doesn't clear the bar it was given, out of sample — that's a finding to report to the user, not a bug to patch. Only edit thresholds if the user explicitly asks to, and tell them to log the reasoning in `DECISIONS.md` if they do.
