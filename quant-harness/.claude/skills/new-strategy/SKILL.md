---
description: Scaffold a new quant strategy folder under strategies/ by copying strategies/_template and rewiring names/imports. Use when the user wants to start a new strategy in this harness (e.g. "new strategy called X", "scaffold a strategy for Y").
argument-hint: <strategy_name>
allowed-tools: Bash(cp:*), Bash(mkdir:*), Bash(find:*), Bash(make bias-check:*), Read, Edit, Write, Glob
---

Scaffold a new strategy named `$1` (or ask for a name if not given — it must be a valid Python identifier in snake_case, e.g. `duration_rotation`, not `Duration-Rotation`).

Run this from the `quant-harness/` root. Steps:

1. **Refuse to overwrite**: if `strategies/$1/` already exists, stop and tell the user instead of overwriting.
2. **Copy the template**: `cp -r strategies/_template strategies/$1`.
3. **Rewrite imports and names** in the copied files (search/replace `_template` → `$1` and `STRATEGY_NAME = "_template"` → `STRATEGY_NAME = "$1"`):
   - `strategies/$1/config.yaml`: `name: _template` → `name: $1`
   - `strategies/$1/signals/signal.py`: no `_template` references expected, leave as-is
   - `strategies/$1/backtest/run.py`: update the `from strategies._template...` import lines to `from strategies.$1...`, and `STRATEGY_NAME = "_template"` to `STRATEGY_NAME = "$1"`
4. **Delete the template's README.md** from the copy (`strategies/$1/README.md`) — it's instructions for starting from `_template`, not documentation for a real strategy. Ask the user for a one-line description of the trade idea and write a fresh, short `strategies/$1/README.md` with that description plus the universe.
5. **Prompt the user for the essentials** before going further (don't guess silently):
   - One-line description of the trade idea → goes in `config.yaml`'s `description:` and the new `README.md`.
   - `universe`: which instruments/series.
   - Walk-forward window (`train_size`/`test_size`/`step`) — if unsure, leave the template's defaults (252/63/63 trading days) and say so.
   - Performance gate (`min_sharpe`/`max_drawdown`/`max_turnover`) — remind the user this should be the bar they actually believe in, not a number picked to pass. Leave template defaults if they don't have a view yet, but flag that those are placeholders.
6. **Confirm the scaffold doesn't trip the bias scanner**: `make bias-check STRATEGY=$1` (should pass — the template ships clean — but confirms wiring is correct).
7. **Tell the user what's NOT done yet**: `data/loader.py`'s `load_raw()` and `signals/signal.py`'s `generate_signal()` both still raise `NotImplementedError` — those are the actual strategy logic and are the user's to write (or ask you to write, with their guidance on the signal/data source). Point them at `strategies/$1/README.md` for the next steps, and mention `/validate-strategy $1` once there's real logic in place.

Do not implement `load_raw()`/`generate_signal()` yourself unless the user explicitly describes the data source and signal logic — scaffolding and writing the actual trade idea are different tasks.
