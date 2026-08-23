---
name: update-strategy-parameters
description: Safely inspect, change, validate, and snapshot quant-harness strategy parameters. Use when the user asks to modify factor windows, weights, risk limits, transaction costs, walk-forward windows, data mappings, or performance gates in a strategy config.yaml, especially wind_macro_daily.
---

# Update Strategy Parameters

Use this workflow from the `quant-harness` repository root.

1. Read `CLAUDE.md`, the target `config.yaml`, the latest file in `reports/<strategy>/parameters/`, and the latest walk-forward summary before editing.
2. Identify whether the request changes factor, risk, execution-cost, data, walk-forward, or gate parameters. Read [parameter_schema.md](references/parameter_schema.md) for invariants.
3. Make only the requested change with `apply_patch`. Never use OOS results as an unrecorded tuning set. Never loosen `gate:` merely to make a failing strategy pass.
4. If changing `gate:` or `walk_forward:`, append an architectural decision to `DECISIONS.md` with the reason.
5. Run `python .claude/skills/update-strategy-parameters/scripts/check_config.py --strategy <name>`. Fix validation failures before proceeding.
6. Run the strategy's per-factor walk-forward command with a new `run_id`. This produces a separate immutable YAML/CSV parameter snapshot; do not overwrite a prior run identifier.
7. Report the exact before/after values, validation command, new parameter snapshot path, OOS result path, and whether performance improved only in-sample or also across untouched OOS windows.

For `wind_macro_daily`, use:

```bash
python -m strategies.wind_macro_daily.validation.run \
  --data-mode manual_excel --workbook <xlsx> --run-id <new_run_id>
```

If `make` is installed, finish with `make verify STRATEGY=<name>` or `make verify-full STRATEGY=<name>`. On Windows hosts without `make`, run the exact lint, typecheck, unit-test, bias-check, walk-forward, and gate module equivalents and disclose the fallback.
