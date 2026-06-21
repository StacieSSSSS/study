# fx_correlation

Multi-model FX correlation trading framework. Ranks every pairwise
combination of a configured FX universe by trading opportunity, using three
independent lenses, and blends them for backtesting plus a daily conviction
report. Background and design rationale: `DECISIONS.md` D004–D006.

## Structure

```
config.yaml              single source of truth: universe, windows, model params, gate
data/loader.py            yfinance fetch + same-day cache -> wide daily log-return panel
lib/                      pure math: pair enumeration, rolling correlation, hedge-ratio
                          spread + z-score, ADF cointegration test
models/                   three selection-score functions sharing one trade mechanic
  model_a_strength.py       correlation strength & stability across the 5 windows
  model_b_divergence.py     short-window vs long-window correlation breakdown
  model_c_cointegration.py  ADF-filtered cointegration (the rigorous one — correlation
                             alone doesn't justify a mean-reversion bet)
signals/signal.py         the blended weight vector for a single date (harness-shaped,
                           but returns a Series, not a float — see note below)
backtest/run.py           blended (all 3 models) backtest; load_data/evaluate_window/main
                           wired into the existing make walk-forward / perf-gate
reporting/
  model_backtests.py        each model's OWN (unblended) historical backtest, for comparing
                             which lens is actually earning its keep
  conviction.py              DAILY REFRESH ENTRYPOINT — current ranking + conviction score
  position_management.py    timing + momentum overlay — turns ranking into a 7-level action
  plots.py                   renders charts from the CSVs the above already wrote to disk
tests/                    unit tests for lib/ and models/base.py (pure math, fast)
```

## Daily refresh

```
python3 -m strategies.fx_correlation.reporting.conviction
```

Re-fetches FX data, ranks every combo under all three models as of today, and
writes:
- `reports/fx_correlation/conviction_<date>.{csv,md}` — full ranked table
- `reports/fx_correlation/conviction_latest.json` — just the top combo, for quick checks

Top of the table = highest average rank across all three models = highest conviction.
Each row's printed report also includes a `reason`: which model ranked it best
and the concrete number behind that (correlation level, divergence magnitude,
or ADF statistic/p-value), plus which other models agree (rank in their own
top quartile). `reason` is also a column in the saved CSV.

## Position management (timing + momentum)

```
python3 -m strategies.fx_correlation.reporting.position_management
```

For every combo, classifies a 7-level action — 大力买入 / 买入 / 谨慎加仓 /
持有 / 观望 / 减仓 / 获利了结 — from two signals as of today:
- **z-score level** (`entry_z`/`exit_z` in `config.yaml`'s `position_management:`):
  how stretched the spread is right now (extreme / moderate / neutral)
- **z-score momentum** (`momentum_lookback`/`momentum_threshold`): is that
  stretch already correcting (reverting) or still building (extending),
  measured over the trailing N trading days

See `lib/momentum.py`'s `ACTIONS` table for exactly which (z-bucket,
momentum-bucket) combination maps to which action. Every printed row shows
the literal parameters used and the raw z-score/momentum numbers behind the
call — nothing is a black box. Writes `reports/fx_correlation/position_management_<date>.{csv,md}`.

**Important nuance**: conviction (from `reporting/conviction.py`) and the
action here can legitimately disagree. Conviction is built from divergence/
strength signals that can lag — by the time a combo shows up as high-
conviction, the spread may have *already* reverted, in which case this report
correctly says 获利了结 (take profit), not 买入. High conviction means "this
relationship mattered recently," not "buy now" — that's exactly the gap this
overlay is for.

## Validating the strategy

```
make bias-check STRATEGY=fx_correlation
make walk-forward STRATEGY=fx_correlation        # ~3-4 min — 16 rolling IS/OOS windows
python3 -m strategies.fx_correlation.backtest.run # writes reports/fx_correlation/metrics.json
make perf-gate STRATEGY=fx_correlation
```

Or compare the three models against each other directly:

```
python3 -m strategies.fx_correlation.reporting.model_backtests
```

## Charts

After running the backtest and/or conviction report above (their CSVs are
the input — `plots.py` never re-runs a backtest itself):

```
python3 -m strategies.fx_correlation.reporting.plots
```

Writes to `reports/fx_correlation/`:
- `equity_curves.png` — blended strategy vs. each model standalone, growth of $1
- `drawdown.png` — underwater chart for the blended strategy
- `conviction_chart.png` — today's conviction ranking as a colored bar chart
- `correlation_heatmap.png` — current 12-month correlation matrix across the universe

Any chart whose input CSV is missing is skipped with a message telling you
which command to run first, rather than failing the whole batch.

Strategy-specific unit tests (not part of `make test-unit`, which only covers `core/`):

```
pytest strategies/fx_correlation/tests/
```

## Known limitations (read before trusting the Sharpe numbers)

- **No transaction costs or slippage are modeled.** Backtested Sharpe (current
  numbers: blended ~2.6, model_c alone ~2.8) is gross, not net. EM-pair
  spreads (USDKRW, USDTWD, USDCNY) in particular are not free in practice —
  treat these numbers as a screening signal, not a tradeable expectation.
  Adding a cost model is the natural next step (see Self-Heal Protocol note
  in `CLAUDE.md`: a too-good perf-gate pass is exactly the kind of result
  that should be questioned, not celebrated).
- **USDCNH has no usable yfinance history** — `USDCNY` is used as an onshore
  proxy (D004). ~14% of its daily bars are stale repeats; any combo involving
  it is noisier than the rest.
- **Hedge ratios are fixed per walk-forward window**, not re-estimated daily
  within a window — realistic for a "rebalance occasionally" strategy, less
  so for one that claims to react to fast-moving regime shifts.

## Extending the framework

To add a new lens (e.g. a macro-timing overlay or a different z-score
weighting):
1. Add a `models/model_d_<name>.py` with a `selection_score(metrics: ComboMetrics, cfg: dict) -> float`
   and a `NEED_ADF: bool`.
2. Add its config block to `config.yaml` and a blend weight under `blend:`.
3. Wire it into `_model_specs()` in `backtest/run.py` and `MODEL_SPECS` in
   `reporting/conviction.py` / `reporting/model_backtests.py`.

Everything else — hedge ratio fitting, point-in-time correlation/z-score
computation, position sizing, rank aggregation — is shared via `models/base.py`
and needs no changes.

## Note: signal interface deviates from `_template`

`_template/signals/signal.py` returns a single float (one instrument). This
strategy trades a basket simultaneously, so `signals/signal.py::generate_weights`
returns a `pd.Series` of weights across `config.yaml`'s `pairs` instead. The
look-ahead constraint is unchanged — every read still goes through
`PointInTimeFrame.as_of()`.
