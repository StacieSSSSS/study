"""Blended backtest entrypoint — the harness-standard surface for fx_correlation.

`evaluate_window()` and `main()` blend all three models (weighted by
`config.yaml`'s `blend:` section) into a single weight vector over the
underlying FX pairs, so the resulting Sharpe/drawdown/turnover plug directly
into the existing `core.validation.walk_forward` and `core.reporting.gate`
machinery with no changes to core/. Each model's *own* (unblended) backtest —
used to judge individual model robustness for the conviction report — lives
in `reporting/model_backtests.py` instead, since `perf-gate`'s schema is
deliberately single-strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from strategies.fx_correlation.data.loader import load_raw
from strategies.fx_correlation.models import model_a_strength, model_b_divergence, model_c_cointegration
from strategies.fx_correlation.models.base import run_over_dates

STRATEGY_NAME = "fx_correlation"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data() -> pd.DataFrame:
    return load_raw()


def _model_specs(config: dict) -> list[tuple]:
    """(selection_score_fn, model_cfg, need_adf, top_k, entry_z, exit_z, blend_weight) per model."""
    return [
        (
            model_a_strength.selection_score,
            config["model_a"],
            model_a_strength.NEED_ADF,
            config["model_a"]["top_k"],
            config["model_a"]["entry_z"],
            config["model_a"]["exit_z"],
            config["blend"]["model_a"],
        ),
        (
            model_b_divergence.selection_score,
            {
                **config["model_b"],
                "short_window": config["short_window"],
                "baseline_window": config["baseline_window"],
            },
            model_b_divergence.NEED_ADF,
            config["model_b"]["top_k"],
            config["model_b"]["entry_z"],
            config["model_b"]["exit_z"],
            config["blend"]["model_b"],
        ),
        (
            model_c_cointegration.selection_score,
            config["model_c"],
            model_c_cointegration.NEED_ADF,
            config["model_c"]["top_k"],
            config["model_c"]["entry_z"],
            config["model_c"]["exit_z"],
            config["blend"]["model_c"],
        ),
    ]


def blended_weights(
    pit: PointInTimeFrame,
    dates: pd.DatetimeIndex,
    universe: list[str],
    train_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=dates, columns=universe)
    for score_fn, model_cfg, need_adf, top_k, entry_z, exit_z, blend_weight in _model_specs(config):
        model_weights, _ = run_over_dates(
            pit, dates, universe, train_df, config["windows"], config["zscore_lookback"],
            top_k, entry_z, exit_z, score_fn, model_cfg, need_adf,
        )
        total = total.add(model_weights * blend_weight, fill_value=0.0)

    max_gross = config["blend"]["max_gross_exposure"]
    gross = total.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0).fillna(1.0)
    return total.mul(scale, axis=0)


def _portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    # Position taken at the close on a given date earns the *next* period's return.
    lagged_weights = weights.shift(1).fillna(0.0)
    return (lagged_weights * returns.loc[dates]).sum(axis=1)


def evaluate_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    config = _load_config()
    universe = config["pairs"]
    history = pd.concat([train_df, test_df]).sort_index()
    pit = PointInTimeFrame(history)
    test_dates = cast(pd.DatetimeIndex, test_df.index)

    weights = blended_weights(pit, test_dates, universe, train_df, config)
    strategy_returns = _portfolio_returns(weights, history, test_dates)
    return summarize(strategy_returns, weights)


def main() -> None:
    config = _load_config()
    universe = config["pairs"]
    data = load_data()

    train_size = config["walk_forward"]["train_size"]
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]
    if test_df.empty:
        raise RuntimeError(
            f"Not enough history ({len(data)} rows) to reserve {train_size} training days "
            "and still have a test window."
        )

    pit = PointInTimeFrame(data)
    test_dates = cast(pd.DatetimeIndex, test_df.index)
    weights = blended_weights(pit, test_dates, universe, train_df, config)
    strategy_returns = _portfolio_returns(weights, data, test_dates)
    metrics = summarize(strategy_returns, weights)

    out_dir = Path("reports") / STRATEGY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # Persisted separately from metrics.json so reporting/plots.py can render
    # charts without re-running the multi-minute backtest just to get the curve.
    strategy_returns.rename("return").to_csv(out_dir / "blended_returns.csv", header=True)
    print(f"wrote {out_dir / 'metrics.json'}: {metrics}")


if __name__ == "__main__":
    main()
