"""Backtests each of the three models *independently* (not blended) so their
historical Sharpe/drawdown/turnover can be compared — this is the evidence
behind each model's contribution to the conviction score, separate from the
single blended metric that `make perf-gate` checks.
"""

from __future__ import annotations

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

MODEL_SPECS = {
    "model_a_strength": (model_a_strength.selection_score, model_a_strength.NEED_ADF, "model_a"),
    "model_b_divergence": (model_b_divergence.selection_score, model_b_divergence.NEED_ADF, "model_b"),
    "model_c_cointegration": (
        model_c_cointegration.selection_score, model_c_cointegration.NEED_ADF, "model_c"
    ),
}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _model_cfg(config: dict, model_key: str) -> dict:
    cfg = dict(config[model_key])
    if model_key == "model_b":
        cfg["short_window"] = config["short_window"]
        cfg["baseline_window"] = config["baseline_window"]
    return cfg


def backtest_one_model(
    model_name: str, config: dict, data: pd.DataFrame
) -> tuple[dict[str, float], pd.Series]:
    score_fn, need_adf, model_key = MODEL_SPECS[model_name]
    model_cfg = _model_cfg(config, model_key)
    universe = config["pairs"]

    train_size = config["walk_forward"]["train_size"]
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]
    if test_df.empty:
        raise RuntimeError(f"Not enough history to reserve {train_size} training days for {model_name}.")

    pit = PointInTimeFrame(data)
    test_dates = cast(pd.DatetimeIndex, test_df.index)
    weights, _ = run_over_dates(
        pit, test_dates, universe, train_df, config["windows"], config["zscore_lookback"],
        config[model_key]["top_k"], config[model_key]["entry_z"], config[model_key]["exit_z"],
        score_fn, model_cfg, need_adf,
    )
    lagged_weights = weights.shift(1).fillna(0.0)
    strategy_returns = (lagged_weights * data.loc[test_dates]).sum(axis=1)
    return summarize(strategy_returns, weights), strategy_returns


def run_all() -> pd.DataFrame:
    config = _load_config()
    data = load_raw()

    rows = []
    returns_by_model = {}
    for model_name in MODEL_SPECS:
        metrics, strategy_returns = backtest_one_model(model_name, config, data)
        rows.append({"model": model_name, **metrics})
        returns_by_model[model_name] = strategy_returns

    table = pd.DataFrame(rows).set_index("model")

    out_dir = Path("reports") / STRATEGY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_json(out_dir / "model_comparison.json", orient="index", indent=2)
    with open(out_dir / "model_comparison.md", "w") as f:
        f.write(table.to_markdown())
    # Persisted alongside backtest/run.py's blended_returns.csv so plots.py
    # can render comparison charts without re-running these backtests.
    pd.DataFrame(returns_by_model).to_csv(out_dir / "model_returns.csv")
    return table


if __name__ == "__main__":
    comparison = run_all()
    print(comparison.to_string())
