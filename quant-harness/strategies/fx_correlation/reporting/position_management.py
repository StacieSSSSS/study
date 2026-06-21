"""Position management: a timing + momentum overlay on top of the conviction
ranking, telling you not just "which combos are interesting" but "what to do
about each one right now."

For every combo, classifies a 7-level action (大力买入/买入/谨慎加仓/持有/
观望/减仓/获利了结) from two signals computed as of today:
- the combo's current z-score level — how stretched the spread is (the same
  value signal the models already trade on)
- that z-score's own momentum over the trailing `momentum_lookback` days —
  is the stretch already correcting, or still building

Run with: `python3 -m strategies.fx_correlation.reporting.position_management`
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from core.data.point_in_time import PointInTimeFrame
from strategies.fx_correlation.data.loader import load_raw
from strategies.fx_correlation.lib.momentum import (
    classify_action,
    momentum_bucket,
    reversion_momentum,
    zscore_bucket,
)
from strategies.fx_correlation.models.base import all_combos, compute_combo_metrics, fit_hedge_ratios
from strategies.fx_correlation.reporting.conviction import build_conviction_table, score_today

STRATEGY_NAME = "fx_correlation"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
LIVE_FIT_WINDOW = "12m"  # same convention as reporting/conviction.py

ACTION_EXPLANATIONS = {
    "大力买入": "价差已显著偏离均值，且已经开始向均值回归——时机和幅度都到位",
    "买入": "价差偏离均值且处于极端区间但走势平稳，或处于中等区间且回归已确认",
    "谨慎加仓": "价差已是极端水平，但仍在继续背离——可能进一步走极端，控制加仓节奏",
    "持有": "价差处于中等区间且走势平稳——维持现有仓位，不新增也不减",
    "观望": "信号不够清晰（中性区间且无动量，或中等区间仍在背离）——暂不操作",
    "减仓": "价差已回到中性区间但又开始背离——没有边际优势，降低暴露",
    "获利了结": "价差已回归至均值附近——此前基于背离建立的仓位可以兑现",
}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _lagged_date(data: pd.DataFrame, as_of: pd.Timestamp, lookback: int) -> pd.Timestamp:
    """The trading date `lookback` rows before `as_of` in `data`'s index."""
    position = data.index.get_loc(as_of)
    lagged_position = max(0, cast(int, position) - lookback)
    return cast(pd.Timestamp, data.index[lagged_position])


def compute_position_actions(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    """One row per combo: today's z-score, its momentum, and the resulting action."""
    universe = config["pairs"]
    combos = all_combos(universe)
    as_of = cast(pd.Timestamp, data.index[-1])

    fit_window_days = config["windows"][LIVE_FIT_WINDOW]
    train_df = data.loc[:as_of].tail(fit_window_days)
    betas = fit_hedge_ratios(train_df, combos)
    pit = PointInTimeFrame(data)

    pm_cfg = config["position_management"]
    lagged_as_of = _lagged_date(data, as_of, pm_cfg["momentum_lookback"])
    view_now = pit.as_of(as_of)
    view_lagged = pit.as_of(lagged_as_of)

    rows = []
    for pair1, pair2 in combos:
        beta = betas[(pair1, pair2)]
        metrics_now = compute_combo_metrics(
            view_now, pair1, pair2, beta, config["windows"], config["zscore_lookback"], need_adf=False
        )
        metrics_lagged = compute_combo_metrics(
            view_lagged, pair1, pair2, beta, config["windows"], config["zscore_lookback"], need_adf=False
        )
        momentum = reversion_momentum(metrics_now.zscore, metrics_lagged.zscore)
        action = classify_action(
            metrics_now.zscore, momentum, pm_cfg["entry_z"], pm_cfg["exit_z"], pm_cfg["momentum_threshold"]
        )
        rows.append({
            "pair1": pair1,
            "pair2": pair2,
            "zscore": metrics_now.zscore,
            "zscore_bucket": zscore_bucket(metrics_now.zscore, pm_cfg["entry_z"], pm_cfg["exit_z"]),
            "momentum": momentum,
            "momentum_bucket": momentum_bucket(momentum, pm_cfg["momentum_threshold"]),
            "action": action,
        })
    return pd.DataFrame(rows)


def _params_text(config: dict) -> str:
    pm_cfg = config["position_management"]
    return (
        f"entry_z={pm_cfg['entry_z']}, exit_z={pm_cfg['exit_z']}, "
        f"momentum_lookback={pm_cfg['momentum_lookback']}个交易日, "
        f"momentum_threshold={pm_cfg['momentum_threshold']}"
    )


def render_report(table: pd.DataFrame, config: dict, top_n: int) -> str:
    top = table.head(top_n).copy()
    lines = [f"参数: {_params_text(config)}", ""]
    for _, row in top.iterrows():
        lines.append(
            f"{row['pair1']}-{row['pair2']}  [{row['action']}]  (conviction {row['conviction']} "
            f"{row['conviction_score']:.1f})\n"
            f"    z={row['zscore']:.2f}({row['zscore_bucket']})  "
            f"momentum={row['momentum']:.2f}({row['momentum_bucket']})\n"
            f"    -> {ACTION_EXPLANATIONS[cast(str, row['action'])]}"
        )
    return "\n".join(lines)


def run(refresh: bool = True) -> pd.DataFrame:
    config = _load_config()
    data = load_raw(refresh=refresh)

    # Rank by conviction first so the report surfaces actions for the combos
    # already worth caring about, in that order, rather than all 21 combos
    # in arbitrary order.
    model_tables = score_today(config, data)
    conviction_table = build_conviction_table(config, model_tables)
    actions = compute_position_actions(config, data)

    merged = conviction_table.merge(actions, on=["pair1", "pair2"])
    merged = merged.sort_values("conviction_score", ascending=False).reset_index(drop=True)

    out_dir = Path("reports") / STRATEGY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    merged.to_csv(out_dir / f"position_management_{today}.csv", index=False)

    top_n = config["position_management"]["top_n_report"]
    report_text = render_report(merged, config, top_n)
    with open(out_dir / f"position_management_{today}.md", "w") as f:
        f.write(f"# fx_correlation position management — {today}\n\n```\n{report_text}\n```\n")

    return merged


if __name__ == "__main__":
    cfg = _load_config()
    result_table = run()
    print(render_report(result_table, cfg, cfg["position_management"]["top_n_report"]))
