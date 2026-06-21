"""Daily refresh entrypoint: re-fetches FX data, scores every pair combination
under all three models *as of today*, and reports a conviction ranking.

Run with: `python3 -m strategies.fx_correlation.reporting.conviction`

Conviction = average rank across the three models (rank 1 = most attractive
under that model's lens). A combo a model's filter rejects (e.g. divergence
too small for Model B, ADF p-value too high for Model C) is scored as that
model's *worst* rank rather than excluded — being unconvincing under one
lens should pull the average down, not be ignored.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from core.data.point_in_time import PointInTimeFrame
from strategies.fx_correlation.data.loader import load_raw
from strategies.fx_correlation.models import model_a_strength, model_b_divergence, model_c_cointegration
from strategies.fx_correlation.models.base import all_combos, fit_hedge_ratios, rank_table, score_date

STRATEGY_NAME = "fx_correlation"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

MODEL_SPECS = {
    "model_a_strength": (model_a_strength.selection_score, model_a_strength.NEED_ADF, "model_a"),
    "model_b_divergence": (model_b_divergence.selection_score, model_b_divergence.NEED_ADF, "model_b"),
    "model_c_cointegration": (
        model_c_cointegration.selection_score, model_c_cointegration.NEED_ADF, "model_c"
    ),
}

LIVE_FIT_WINDOW = "12m"  # which configured window to use as the hedge-ratio fitting lookback


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _model_cfg(config: dict, model_key: str) -> dict:
    cfg = dict(config[model_key])
    if model_key == "model_b":
        cfg["short_window"] = config["short_window"]
        cfg["baseline_window"] = config["baseline_window"]
    return cfg


def score_today(config: dict, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    universe = config["pairs"]
    combos = all_combos(universe)
    as_of = cast(pd.Timestamp, data.index[-1])

    fit_window_days = config["windows"][LIVE_FIT_WINDOW]
    train_df = data.loc[:as_of].tail(fit_window_days)
    betas = fit_hedge_ratios(train_df, combos)
    pit = PointInTimeFrame(data)

    tables: dict[str, pd.DataFrame] = {}
    for model_name, (score_fn, need_adf, model_key) in MODEL_SPECS.items():
        model_cfg = _model_cfg(config, model_key)
        metrics_by_combo, scores = score_date(
            pit, as_of, universe, combos, betas, config["windows"], config["zscore_lookback"],
            score_fn, model_cfg, need_adf,
        )
        tables[model_name] = rank_table(combos, metrics_by_combo, scores)
    return tables


def _direction_label(avg_zscore: float) -> str:
    if pd.isna(avg_zscore):
        return "n/a"
    return "short pair2 / long pair1" if avg_zscore > 0 else "long pair2 / short pair1"


MODEL_LABELS = {
    "model_a_strength": "Model A(强度/稳定性)",
    "model_b_divergence": "Model B(背离)",
    "model_c_cointegration": "Model C(cointegration)",
}


def _model_detail(model_name: str, row: pd.Series, config: dict, n_combos: int) -> str:
    """The concrete number that drove `model_name`'s ranking of this combo."""
    rank = int(cast(float, row[f"rank_{model_name}"]))
    if model_name == "model_a_strength":
        c_short = cast(float, row[f"corr_{config['short_window']}_{model_name}"])
        c_base = cast(float, row[f"corr_{config['baseline_window']}_{model_name}"])
        detail = (
            f"{config['short_window']}相关性{c_short:.2f}、{config['baseline_window']}相关性{c_base:.2f}，"
            "多窗口下相关性强且一致"
        )
    elif model_name == "model_b_divergence":
        c_short = cast(float, row[f"corr_{config['short_window']}_{model_name}"])
        c_base = cast(float, row[f"corr_{config['baseline_window']}_{model_name}"])
        divergence = abs(c_short - c_base) if pd.notna(c_short) and pd.notna(c_base) else float("nan")
        detail = (
            f"{config['short_window']}相关性{c_short:.2f} 偏离 {config['baseline_window']}基准{c_base:.2f}，"
            f"背离幅度{divergence:.2f}"
        )
    else:
        adf_p = cast(float, row[f"adf_p_{model_name}"])
        adf_stat = cast(float, row[f"adf_stat_{model_name}"])
        p_text = "<0.001" if adf_p < 0.001 else f"{adf_p:.3f}"
        detail = (
            f"价差ADF检验统计量={adf_stat:.1f}（越负越显著），p值{p_text}，"
            "存在统计意义上的均值回归证据"
        )
    return f"{MODEL_LABELS[model_name]}排名第{rank}/{n_combos}：{detail}"


def _build_reason(row: pd.Series, config: dict, n_combos: int) -> str:
    ranks = {name: row[f"rank_{name}"] for name in MODEL_LABELS}
    best_model = min(ranks, key=lambda name: ranks[name])
    main_reason = _model_detail(best_model, row, config, n_combos)

    top_quartile = max(1, round(n_combos * 0.25))
    agreeing = [
        MODEL_LABELS[name] for name, rank in ranks.items() if name != best_model and rank <= top_quartile
    ]
    if agreeing:
        return f"{main_reason}；{'、'.join(agreeing)}也将其排进前{top_quartile}"
    return main_reason


def build_conviction_table(config: dict, model_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    n_combos = len(next(iter(model_tables.values())))
    worst_rank = n_combos
    corr_cols = [f"corr_{name}" for name in config["windows"]]

    slim_tables: list[pd.DataFrame] = []
    for model_name, table in model_tables.items():
        rank_filled = cast(pd.Series, table["rank"]).fillna(worst_rank)
        fields = {
            "pair1": table["pair1"],
            "pair2": table["pair2"],
            f"rank_{model_name}": rank_filled,
            f"zscore_{model_name}": table["zscore"],
            f"adf_p_{model_name}": table["adf_p"],
            f"adf_stat_{model_name}": table["adf_stat"],
        }
        fields.update({f"{col}_{model_name}": table[col] for col in corr_cols})
        slim_tables.append(pd.DataFrame(fields))

    merged = slim_tables[0]
    for slim in slim_tables[1:]:
        merged = merged.merge(slim, on=["pair1", "pair2"])

    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    zscore_cols = [c for c in merged.columns if c.startswith("zscore_")]
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged["conviction_score"] = 100 * (n_combos - merged["avg_rank"]) / max(n_combos - 1, 1)
    merged["avg_zscore"] = merged[zscore_cols].mean(axis=1, skipna=True)

    high = config["conviction"]["high_threshold"]
    medium = config["conviction"]["medium_threshold"]

    def label(score: float) -> str:
        if score >= high:
            return "High"
        if score >= medium:
            return "Medium"
        return "Low"

    merged["conviction"] = merged["conviction_score"].apply(label)
    merged["suggested_direction"] = merged["avg_zscore"].apply(_direction_label)
    merged["reason"] = merged.apply(lambda row: _build_reason(row, config, n_combos), axis=1)

    merged = merged.sort_values("conviction_score", ascending=False).reset_index(drop=True)
    merged.insert(0, "overall_rank", range(1, len(merged) + 1))
    return merged


def render_report(conviction_table: pd.DataFrame, top_n: int) -> str:
    display_cols = [
        "overall_rank", "pair1", "pair2", "conviction_score", "conviction",
        "suggested_direction", "reason",
    ]
    top = conviction_table[display_cols].head(top_n).copy()
    top["conviction_score"] = top["conviction_score"].round(1)
    lines = []
    for _, row in top.iterrows():
        lines.append(
            f"#{row['overall_rank']} {row['pair1']}-{row['pair2']}  "
            f"[{row['conviction']} {row['conviction_score']}]  {row['suggested_direction']}\n"
            f"    -> {row['reason']}"
        )
    return "\n".join(lines)


def run(refresh: bool = True) -> pd.DataFrame:
    config = _load_config()
    data = load_raw(refresh=refresh)
    model_tables = score_today(config, data)
    conviction_table = build_conviction_table(config, model_tables)

    out_dir = Path("reports") / STRATEGY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    conviction_table.to_csv(out_dir / f"conviction_{today}.csv", index=False)

    report_text = render_report(conviction_table, config["conviction"]["top_n_report"])
    with open(out_dir / f"conviction_{today}.md", "w") as f:
        f.write(f"# fx_correlation conviction report — {today}\n\n```\n{report_text}\n```\n")

    summary = {
        "date": today,
        "as_of": str(cast(pd.Timestamp, data.index[-1]).date()),
        "top_combo": f"{conviction_table.iloc[0]['pair1']}-{conviction_table.iloc[0]['pair2']}",
        "top_conviction_score": round(float(conviction_table.iloc[0]["conviction_score"]), 1),
    }
    with open(out_dir / "conviction_latest.json", "w") as f:
        json.dump(summary, f, indent=2)

    return conviction_table


if __name__ == "__main__":
    table = run()
    print(render_report(table, 10))
