"""Static performance charts for independent instrument/factor validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant-harness-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "momentum": "#0072B2",
    "carry_signal": "#E69F00",
    "mean_reversion": "#009E73",
    "macro_signal": "#CC79A7",
    "composite_signal": "#D55E00",
}


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _draw_instrument_chart(instrument: str, daily: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    upper, lower = axes
    for variant, group in daily.groupby("signal_variant", sort=False):
        ordered = cast(pd.DataFrame, group).sort_values("date")
        if not cast(pd.Series, ordered["position"]).abs().gt(0).any():
            continue
        returns = cast(pd.Series, ordered["oos_return"]).fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        color = COLORS.get(str(variant), "#666666")
        upper.plot(ordered["date"], equity, label=str(variant), color=color, linewidth=1.6)
        lower.plot(ordered["date"], drawdown, color=color, linewidth=1.2)

    upper.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    upper.set_title(f"{instrument} — 独立因子 Walk-forward 样本外净值", fontweight="bold")
    upper.set_ylabel("净值（起点=1）")
    upper.legend(ncol=3, frameon=False, fontsize=8)
    lower.axhline(0.0, color="#777777", linewidth=0.8)
    lower.set_title("样本外回撤", loc="left", fontsize=10)
    lower.set_ylabel("回撤")
    lower.set_xlabel("日期")
    lower.yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _draw_unavailable_chart(instrument: str, reason: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.65, instrument, ha="center", va="center", fontsize=20, fontweight="bold")
    axis.text(
        0.5,
        0.48,
        "DATA UNAVAILABLE / 无法回测",
        ha="center",
        va="center",
        fontsize=14,
        color="#B22222",
    )
    axis.text(0.5, 0.30, reason, ha="center", va="center", fontsize=10, wrap=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _draw_heatmap(summary: pd.DataFrame, output: Path) -> None:
    matrix = summary.pivot(index="instrument", columns="signal_variant", values="average_oos_sharpe")
    statuses = summary.pivot(index="instrument", columns="signal_variant", values="effectiveness_status")
    matrix = matrix.reindex(columns=list(COLORS))
    statuses = statuses.reindex(index=matrix.index, columns=matrix.columns)
    matrix = matrix.mask(statuses.isin(["unavailable_input_or_no_signal", "insufficient_windows"]))
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0
    figure, axis = plt.subplots(figsize=(11, max(4.5, 0.6 * len(matrix.index) + 2.0)))
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "N/A"
            else:
                status = str(statuses.iloc[row, column])
                marker = " 有效" if status == "effective" else ""
                label = f"{value:.2f}{marker}"
            axis.text(column, row, label, ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("逐标的 × 因子：平均样本外 Sharpe（按阈值标注有效）", fontweight="bold")
    axis.set_xlabel("因子 / 技术指标")
    axis.set_ylabel("交易标的")
    figure.colorbar(image, ax=axis, label="平均 OOS Sharpe", shrink=0.8)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_validation_charts(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Render one OOS performance chart per configured instrument and a heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for row in coverage.to_dict(orient="records"):
        instrument = str(row["instrument"])
        output = output_dir / f"{_safe_name(instrument)}_strategy_performance.png"
        instrument_daily = cast(pd.DataFrame, daily[daily["instrument"].eq(instrument)])
        if instrument_daily.empty:
            _draw_unavailable_chart(instrument, str(row["reason"]), output)
        else:
            _draw_instrument_chart(instrument, instrument_daily, output)
        outputs.append(output)
    heatmap = output_dir / "factor_effectiveness_heatmap.png"
    _draw_heatmap(summary, heatmap)
    outputs.append(heatmap)
    return outputs
