"""Chart generation for backtest results and the daily conviction report.

Reads the CSVs that `backtest/run.py`, `reporting/model_backtests.py`, and
`reporting/conviction.py` already persist — never re-runs a backtest itself,
since those take minutes. Run those first if their reports/ outputs are
missing or stale; this module only renders what's already on disk.

Run with: `python3 -m strategies.fx_correlation.reporting.plots`
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # headless — this environment has no display
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from core.data.point_in_time import PointInTimeFrame
from strategies.fx_correlation.data.loader import load_raw
from strategies.fx_correlation.lib.correlation import multi_window_correlation

STRATEGY_NAME = "fx_correlation"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
REPORTS_DIR = Path("reports") / STRATEGY_NAME

CONVICTION_COLORS = {"High": "#2a9d8f", "Medium": "#e9c46a", "Low": "#e76f51"}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _require(path: Path, generator_hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `{generator_hint}` first.")
    return path


def plot_equity_curves() -> Path:
    """Cumulative growth of the blended strategy vs. each model run standalone."""
    blended_path = _require(
        REPORTS_DIR / "blended_returns.csv", "python3 -m strategies.fx_correlation.backtest.run"
    )
    model_path = _require(
        REPORTS_DIR / "model_returns.csv", "python3 -m strategies.fx_correlation.reporting.model_backtests"
    )

    blended = pd.read_csv(blended_path, index_col=0, parse_dates=True)["return"]
    models = pd.read_csv(model_path, index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    (1.0 + blended).cumprod().plot(ax=ax, label="blended (all 3 models)", linewidth=2.5, color="black")
    for column in models.columns:
        (1.0 + models[column]).cumprod().plot(ax=ax, label=column, linewidth=1.2, alpha=0.8)

    ax.set_title("fx_correlation — cumulative growth of $1, out-of-sample")
    ax.set_ylabel("growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()

    out_path = REPORTS_DIR / "equity_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_drawdown() -> Path:
    """Underwater (drawdown) chart for the blended strategy."""
    blended_path = _require(
        REPORTS_DIR / "blended_returns.csv", "python3 -m strategies.fx_correlation.backtest.run"
    )
    blended = pd.read_csv(blended_path, index_col=0, parse_dates=True)["return"]
    equity = (1.0 + blended).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    drawdown.plot(ax=ax, color="#e76f51", linewidth=1.2)
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#e76f51", alpha=0.3)
    ax.set_title("fx_correlation — blended strategy drawdown")
    ax.set_ylabel("drawdown")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    fig.tight_layout()

    out_path = REPORTS_DIR / "drawdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_conviction_bar() -> Path:
    """Bar chart of today's (or the latest saved) conviction ranking."""
    candidates = sorted(glob.glob(str(REPORTS_DIR / "conviction_*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"No conviction_*.csv under {REPORTS_DIR} — "
            "run `python3 -m strategies.fx_correlation.reporting.conviction` first."
        )
    latest_csv = Path(candidates[-1])
    table = pd.read_csv(latest_csv)
    top = table.sort_values("conviction_score", ascending=False).head(15)
    labels = [f"{row['pair1']}-{row['pair2']}" for _, row in top.iterrows()]
    colors = [CONVICTION_COLORS.get(c, "#999999") for c in top["conviction"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], top["conviction_score"][::-1], color=colors[::-1])
    ax.set_xlabel("conviction score (0-100, avg rank across 3 models)")
    ax.set_title(f"fx_correlation — conviction ranking ({latest_csv.stem.replace('conviction_', '')})")
    ax.set_xlim(0, 100)
    fig.tight_layout()

    out_path = REPORTS_DIR / "conviction_chart.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap() -> Path:
    """Current pairwise correlation matrix (12-month window) across the universe."""
    config = _load_config()
    universe = config["pairs"]
    data = load_raw()
    pit = PointInTimeFrame(data)
    as_of = cast(pd.Timestamp, data.index[-1])
    view = pit.as_of(as_of)

    matrix = pd.DataFrame(index=universe, columns=universe, dtype=float)
    for p1 in universe:
        for p2 in universe:
            if p1 == p2:
                matrix.loc[p1, p2] = 1.0
                continue
            x, y = cast(pd.Series, view[p1]), cast(pd.Series, view[p2])
            corrs = multi_window_correlation(x, y, {"12m": config["windows"]["12m"]})
            matrix.loc[p1, p2] = corrs["12m"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(universe)), universe, rotation=45, ha="right")
    ax.set_yticks(range(len(universe)), universe)
    for i in range(len(universe)):
        for j in range(len(universe)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="12m correlation")
    ax.set_title(f"fx_correlation — 12m correlation matrix ({as_of.date()})")
    fig.tight_layout()

    out_path = REPORTS_DIR / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_all() -> list[Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for plot_fn in (plot_equity_curves, plot_drawdown, plot_conviction_bar, plot_correlation_heatmap):
        try:
            paths.append(plot_fn())
        except FileNotFoundError as e:
            print(f"skipped {plot_fn.__name__}: {e}")
    return paths


if __name__ == "__main__":
    for path in render_all():
        print(f"wrote {path}")
