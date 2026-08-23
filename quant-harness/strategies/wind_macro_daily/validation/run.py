"""Persist per-factor walk-forward out-of-sample results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from core.validation.walk_forward import walk_forward_splits
from strategies.wind_macro_daily.backtest.run import (
    STRATEGY_NAME,
    _active_config,
    _load_for_run,
    _run_over_dates,
    _write_csv,
)
from strategies.wind_macro_daily.data.loader import CONFIG_PATH, load_config


def _parameter_rows(config: dict[str, Any], variants: list[str]) -> pd.DataFrame:
    factors = config["factors"]
    risk = config["risk"]
    walk_forward = config["walk_forward"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.append(
            {
                "signal_variant": variant,
                "fast_window": factors["fast_window"],
                "medium_window": factors["medium_window"],
                "zscore_window": factors["zscore_window"],
                "signal_clip": factors["signal_clip"],
                "volatility_window": risk["volatility_window"],
                "target_volatility": risk["target_volatility"],
                "max_instrument_leverage": risk["max_instrument_leverage"],
                "max_portfolio_gross": risk["max_portfolio_gross"],
                "train_size": walk_forward["train_size"],
                "test_size": walk_forward["test_size"],
                "step": walk_forward["step"],
                "parameter_selection": "fixed_before_oos",
            }
        )
    return pd.DataFrame(rows)


def _summary(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in windows.groupby("signal_variant", sort=False):
        active = cast(pd.Series, group["active_position_days"]).sum()
        rows.append(
            {
                "signal_variant": variant,
                "windows": len(group),
                "average_oos_sharpe": group["sharpe"].mean(),
                "worst_window_sharpe": group["sharpe"].min(),
                "positive_sharpe_fraction": group["sharpe"].gt(0).mean(),
                "average_oos_annualized_return": group["annualized_return"].mean(),
                "average_oos_max_drawdown": group["max_drawdown"].mean(),
                "worst_oos_max_drawdown": group["max_drawdown"].min(),
                "average_oos_turnover": group["turnover"].mean(),
                "active_position_days": int(active),
                "status": "evaluated" if active > 0 else "unavailable_input_or_no_signal",
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Per-factor walk-forward validation")
    parser.add_argument("--data-mode", choices=["synthetic", "wind", "manual_excel"], default=None)
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    config = load_config()
    data_mode = args.data_mode or str(config["data"]["mode"])
    data, manual_bundle = _load_for_run(config, data_mode, args.workbook)
    config = _active_config(config, data)
    run_id = args.run_id or (
        f"manual_{manual_bundle.workbook_sha256[:10]}" if manual_bundle else data_mode
    )
    variants = [str(value) for value in config["walk_forward"]["signal_variants"]]
    train_size = int(config["walk_forward"]["train_size"])
    test_size = int(config["walk_forward"]["test_size"])
    step = int(config["walk_forward"]["step"])
    index = cast(pd.DatetimeIndex, data.index)
    splits = list(walk_forward_splits(index, train_size, test_size, step))
    if not splits:
        raise RuntimeError("walk-forward produced no windows; check train/test/step versus data length")

    window_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    for variant in variants:
        for window_number, window in enumerate(splits, start=1):
            train_df = data.loc[window.train]
            test_df = data.loc[window.test]
            history = pd.concat([train_df, test_df]).sort_index()
            pit = PointInTimeFrame(history)
            positions, pnl, _ = _run_over_dates(
                pit,
                history,
                cast(pd.DatetimeIndex, test_df.index),
                config,
                signal_variant=variant,
            )
            portfolio = pnl.sum(axis=1, min_count=1).fillna(0.0)
            metrics = summarize(portfolio, positions)
            active_days = int(positions.abs().sum(axis=1).gt(0).sum())
            window_rows.append(
                {
                    "signal_variant": variant,
                    "window": window_number,
                    "train_start": window.train[0],
                    "train_end": window.train[-1],
                    "test_start": window.test[0],
                    "test_end": window.test[-1],
                    "train_observations": len(window.train),
                    "test_observations": len(window.test),
                    "active_position_days": active_days,
                    **metrics,
                }
            )
            daily_rows.append(
                pd.DataFrame(
                    {
                        "date": portfolio.index,
                        "signal_variant": variant,
                        "window": window_number,
                        "oos_return": portfolio.to_numpy(),
                        "gross_exposure": positions.abs().sum(axis=1).to_numpy(),
                    }
                )
            )

    windows = pd.DataFrame(window_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    summary = _summary(windows)
    parameters = _parameter_rows(config, variants)
    report_root = Path("reports") / STRATEGY_NAME
    report_dir = report_root / "walk_forward" / run_id
    parameter_yaml = report_root / "parameters" / f"{run_id}_walk_forward.yaml"
    parameter_csv = report_root / "parameters" / f"{run_id}_factor_parameters.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    parameter_yaml.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(windows, report_dir / "windows.csv", index=False)
    _write_csv(summary, report_dir / "summary.csv", index=False)
    _write_csv(daily, report_dir / "oos_daily_returns.csv.gz", index=False)
    _write_csv(parameters, parameter_csv, index=False)
    parameter_yaml.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {
        "strategy": STRATEGY_NAME,
        "run_id": run_id,
        "data_mode": data_mode,
        "active_universe": config["universe"],
        "signal_variants": variants,
        "walk_forward_windows": len(splits),
        "parameter_selection": "fixed before each out-of-sample test; no test-window tuning",
        "parameters_yaml": parameter_yaml.as_posix(),
        "parameters_csv": parameter_csv.as_posix(),
        "lookahead_status": (
            "release-timing guarded; historical revised-vintage leakage unresolved"
            if manual_bundle
            else "synthetic engineering data"
        ),
    }
    (report_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {report_dir / 'summary.csv'} for {len(variants)} signal variants")


if __name__ == "__main__":
    main()
