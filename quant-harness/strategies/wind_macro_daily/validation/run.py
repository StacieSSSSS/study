"""Persist instrument-by-instrument, per-factor walk-forward OOS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from core.validation.walk_forward import walk_forward_splits
from strategies.wind_macro_daily.backtest.run import (
    INDICATOR_LABELS,
    STRATEGY_NAME,
    _active_config,
    _load_for_run,
    _write_csv,
    run_independent_grid,
)
from strategies.wind_macro_daily.data.loader import CONFIG_PATH, load_config
from strategies.wind_macro_daily.validation.reporting import render_validation_charts

IMPROVEMENT_LOG = Path(__file__).resolve().parents[1] / "IMPROVEMENTS.yaml"


def _parameter_rows(
    config: dict[str, Any], variants: list[str], configured_universe: list[str]
) -> pd.DataFrame:
    factors = config["factors"]
    risk = config["risk"]
    walk_forward = config["walk_forward"]
    active = set(config["universe"])
    rows: list[dict[str, Any]] = []
    for instrument in configured_universe:
        spec = config["instruments"][instrument]
        for variant in variants:
            rows.append(
                {
                    "instrument": instrument,
                    "asset_class": spec["asset_class"],
                    "data_status": "available" if instrument in active else "missing",
                    "signal_variant": variant,
                    "technical_indicator": INDICATOR_LABELS[variant],
                    "fast_window": factors["fast_window"],
                    "medium_window": factors["medium_window"],
                    "zscore_window": factors["zscore_window"],
                    "signal_clip": factors["signal_clip"],
                    "volatility_window": risk["volatility_window"],
                    "target_volatility": risk["target_volatility"],
                    "max_instrument_leverage": risk["max_instrument_leverage"],
                    "cost_bps": spec["cost_bps"],
                    "return_model": spec.get("return_model", "log_price"),
                    "duration": spec.get("duration"),
                    "train_size": walk_forward["train_size"],
                    "test_size": walk_forward["test_size"],
                    "step": walk_forward["step"],
                    "parameter_selection": "fixed_before_oos",
                }
            )
    return pd.DataFrame(rows)


def _effectiveness_reason(row: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, bool]:
    if int(row["active_position_days"]) < int(thresholds["minimum_active_position_days"]):
        return "unavailable_input_or_no_signal", False
    if int(row["windows"]) < int(thresholds["minimum_windows"]):
        return "insufficient_windows", False

    failures: list[str] = []
    checks = (
        ("average_oos_sharpe", ">=", "minimum_average_oos_sharpe"),
        ("worst_window_sharpe", ">=", "minimum_worst_window_sharpe"),
        ("positive_sharpe_fraction", ">=", "minimum_positive_sharpe_fraction"),
        ("average_oos_annualized_return", ">=", "minimum_average_oos_annualized_return"),
        ("worst_oos_max_drawdown", ">=", "maximum_worst_oos_drawdown"),
    )
    for metric, operator, threshold_name in checks:
        value = float(row[metric])
        threshold = float(thresholds[threshold_name])
        if value < threshold:
            failures.append(f"{metric} {value:.4f} {operator} {threshold:.4f} failed")
    return ("all_oos_thresholds_passed", True) if not failures else ("; ".join(failures), False)


def _summary(windows: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = windows.groupby(["instrument", "asset_class", "signal_variant"], sort=False)
    for keys, group in grouped:
        instrument, asset_class, variant = cast(tuple[str, str, str], keys)
        row: dict[str, Any] = {
            "instrument": instrument,
            "asset_class": asset_class,
            "signal_variant": variant,
            "technical_indicator": INDICATOR_LABELS[str(variant)],
            "windows": len(group),
            "average_oos_sharpe": group["sharpe"].mean(),
            "worst_window_sharpe": group["sharpe"].min(),
            "positive_sharpe_fraction": group["sharpe"].gt(0).mean(),
            "average_oos_annualized_return": group["annualized_return"].mean(),
            "average_oos_max_drawdown": group["max_drawdown"].mean(),
            "worst_oos_max_drawdown": group["max_drawdown"].min(),
            "average_oos_turnover": group["turnover"].mean(),
            "active_position_days": int(
                np.sum(cast(pd.Series, group["active_position_days"]).to_numpy(dtype=int))
            ),
        }
        reason, effective = _effectiveness_reason(row, thresholds)
        row["effective"] = effective
        unavailable = reason in {"unavailable_input_or_no_signal", "insufficient_windows"}
        row["effectiveness_status"] = (
            "effective" if effective else reason if unavailable else "not_effective"
        )
        row["effectiveness_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(configured: dict[str, Any], active_universe: list[str]) -> pd.DataFrame:
    active = set(active_universe)
    rows: list[dict[str, Any]] = []
    for instrument in configured["universe"]:
        spec = configured["instruments"][instrument]
        available = instrument in active
        if available:
            reason = "workbook column mapped and usable"
        elif instrument == "CN_IRS_5Y":
            reason = "需要5Y FR007 IRS每日收盘/中间价；当前工作簿没有该列"
        elif instrument == "US_IRS_5Y":
            reason = "需要5Y SOFR OIS/IRS par rate；美国国债收益率不能替代IRS"
        else:
            reason = "configured price column missing or contains no usable observations"
        rows.append(
            {
                "instrument": instrument,
                "asset_class": spec["asset_class"],
                "data_status": "available" if available else "missing",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _load_improvements() -> dict[str, Any]:
    with open(IMPROVEMENT_LOG, encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Independent instrument/factor walk-forward validation")
    parser.add_argument("--data-mode", choices=["synthetic", "wind", "manual_excel"], default=None)
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    configured = load_config()
    configured_universe = list(configured["universe"])
    data_mode = args.data_mode or str(configured["data"]["mode"])
    data, manual_bundle = _load_for_run(configured, data_mode, args.workbook)
    config = _active_config(configured, data)
    run_id = args.run_id or (
        f"manual_{manual_bundle.workbook_sha256[:10]}" if manual_bundle else data_mode
    )
    variants = [str(value) for value in config["walk_forward"]["signal_variants"]]
    train_size = int(config["walk_forward"]["train_size"])
    test_size = int(config["walk_forward"]["test_size"])
    step = int(config["walk_forward"]["step"])
    thresholds = cast(dict[str, Any], config["walk_forward"]["effectiveness"])
    index = cast(pd.DatetimeIndex, data.index)
    splits = list(walk_forward_splits(index, train_size, test_size, step))
    if not splits:
        raise RuntimeError("walk-forward produced no windows; check train/test/step versus data length")

    pit = PointInTimeFrame(data)
    _, full_daily = run_independent_grid(pit, data, index, index, config, variants)
    full_daily["date"] = pd.to_datetime(full_daily["date"])

    window_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    for instrument in config["universe"]:
        asset_class = config["instruments"][instrument]["asset_class"]
        for variant in variants:
            pair = cast(
                pd.DataFrame,
                full_daily[
                    full_daily["instrument"].eq(instrument)
                    & full_daily["signal_variant"].eq(variant)
                ].set_index("date"),
            )
            for window_number, window in enumerate(splits, start=1):
                oos = cast(pd.DataFrame, pair.loc[window.test])
                oos_return = cast(pd.Series, oos["daily_return"]).fillna(0.0)
                oos_position = cast(pd.Series, oos["position"]).rename(instrument).to_frame()
                active_days = int(
                    np.count_nonzero(
                        np.abs(cast(pd.Series, oos_position[instrument]).to_numpy(dtype=float)) > 0.0
                    )
                )
                window_rows.append(
                    {
                        "instrument": instrument,
                        "asset_class": asset_class,
                        "signal_variant": variant,
                        "technical_indicator": INDICATOR_LABELS[variant],
                        "window": window_number,
                        "train_start": window.train[0],
                        "train_end": window.train[-1],
                        "test_start": window.test[0],
                        "test_end": window.test[-1],
                        "train_observations": len(window.train),
                        "test_observations": len(window.test),
                        "active_position_days": active_days,
                        **summarize(oos_return, oos_position),
                    }
                )
                daily_rows.append(
                    pd.DataFrame(
                        {
                            "date": window.test,
                            "instrument": instrument,
                            "asset_class": asset_class,
                            "signal_variant": variant,
                            "window": window_number,
                            "oos_return": oos_return.to_numpy(),
                            "position": oos_position[instrument].to_numpy(),
                        }
                    )
                )

    windows = pd.DataFrame(window_rows)
    daily = pd.concat(daily_rows, ignore_index=True)
    summary = _summary(windows, thresholds)
    coverage = _coverage(configured, config["universe"])
    parameters = _parameter_rows(config, variants, configured_universe)
    improvements = _load_improvements()

    report_root = Path("reports") / STRATEGY_NAME
    report_dir = report_root / "walk_forward" / run_id
    visualization_dir = report_root / "visualizations" / run_id
    parameter_yaml = report_root / "parameters" / f"{run_id}_walk_forward.yaml"
    parameter_csv = report_root / "parameters" / f"{run_id}_instrument_factor_parameters.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    parameter_yaml.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(windows, report_dir / "windows_by_instrument_factor.csv", index=False)
    _write_csv(summary, report_dir / "factor_effectiveness.csv", index=False)
    _write_csv(daily, report_dir / "oos_daily_returns_by_instrument_factor.csv.gz", index=False)
    _write_csv(coverage, report_dir / "instrument_data_coverage.csv", index=False)
    _write_csv(parameters, parameter_csv, index=False)
    parameter_yaml.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    chart_paths = render_validation_charts(daily, summary, coverage, visualization_dir)

    harness_status = {
        "strategy": STRATEGY_NAME,
        "run_id": run_id,
        "validation_unit": "instrument_x_factor",
        "configured_instruments": configured_universe,
        "active_instruments": config["universe"],
        "effective_pairs": summary.loc[
            summary["effective"], ["instrument", "signal_variant"]
        ].to_dict(orient="records"),
        "unavailable_instruments": coverage.loc[
            coverage["data_status"].eq("missing"), "instrument"
        ].tolist(),
        "open_improvements": [
            item["id"] for item in improvements["improvements"] if item["status"] == "open"
        ],
        "artifacts": {
            "windows": (report_dir / "windows_by_instrument_factor.csv").as_posix(),
            "factor_effectiveness": (report_dir / "factor_effectiveness.csv").as_posix(),
            "oos_daily_returns": (
                report_dir / "oos_daily_returns_by_instrument_factor.csv.gz"
            ).as_posix(),
            "coverage": (report_dir / "instrument_data_coverage.csv").as_posix(),
            "parameters": parameter_csv.as_posix(),
            "visualizations": [path.as_posix() for path in chart_paths],
        },
        "lookahead_status": (
            "release-timing guarded; historical revised-vintage leakage unresolved"
            if manual_bundle
            else "synthetic engineering data"
        ),
    }
    (report_dir / "harness_status.json").write_text(
        json.dumps(harness_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "improvements_snapshot.yaml").write_text(
        yaml.safe_dump(improvements, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(
        f"wrote {report_dir / 'factor_effectiveness.csv'} for "
        f"{len(config['universe'])} instruments x {len(variants)} factors"
    )


if __name__ == "__main__":
    main()
