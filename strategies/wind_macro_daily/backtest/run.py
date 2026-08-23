"""Harness-standard backtest entrypoint for wind_macro_daily."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from strategies.wind_macro_daily.data.excel_loader import ManualExcelBundle, load_manual_excel_bundle
from strategies.wind_macro_daily.data.loader import CONFIG_PATH, load_config, load_raw
from strategies.wind_macro_daily.factors.engine import factor_snapshot
from strategies.wind_macro_daily.signals.signal import generate_signal

STRATEGY_NAME = "wind_macro_daily"


def load_data() -> pd.DataFrame:
    """Compatibility entrypoint used by the shared harness."""
    return load_raw()


def _active_config(config: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    active = [
        instrument
        for instrument in config["universe"]
        if f"{instrument}__close" in data and f"{instrument}__return" in data
    ]
    if not active:
        raise ValueError("no configured instruments have both close and return columns")
    run_config["universe"] = active
    return run_config


def _returns(data: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {instrument: data[f"{instrument}__return"].astype(float) for instrument in universe},
        index=data.index,
    )


def _run_over_dates(
    pit: PointInTimeFrame,
    history: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
    signal_variant: str = "composite_signal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = config["universe"]
    returns = _returns(history, universe)
    positions = pd.DataFrame(0.0, index=dates, columns=universe)
    factor_frames: list[pd.DataFrame] = []
    risk = config["risk"]

    for as_of in dates:
        view = pit.as_of(as_of)
        signal = generate_signal(pit, as_of, config, signal_variant=signal_variant)
        snapshot = factor_snapshot(pit, as_of, config)
        snapshot["signal_variant"] = signal_variant
        factor_frames.append(snapshot)
        volatility = (
            _returns(view, universe)
            .tail(int(risk["volatility_window"]))
            .std(ddof=1)
            * np.sqrt(int(risk["annualization"]))
        )
        scalar = (float(risk["target_volatility"]) / volatility).clip(
            upper=float(risk["max_instrument_leverage"])
        )
        target = (signal * scalar).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = float(target.abs().sum())
        if gross > float(risk["max_portfolio_gross"]):
            target *= float(risk["max_portfolio_gross"]) / gross
        positions.loc[as_of] = target

    lagged_positions = positions.shift(1).fillna(0.0)
    gross_pnl = lagged_positions * returns.loc[dates]
    turnover = positions.diff().abs().fillna(positions.abs())
    cost_rates = pd.Series(
        {
            instrument: float(config["instruments"][instrument]["cost_bps"]) / 10_000.0
            for instrument in universe
        }
    )
    costs = turnover.mul(cost_rates, axis=1)
    pnl = gross_pnl - costs
    factor_values = pd.concat(factor_frames, ignore_index=True)
    stacked_positions = cast(pd.Series, positions.stack()).rename("target_position")
    return positions, pnl, factor_values.merge(
        stacked_positions.rename_axis(["date", "instrument"]).reset_index(),
        on=["date", "instrument"],
        how="left",
    )


def evaluate_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Compatibility evaluator for the shared composite-signal walk-forward command."""
    config = _active_config(load_config(), pd.concat([train_df, test_df]))
    history = pd.concat([train_df, test_df]).sort_index()
    pit = PointInTimeFrame(history)
    dates = cast(pd.DatetimeIndex, test_df.index)
    positions, instrument_pnl, _ = _run_over_dates(pit, history, dates, config)
    portfolio = instrument_pnl.sum(axis=1, min_count=1).fillna(0.0)
    return summarize(portfolio, positions)


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression: str | dict[str, object] | None = None
    if path.suffix == ".gz":
        compression = {"method": "gzip", "compresslevel": 9, "mtime": 0}
    frame.to_csv(path, index=index, float_format="%.10g", compression=compression)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _artifact_paths(data_mode: str, run_id: str) -> tuple[Path, Path, Path, Path]:
    if data_mode == "manual_excel":
        data_path = (
            Path("strategies") / STRATEGY_NAME / "data" / "manual" / run_id / "market_data.csv.gz"
        )
        factor_path = (
            Path("strategies")
            / STRATEGY_NAME
            / "factors"
            / "output"
            / "manual"
            / run_id
            / "factor_values.csv.gz"
        )
        position_path = (
            Path("strategies")
            / STRATEGY_NAME
            / "strategy_output"
            / "manual"
            / run_id
            / "positions.csv.gz"
        )
        backtest_path = (
            Path("strategies")
            / STRATEGY_NAME
            / "backtest"
            / "output"
            / "manual"
            / run_id
            / "daily_pnl.csv.gz"
        )
        return data_path, factor_path, position_path, backtest_path

    tag = "sample" if data_mode == "synthetic" else "wind"
    return (
        Path("strategies") / STRATEGY_NAME / "data" / tag / "market_data.csv.gz",
        Path("strategies") / STRATEGY_NAME / "factors" / "output" / tag / "factor_values.csv.gz",
        Path("strategies") / STRATEGY_NAME / "strategy_output" / tag / "positions.csv.gz",
        Path("strategies") / STRATEGY_NAME / "backtest" / "output" / tag / "daily_pnl.csv.gz",
    )


def _load_for_run(
    config: dict[str, Any], data_mode: str, workbook: str | None
) -> tuple[pd.DataFrame, ManualExcelBundle | None]:
    if data_mode == "manual_excel":
        bundle = load_manual_excel_bundle(config, workbook)
        return bundle.market_data, bundle
    return load_raw(mode_override=data_mode), None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the wind_macro_daily backtest")
    parser.add_argument("--data-mode", choices=["synthetic", "wind", "manual_excel"], default=None)
    parser.add_argument("--workbook", default=None, help="manual workbook path")
    parser.add_argument("--run-id", default=None, help="artifact folder name")
    parser.add_argument("--signal-variant", default="composite_signal")
    args = parser.parse_args(argv)

    config = load_config()
    data_mode = args.data_mode or str(config["data"]["mode"])
    data, manual_bundle = _load_for_run(config, data_mode, args.workbook)
    config = _active_config(config, data)
    run_id = args.run_id or (
        f"manual_{manual_bundle.workbook_sha256[:10]}" if manual_bundle else data_mode
    )
    train_size = int(config["walk_forward"]["train_size"])
    test_df = data.iloc[train_size:]
    if test_df.empty:
        raise RuntimeError("not enough data after the reserved training window")
    pit = PointInTimeFrame(data)
    dates = cast(pd.DatetimeIndex, test_df.index)
    positions, instrument_pnl, factor_values = _run_over_dates(
        pit,
        data,
        dates,
        config,
        signal_variant=args.signal_variant,
    )
    portfolio_returns = instrument_pnl.sum(axis=1, min_count=1).fillna(0.0)
    metrics = summarize(portfolio_returns, positions)
    metrics_by_instrument = pd.DataFrame(
        [
            {
                "instrument": instrument,
                **summarize(
                    cast(pd.Series, instrument_pnl[instrument]).fillna(0.0),
                    cast(pd.DataFrame, positions[[instrument]]),
                ),
            }
            for instrument in config["universe"]
        ]
    )
    equity = (1.0 + portfolio_returns).cumprod().rename("equity").to_frame()

    data_path, factor_path, position_path, backtest_path = _artifact_paths(data_mode, run_id)
    report_root = Path("reports") / STRATEGY_NAME
    report_dir = report_root / "backtests" / run_id
    parameter_path = report_root / "parameters" / f"{run_id}.yaml"
    manifest_path = report_root / "manifests" / f"{run_id}.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(data, data_path)
    _write_csv(factor_values, factor_path, index=False)
    _write_csv(positions, position_path)
    _write_csv(instrument_pnl, backtest_path)
    _write_csv(equity, report_dir / "equity_curve.csv")
    _write_csv(metrics_by_instrument, report_dir / "metrics_by_instrument.csv", index=False)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    parameter_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    if manual_bundle is not None:
        manual_dir = data_path.parent
        _write_csv(
            manual_bundle.macro_observations,
            manual_dir / "macro_observations.csv.gz",
            index=False,
        )
        _write_csv(manual_bundle.price_quality, manual_dir / "price_quality.csv", index=False)
        _write_csv(
            manual_bundle.macro_update_audit,
            report_dir / "macro_update_audit.csv",
            index=False,
        )

    latest_metrics = report_root / "metrics.json"
    latest_parameters = report_root / "parameters_snapshot.yaml"
    latest_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    latest_parameters.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts = [
        data_path,
        factor_path,
        position_path,
        backtest_path,
        report_dir / "equity_curve.csv",
        report_dir / "metrics.json",
        report_dir / "metrics_by_instrument.csv",
        parameter_path,
    ]
    if manual_bundle is not None:
        artifacts.extend(
            [
                data_path.parent / "macro_observations.csv.gz",
                data_path.parent / "price_quality.csv",
                report_dir / "macro_update_audit.csv",
            ]
        )
    manifest = {
        "strategy": STRATEGY_NAME,
        "run_id": run_id,
        "data_mode": data_mode,
        "signal_variant": args.signal_variant,
        "active_universe": config["universe"],
        "workbook": (
            {
                "filename": manual_bundle.workbook_path.name,
                "sha256": manual_bundle.workbook_sha256,
                "strict_point_in_time_eligible": False,
                "revision_warning": "release dates guarded; historical revised-vintage leakage remains",
            }
            if manual_bundle
            else None
        ),
        "artifacts": {
            path.as_posix(): {"rows": _line_count(path), "sha256": _sha256(path)}
            for path in artifacts
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {report_dir / 'metrics.json'}: {metrics}")


if __name__ == "__main__":
    main()
