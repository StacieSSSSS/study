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
SIGNAL_VARIANTS = (
    "momentum",
    "carry_signal",
    "mean_reversion",
    "macro_signal",
    "composite_signal",
)
INDICATOR_LABELS = {
    "momentum": "20/60日价格趋势（利率为反向收益率变化）",
    "carry_signal": "持有收益/曲线滚降",
    "mean_reversion": "60日价格或收益率水平Z分数反转",
    "macro_signal": "发布时点对齐的宏观篮子Z分数",
    "composite_signal": "可用因子动态再加权综合信号",
}


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


def _single_instrument_config(config: dict[str, Any], instrument: str) -> dict[str, Any]:
    if instrument not in config["universe"]:
        raise ValueError(f"instrument is not active: {instrument}")
    run_config = copy.deepcopy(config)
    run_config["universe"] = [instrument]
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


def run_independent_grid(
    pit: PointInTimeFrame,
    history: pd.DataFrame,
    run_dates: pd.DatetimeIndex,
    evaluation_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    variants: list[str] | tuple[str, ...] = SIGNAL_VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest every instrument/factor independently, never as a portfolio sleeve."""
    _ = pit, run_dates  # API symmetry; vectorized calculations remain strictly backward-looking.
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    factor_config = config["factors"]
    risk = config["risk"]
    fast = int(factor_config["fast_window"])
    medium = int(factor_config["medium_window"])
    z_window = int(factor_config["zscore_window"])
    clip = float(factor_config["signal_clip"])

    def rolling_z(series: pd.Series, window: int, minimum: int) -> pd.Series:
        rolling = series.rolling(window, min_periods=minimum)
        standard_deviation = rolling.std(ddof=1).replace(0.0, np.nan)
        return cast(pd.Series, ((series - rolling.mean()) / standard_deviation).clip(-clip, clip))

    for instrument in config["universe"]:
        spec = config["instruments"][instrument]
        close = cast(pd.Series, history[f"{instrument}__close"]).astype(float)
        instrument_return = cast(pd.Series, history[f"{instrument}__return"]).astype(float)
        is_rate = spec["asset_class"] in {"IRS", "UST"}
        transformed = close if is_rate else cast(pd.Series, np.log(close))
        momentum_raw = 0.5 * transformed.diff(fast) + 0.5 * transformed.diff(medium)
        if is_rate:
            momentum_raw = -momentum_raw
        momentum = rolling_z(momentum_raw, z_window, max(60, z_window // 3))
        carry = rolling_z(
            cast(pd.Series, history[f"{instrument}__carry"]).astype(float),
            z_window,
            max(60, z_window // 3),
        )
        level_z = rolling_z(close, medium, max(20, medium // 3))
        mean_reversion = level_z if is_rate else -level_z
        macro = rolling_z(
            cast(pd.Series, history[f"{instrument}__macro"]).astype(float),
            z_window,
            max(60, z_window // 3),
        )
        components = pd.DataFrame(
            {
                "momentum": momentum,
                "carry_signal": carry,
                "mean_reversion": mean_reversion,
                "macro_signal": macro,
            },
            index=history.index,
        )
        weights = pd.Series(
            {
                "momentum": factor_config["weights"][spec["asset_class"]]["momentum"],
                "carry_signal": factor_config["weights"][spec["asset_class"]]["carry"],
                "mean_reversion": factor_config["weights"][spec["asset_class"]]["mean_reversion"],
                "macro_signal": factor_config["weights"][spec["asset_class"]]["macro"],
            },
            dtype=float,
        )
        available_weight = components.notna().mul(weights, axis=1).sum(axis=1)
        composite_raw = components.mul(weights, axis=1).sum(axis=1, min_count=1).div(
            available_weight.replace(0.0, np.nan)
        )
        components["composite_signal"] = np.tanh(composite_raw)
        annualized_volatility = (
            instrument_return.rolling(int(risk["volatility_window"]), min_periods=2).std(ddof=1)
            * np.sqrt(int(risk["annualization"]))
        )
        volatility_scalar = (
            float(risk["target_volatility"]) / annualized_volatility
        ).clip(upper=float(risk["max_instrument_leverage"]))
        cost_rate = float(spec["cost_bps"]) / 10_000.0
        for variant in variants:
            factor = cast(pd.Series, components[str(variant)]).astype(float)
            signal = factor if variant == "composite_signal" else cast(pd.Series, np.tanh(factor))
            position = (signal * volatility_scalar).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            strategy_return = position.shift(1).fillna(0.0) * instrument_return.fillna(0.0)
            strategy_return -= position.diff().abs().fillna(position.abs()) * cost_rate
            evaluated_position = cast(pd.Series, position.loc[evaluation_dates])
            evaluated_positions = evaluated_position.rename(instrument).to_frame()
            evaluated_return = cast(pd.Series, strategy_return.loc[evaluation_dates]).fillna(0.0)
            active_days = int(
                np.count_nonzero(np.abs(evaluated_position.to_numpy(dtype=float)) > 0.0)
            )
            metric_rows.append(
                {
                    "instrument": instrument,
                    "asset_class": config["instruments"][instrument]["asset_class"],
                    "signal_variant": variant,
                    "technical_indicator": INDICATOR_LABELS[str(variant)],
                    "active_position_days": active_days,
                    "status": "evaluated" if active_days else "unavailable_input_or_no_signal",
                    **summarize(evaluated_return, evaluated_positions),
                }
            )
            daily_rows.append(
                pd.DataFrame(
                    {
                        "date": evaluation_dates,
                        "instrument": instrument,
                        "signal_variant": variant,
                        "daily_return": evaluated_return.to_numpy(),
                        "position": evaluated_position.to_numpy(),
                    }
                )
            )
    return pd.DataFrame(metric_rows), pd.concat(daily_rows, ignore_index=True)


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
    start = max(train_size - 1, 0)
    run_dates = cast(pd.DatetimeIndex, data.index[start:])
    run_positions, run_instrument_pnl, factor_values = _run_over_dates(
        pit,
        data,
        run_dates,
        config,
        signal_variant=args.signal_variant,
    )
    positions = cast(pd.DataFrame, run_positions.loc[dates])
    instrument_pnl = cast(pd.DataFrame, run_instrument_pnl.loc[dates])
    portfolio_returns = instrument_pnl.sum(axis=1, min_count=1).fillna(0.0)
    metrics = summarize(portfolio_returns, positions)
    independent_metrics, independent_daily = run_independent_grid(
        pit,
        data,
        run_dates,
        dates,
        config,
    )
    metrics_by_instrument = cast(
        pd.DataFrame,
        independent_metrics.loc[
            independent_metrics["signal_variant"].eq("composite_signal"),
            ["instrument", "sharpe", "max_drawdown", "annualized_return", "turnover"],
        ],
    ).reset_index(drop=True)
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
    _write_csv(
        independent_metrics,
        report_dir / "metrics_by_instrument_factor.csv",
        index=False,
    )
    _write_csv(
        independent_daily,
        report_dir / "daily_returns_by_instrument_factor.csv.gz",
        index=False,
    )
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
        report_dir / "metrics_by_instrument_factor.csv",
        report_dir / "daily_returns_by_instrument_factor.csv.gz",
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
