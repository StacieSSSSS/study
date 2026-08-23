"""Harness-standard backtest entrypoint for wind_macro_daily."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from strategies.wind_macro_daily.data.loader import CONFIG_PATH, load_config, load_raw
from strategies.wind_macro_daily.factors.engine import factor_snapshot
from strategies.wind_macro_daily.signals.signal import generate_signal

STRATEGY_NAME = "wind_macro_daily"


def load_data() -> pd.DataFrame:
    return load_raw()


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = config["universe"]
    returns = _returns(history, universe)
    positions = pd.DataFrame(0.0, index=dates, columns=universe)
    factor_frames: list[pd.DataFrame] = []
    risk = config["risk"]

    for as_of in dates:
        view = pit.as_of(as_of)
        signal = generate_signal(pit, as_of, config)
        factor_frames.append(factor_snapshot(pit, as_of, config))
        volatility = (
            _returns(view, universe)
            .tail(int(risk["volatility_window"]))
            .std(ddof=1)
            * np.sqrt(int(risk["annualization"]))
        )
        scalar = (float(risk["target_volatility"]) / volatility).clip(
            upper=float(risk["max_instrument_leverage"])
        )
        target = (signal * scalar).fillna(0.0)
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
    config = load_config()
    history = pd.concat([train_df, test_df]).sort_index()
    pit = PointInTimeFrame(history)
    dates = cast(pd.DatetimeIndex, test_df.index)
    positions, instrument_pnl, _ = _run_over_dates(pit, history, dates, config)
    return summarize(instrument_pnl.sum(axis=1), positions)


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


def main() -> None:
    config = load_config()
    data = load_data()
    train_size = int(config["walk_forward"]["train_size"])
    test_df = data.iloc[train_size:]
    if test_df.empty:
        raise RuntimeError("not enough data after the reserved training window")
    pit = PointInTimeFrame(data)
    dates = cast(pd.DatetimeIndex, test_df.index)
    positions, instrument_pnl, factor_values = _run_over_dates(pit, data, dates, config)
    portfolio_returns = instrument_pnl.sum(axis=1)
    metrics = summarize(portfolio_returns, positions)
    metrics_by_instrument = pd.DataFrame(
        [
            {
                "instrument": instrument,
                **summarize(
                    cast(pd.Series, instrument_pnl[instrument]),
                    cast(pd.DataFrame, positions[[instrument]]),
                ),
            }
            for instrument in config["universe"]
        ]
    )
    equity = (1.0 + portfolio_returns).cumprod().rename("equity").to_frame()

    sample_tag = "sample" if config["data"]["mode"] == "synthetic" else "wind"
    data_path = Path("strategies") / STRATEGY_NAME / "data" / sample_tag / "market_data.csv.gz"
    factor_path = (
        Path("strategies")
        / STRATEGY_NAME
        / "factors"
        / "output"
        / sample_tag
        / "factor_values.csv.gz"
    )
    position_path = (
        Path("strategies") / STRATEGY_NAME / "strategy_output" / sample_tag / "positions.csv.gz"
    )
    backtest_path = (
        Path("strategies") / STRATEGY_NAME / "backtest" / "output" / sample_tag / "daily_pnl.csv.gz"
    )
    report_dir = Path("reports") / STRATEGY_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(data, data_path)
    _write_csv(factor_values, factor_path, index=False)
    _write_csv(positions, position_path)
    _write_csv(instrument_pnl, backtest_path)
    _write_csv(equity, report_dir / "equity_curve.csv")
    _write_csv(metrics_by_instrument, report_dir / "metrics_by_instrument.csv", index=False)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "parameters_snapshot.yaml").write_text(
        CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    artifacts = [
        data_path,
        factor_path,
        position_path,
        backtest_path,
        report_dir / "equity_curve.csv",
        report_dir / "metrics.json",
        report_dir / "metrics_by_instrument.csv",
        report_dir / "parameters_snapshot.yaml",
    ]
    manifest = {
        "strategy": STRATEGY_NAME,
        "data_mode": config["data"]["mode"],
        "warning": config["data"].get("synthetic_warning", ""),
        "artifacts": {
            path.as_posix(): {"rows": _line_count(path), "sha256": _sha256(path)}
            for path in artifacts
        },
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {report_dir / 'metrics.json'}: {metrics}")


if __name__ == "__main__":
    main()
