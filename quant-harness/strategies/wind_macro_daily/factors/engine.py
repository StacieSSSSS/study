"""Past-only factor snapshots computed from a PointInTimeFrame view."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from core.data.point_in_time import PointInTimeFrame


def _last_z(series: pd.Series, window: int, minimum: int, clip: float) -> float:
    clean = series.dropna().tail(window)
    if len(clean) < minimum:
        return float("nan")
    standard_deviation = float(cast(float, clean.std(ddof=1)))
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-12:
        return 0.0
    last_value = float(cast(float, clean.iloc[-1]))
    mean = float(cast(float, clean.mean()))
    return float(np.clip((last_value - mean) / standard_deviation, -clip, clip))


def factor_snapshot(
    pit: PointInTimeFrame, as_of: pd.Timestamp, config: dict[str, Any]
) -> pd.DataFrame:
    history = pit.as_of(as_of)
    factor_config = config["factors"]
    fast = int(factor_config["fast_window"])
    medium = int(factor_config["medium_window"])
    z_window = int(factor_config["zscore_window"])
    clip = float(factor_config["signal_clip"])
    rows: list[dict[str, object]] = []

    for instrument in config["universe"]:
        spec = config["instruments"][instrument]
        close = cast(pd.Series, history[f"{instrument}__close"]).astype(float)
        if spec["asset_class"] == "FX":
            transformed = pd.Series(np.log(close.to_numpy()), index=close.index)
            momentum_raw = 0.5 * transformed.diff(fast) + 0.5 * transformed.diff(medium)
        else:
            momentum_raw = -(0.5 * close.diff(fast) + 0.5 * close.diff(medium))
        momentum = _last_z(momentum_raw, z_window, max(60, z_window // 3), clip)
        carry = _last_z(
            cast(pd.Series, history[f"{instrument}__carry"]).astype(float),
            z_window,
            max(60, z_window // 3),
            clip,
        )
        mean_reversion = -_last_z(close, medium, max(20, medium // 3), clip)
        macro = _last_z(
            cast(pd.Series, history[f"{instrument}__macro"]).astype(float),
            z_window,
            max(60, z_window // 3),
            clip,
        )
        weights = factor_config["weights"][spec["asset_class"]]
        values = np.array([momentum, carry, mean_reversion, macro], dtype=float)
        factor_weights = np.array(
            [weights["momentum"], weights["carry"], weights["mean_reversion"], weights["macro"]],
            dtype=float,
        )
        composite = float(np.dot(values, factor_weights)) if np.isfinite(values).all() else float("nan")
        rows.append(
            {
                "date": as_of,
                "instrument": instrument,
                "asset_class": spec["asset_class"],
                "momentum": momentum,
                "carry_signal": carry,
                "mean_reversion": mean_reversion,
                "macro_signal": macro,
                "composite_raw": composite,
                "composite_signal": float(np.tanh(composite)) if np.isfinite(composite) else 0.0,
            }
        )
    return pd.DataFrame(rows)
