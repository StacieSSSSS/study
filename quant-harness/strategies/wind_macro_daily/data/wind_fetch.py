"""Thin WindPy runner driven by the strategy's series catalog.

This module writes raw extracts only. Promotion into the backtest-safe clean
observation file is intentionally a separate audited step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from strategies.wind_macro_daily.data.loader import load_config


def _import_windpy(windpy_path: str | None) -> Any:
    if windpy_path:
        sys.path.insert(0, windpy_path)
    try:
        module = importlib.import_module("WindPy")
    except ImportError as exc:
        raise RuntimeError("WindPy unavailable; repair the Python API from the Wind terminal") from exc
    return module.w


def _normalise(result: Any, dataset_id: str, wind_code: str) -> pd.DataFrame:
    error_code = int(getattr(result, "ErrorCode", -1))
    if error_code != 0:
        raise RuntimeError(f"Wind request failed for {dataset_id}: ErrorCode={error_code}")
    times = list(getattr(result, "Times", []) or [])
    data = list(getattr(result, "Data", []) or [])
    fields = list(getattr(result, "Fields", []) or [])
    if not times or not data:
        return pd.DataFrame(columns=["reference_ts_utc", "dataset_id", "wind_code", "field", "value"])
    field_names = fields or ["CLOSE"] * len(data)
    rows: list[dict[str, object]] = []
    for field, values in zip(field_names, data):
        for timestamp, value in zip(times, values):
            rows.append(
                {
                    "reference_ts_utc": pd.Timestamp(timestamp, tz="UTC"),
                    "dataset_id": dataset_id,
                    "wind_code": wind_code,
                    "field": field,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def fetch_catalog(start: str, end: str, assets: set[str] | None = None) -> Path:
    config = load_config()
    wind_config = config["data"]["wind"]
    catalog = pd.read_csv(wind_config["catalog_path"]).fillna("")
    catalog = pd.DataFrame(catalog[catalog["enabled"].astype(int).eq(1)])
    if assets:
        catalog = pd.DataFrame(catalog[catalog["asset"].isin(sorted(assets))])
    wind = _import_windpy(wind_config.get("windpy_path"))
    started = wind.start(waitTime=120)
    if int(getattr(started, "ErrorCode", -1)) != 0 or not bool(wind.isconnected()):
        raise RuntimeError(f"Wind session unavailable: ErrorCode={getattr(started, 'ErrorCode', -1)}")

    frames: list[pd.DataFrame] = []
    request_log: list[dict[str, object]] = []
    records: list[dict[str, object]] = catalog.to_dict(orient="records")
    for row in records:
        interface = str(row["interface"]).upper()
        if interface == "WSD":
            result = wind.wsd(
                row["wind_code"], row["fields"], start, end, row["options"]
            )
        elif interface == "EDB":
            result = wind.edb(row["wind_code"], start, end, row["options"])
        else:
            request_log.append({"dataset_id": row["dataset_id"], "status": "SKIPPED", "interface": interface})
            continue
        dataset_id = str(row["dataset_id"])
        wind_code = str(row["wind_code"])
        frame = _normalise(result, dataset_id, wind_code)
        frames.append(frame)
        request_log.append(
            {"dataset_id": dataset_id, "status": "OK", "rows": len(frame), "interface": interface}
        )
    wind.stop()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("strategies/wind_macro_daily/data/raw") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = output_dir / "observations.csv"
    output.to_csv(output_path, index=False)
    (output_dir / "request_log.json").write_text(
        json.dumps(request_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch enabled Wind catalog series")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--assets", default="")
    args = parser.parse_args()
    assets = {item.strip() for item in args.assets.split(",") if item.strip()} or None
    print(fetch_catalog(args.start, args.end, assets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
