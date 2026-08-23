"""Print a compact machine-readable summary of a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path("reports") / args.strategy
    metrics_path = root / "backtests" / args.run_id / "metrics.json"
    summary_path = root / "walk_forward" / args.run_id / "factor_effectiveness.csv"
    windows_path = root / "walk_forward" / args.run_id / "windows_by_instrument_factor.csv"
    status_path = root / "walk_forward" / args.run_id / "harness_status.json"
    missing = [
        str(path) for path in [metrics_path, summary_path, windows_path, status_path] if not path.exists()
    ]
    if missing:
        print(json.dumps({"status": "missing", "paths": missing}, indent=2))
        return 1
    payload = {
        "status": "complete",
        "strategy": args.strategy,
        "run_id": args.run_id,
        "backtest": json.loads(metrics_path.read_text(encoding="utf-8")),
        "walk_forward_summary": pd.read_csv(summary_path).to_dict(orient="records"),
        "walk_forward_windows": len(pd.read_csv(windows_path)),
        "harness_status": json.loads(status_path.read_text(encoding="utf-8")),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
