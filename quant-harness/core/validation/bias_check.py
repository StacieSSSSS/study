"""Static scanner for common look-ahead ("future function") bias patterns.

This is a heuristic, not a proof of correctness — it catches the classic
mistakes (negative shift, backward-fill across time, future-leaning method
names) so they can't slip into a strategy unnoticed. Real prevention of
look-ahead bias relies on routing data access through
``core.data.point_in_time.PointInTimeFrame``; this scanner is the safety net.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    file: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.message}"


def _is_negative_number(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    return False


def _check_call(node: ast.Call, file: Path) -> list[Finding]:
    findings: list[Finding] = []
    func = node.func
    if not isinstance(func, ast.Attribute):
        return findings

    attr = func.attr

    if attr == "shift":
        msg = "shift() with a negative period leaks future data"
        for arg in node.args:
            if _is_negative_number(arg):
                findings.append(Finding(file, node.lineno, msg))
        for kw in node.keywords:
            if kw.arg == "periods" and _is_negative_number(kw.value):
                findings.append(Finding(file, node.lineno, "shift(periods=negative) leaks future data"))

    if attr == "bfill":
        findings.append(Finding(file, node.lineno, "bfill() pulls future values backward in time"))

    if attr == "fillna":
        bfill_msg = "fillna(method='bfill') pulls future values backward in time"
        for kw in node.keywords:
            if kw.arg == "method" and isinstance(kw.value, ast.Constant) and kw.value.value == "bfill":
                findings.append(Finding(file, node.lineno, bfill_msg))

    return findings


def check_source(source: str, file: Path) -> list[Finding]:
    tree = ast.parse(source, filename=str(file))
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            findings.extend(_check_call(node, file))
    return findings


def check_file(path: Path) -> list[Finding]:
    return check_source(path.read_text(), path)


def check_path(path: Path) -> list[Finding]:
    if path.is_file():
        return check_file(path)
    findings: list[Finding] = []
    for py_file in sorted(path.rglob("*.py")):
        findings.extend(check_file(py_file))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan strategy code for look-ahead bias patterns")
    parser.add_argument("path", type=Path, help="strategy directory or file to scan")
    args = parser.parse_args(argv)

    findings = check_path(args.path)
    for finding in findings:
        print(finding)

    if findings:
        print(f"bias-check FAILED: {len(findings)} finding(s) in {args.path}")
        return 1

    print(f"bias-check OK: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
