"""Enumerate the tradable units: combinations of two FX pairs."""

from __future__ import annotations

from itertools import combinations


def enumerate_combos(universe: list[str]) -> list[tuple[str, str]]:
    """All unordered combinations of two pairs from the universe, e.g.
    ["EURUSD", "GBPUSD"] -> [("EURUSD", "GBPUSD")].
    """
    return list(combinations(universe, 2))


def combo_label(combo: tuple[str, str]) -> str:
    return f"{combo[0]}-{combo[1]}"
