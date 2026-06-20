from strategies.fx_correlation.lib.pairs import combo_label, enumerate_combos


def test_enumerate_combos_is_unordered_pairs():
    combos = enumerate_combos(["A", "B", "C"])
    assert combos == [("A", "B"), ("A", "C"), ("B", "C")]


def test_enumerate_combos_count_matches_n_choose_2():
    universe = ["A", "B", "C", "D"]
    assert len(enumerate_combos(universe)) == 6


def test_combo_label_format():
    assert combo_label(("EURUSD", "GBPUSD")) == "EURUSD-GBPUSD"
