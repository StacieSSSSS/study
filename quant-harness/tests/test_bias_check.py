from pathlib import Path

from core.validation.bias_check import check_source


def test_detects_negative_shift():
    findings = check_source("df['x'].shift(-1)\n", Path("strategy.py"))
    assert len(findings) == 1
    assert "shift" in findings[0].message


def test_detects_negative_shift_via_keyword():
    findings = check_source("df['x'].shift(periods=-2)\n", Path("strategy.py"))
    assert len(findings) == 1


def test_detects_bfill_method_call():
    findings = check_source("df['x'].bfill()\n", Path("strategy.py"))
    assert len(findings) == 1
    assert "bfill" in findings[0].message


def test_detects_fillna_bfill_keyword():
    findings = check_source("df['x'].fillna(method='bfill')\n", Path("strategy.py"))
    assert len(findings) == 1


def test_positive_shift_is_not_flagged():
    findings = check_source("df['x'].shift(1)\n", Path("strategy.py"))
    assert findings == []


def test_ffill_is_not_flagged():
    findings = check_source("df['x'].ffill()\n", Path("strategy.py"))
    assert findings == []


def test_clean_file_has_no_findings(tmp_path):
    file = tmp_path / "clean.py"
    file.write_text("def f(x):\n    return x.shift(1).ffill()\n")
    from core.validation.bias_check import check_file

    assert check_file(file) == []
