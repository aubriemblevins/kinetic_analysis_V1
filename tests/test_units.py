import pytest

from enzkin.units import (UnitError, best_display_unit, convert_time,
                          parse_quantity)


@pytest.mark.parametrize("text,expected_molar", [
    ("50 uM", 5e-5), ("50uM", 5e-5), ("50 µM", 5e-5), ("50 μM", 5e-5),
    ("50 micromolar", 5e-5), ("3 nM", 3e-9), ("0.5 mM", 5e-4), ("1e-3 M", 1e-3),
    ("100 NM", 1e-7), ("2 pM", 2e-12),
])
def test_molar_spellings(text, expected_molar):
    assert parse_quantity(text).canonical == pytest.approx(expected_molar)


def test_mixed_units_compare_correctly():
    """The whole point: 1000 nM and 1 uM are the same amount."""
    assert parse_quantity("1000 nM").canonical == pytest.approx(
        parse_quantity("1 uM").canonical)


def test_conversion_between_units():
    assert parse_quantity("2.5 uM").to("nM") == pytest.approx(2500)
    assert parse_quantity("0.25 mg/mL").to("ug/mL") == pytest.approx(250)


def test_blank_is_none_but_zero_is_zero():
    assert parse_quantity("") is None
    assert parse_quantity("n/a") is None
    assert parse_quantity("0").canonical == 0.0
    assert parse_quantity(0).canonical == 0.0


def test_default_unit_from_context():
    assert parse_quantity(50, default_unit="uM").canonical == pytest.approx(5e-5)
    assert parse_quantity("50", default_unit="nM").canonical == pytest.approx(5e-8)


def test_bare_number_without_context_is_refused():
    with pytest.raises(UnitError):
        parse_quantity("50")


def test_families_do_not_mix():
    with pytest.raises(UnitError):
        parse_quantity("5 mg/mL").to("nM")


def test_unknown_unit_is_refused():
    with pytest.raises(UnitError):
        parse_quantity("5 bananas")


def test_display_unit_is_readable():
    picked = best_display_unit([parse_quantity(x) for x in
                                ["50 uM", "25 uM", "1.5 uM"]])
    assert picked == "uM"
    picked = best_display_unit([parse_quantity(x) for x in ["0.5 nM", "2 nM"]])
    assert picked == "nM"


def test_time_conversion():
    assert convert_time(90, "s", "min") == pytest.approx(1.5)
    assert convert_time(2, "h", "min") == pytest.approx(120)
