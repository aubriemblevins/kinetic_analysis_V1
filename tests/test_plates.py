import pytest

from enzkin.plates import (PlateError, PlateFormat, expand_wells,
                           normalise_well, sort_wells)


@pytest.mark.parametrize("text,expected", [
    ("a1", "A1"), ("A01", "A1"), ("A 1", "A1"), ("h12", "H12"), ("P24", "P24"),
    ("AA48", "AA48"),
])
def test_well_spellings(text, expected):
    assert normalise_well(text) == expected


def test_plate_inference():
    assert PlateFormat.infer(["A1", "H12"]).size == 96
    assert PlateFormat.infer(["A1", "P24"]).size == 384
    assert PlateFormat.infer(["A1", "B2"]).size == 6


def test_rectangular_range():
    assert expand_wells("A1:B3") == ["A1", "A2", "A3", "B1", "B2", "B3"]


def test_row_and_column_specs():
    plate = PlateFormat.of(96)
    assert len(expand_wells("G", plate)) == 12
    assert len(expand_wells("A:F", plate)) == 72
    assert expand_wells("11:12", plate)[:2] == ["A11", "A12"]
    assert len(expand_wells("all", plate)) == 96


def test_mixed_specification_deduplicates():
    assert expand_wells("A1, A1:A3, A2") == ["A1", "A2", "A3"]


def test_row_spec_needs_a_plate():
    with pytest.raises(PlateError):
        expand_wells("G")


def test_wells_sort_like_a_plate_not_like_strings():
    assert sort_wells(["A10", "A2", "B1"]) == ["A2", "A10", "B1"]
