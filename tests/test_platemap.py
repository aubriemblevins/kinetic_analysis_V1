import io

import pytest

from enzkin.platemap import (PlateMapError, map_from_layout, read_grid_map,
                             read_long_map)


def test_serial_dilution_across_columns_in_duplicate():
    plate_map = map_from_layout({
        "plate": 96,
        "blocks": [{
            "wells": "A1:A12",
            "compound": {"name": "X", "series": {
                "across": "columns", "replicates": 2,
                "top": "10 uM", "dilution": 3}},
        }],
    })
    assert plate_map.wells["A1"].factor("compound").conc.to("uM") == pytest.approx(10)
    assert plate_map.wells["A2"].factor("compound").conc.to("uM") == pytest.approx(10)
    assert plate_map.wells["A3"].factor("compound").conc.to("uM") == pytest.approx(10 / 3)
    assert plate_map.wells["A11"].factor("compound").conc.to("uM") == pytest.approx(
        10 / 3 ** 5)


def test_series_down_rows():
    plate_map = map_from_layout({
        "plate": 96,
        "blocks": [{"wells": "A1:F1", "substrate": {"series": {
            "across": "rows", "top": "50 uM", "dilution": 2}}}],
    })
    assert plate_map.wells["A1"].factor("substrate").conc.to("uM") == pytest.approx(50)
    assert plate_map.wells["F1"].factor("substrate").conc.to("uM") == pytest.approx(
        50 / 32)


def test_reverse_puts_the_lowest_first():
    plate_map = map_from_layout({
        "plate": 96,
        "blocks": [{"wells": "A1:A4", "compound": {"series": {
            "across": "columns", "top": "8 uM", "dilution": 2, "reverse": True}}}],
    })
    assert plate_map.wells["A1"].factor("compound").conc.to("uM") == pytest.approx(1)
    assert plate_map.wells["A4"].factor("compound").conc.to("uM") == pytest.approx(8)


def test_explicit_values():
    plate_map = map_from_layout({
        "plate": 96,
        "blocks": [{"wells": "A1:A3", "compound": {"series": {
            "across": "columns", "values": ["10 uM", "500 nM", "0"]}}}],
    })
    assert plate_map.wells["A2"].factor("compound").conc.to("nM") == pytest.approx(500)
    assert plate_map.wells["A3"].factor("compound").conc.canonical == 0


def test_wrong_number_of_values_is_explained():
    with pytest.raises(PlateMapError, match="values were given"):
        map_from_layout({"plate": 96, "blocks": [{
            "wells": "A1:A12",
            "compound": {"series": {"across": "columns",
                                    "values": ["1 uM", "2 uM"]}}}]})


def test_later_blocks_paint_over_earlier_ones():
    plate_map = map_from_layout({
        "plate": 96,
        "blocks": [
            {"wells": "A1:A12", "compound": {"name": "X", "conc": "1 uM"}},
            {"wells": "A12", "role": "blank", "compound": {"conc": 0}},
        ],
    })
    assert plate_map.wells["A11"].role == "sample"
    assert plate_map.wells["A12"].role == "blank"


def test_empty_blocks_hold_nothing_not_even_defaults():
    plate_map = map_from_layout({
        "plate": 96,
        "defaults": {"protein": {"name": "E", "conc": "5 nM"}},
        "blocks": [{"wells": "H3:H12", "role": "empty"}],
    })
    assert plate_map.wells["H5"].factors == {}


def test_mixed_units_within_one_plate(example_map):
    """A nanomolar enzyme and a micromolar substrate coexist untouched."""
    well = example_map.wells["A1"]
    assert well.factor("protein").conc.to("nM") == pytest.approx(5)
    assert well.factor("substrate").conc.to("uM") == pytest.approx(50)
    assert well.factor("protein").conc.canonical < well.factor("substrate").conc.canonical


def test_replicate_grouping(example_map):
    groups = example_map.replicate_groups()
    assert len(groups) == 43
    assert all(len(wells) == 2 for wells in groups.values())
    assert set(groups[example_map.wells["A1"].condition_key(
        example_map.active_factors())]) == {"A1", "A2"}


def _csv(text):
    buffer = io.StringIO(text)
    buffer.name = "map.csv"
    return buffer


def test_long_table_map():
    plate_map = read_long_map(_csv(
        "well,protein,protein_conc,compound,compound_conc,substrate,"
        "substrate_conc,role\n"
        "A1,MMP9,5 nM,Cmpd1,10 uM,FRET,50 uM,sample\n"
        "A2,MMP9,5 nM,Cmpd1,10 uM,FRET,50 uM,sample\n"
        "H1,MMP9,5 nM,DMSO,0,FRET,0,blank\n"))
    assert plate_map.wells["A1"].factor("compound").conc.to("nM") == pytest.approx(10000)
    assert plate_map.wells["H1"].role == "blank"
    assert len(plate_map.replicate_groups()) == 2


def test_long_table_takes_the_unit_from_the_header():
    plate_map = read_long_map(_csv(
        "well,substrate,substrate_conc_uM,role\nA1,FRET,50,sample\n"))
    assert plate_map.wells["A1"].factor("substrate").conc.to("uM") == pytest.approx(50)


def test_long_table_accepts_familiar_column_names():
    plate_map = read_long_map(_csv(
        "Well,Enzyme,Enzyme conc,Inhibitor,Inhibitor conc,role\n"
        "A1,MMP9,5 nM,Cmpd1,1 uM,sample\n"))
    well = plate_map.wells["A1"]
    assert well.factor("protein").name == "MMP9"
    assert well.factor("compound").conc.to("uM") == pytest.approx(1)


def test_a_map_without_a_well_column_says_so():
    with pytest.raises(PlateMapError, match="well"):
        read_long_map(_csv("plate,compound\n1,X\n"))


def test_grid_map():
    plate_map = read_grid_map(
        "# substrate_conc_uM\n,1,2\nA,50,25\nB,12.5,6.25\n"
        "# compound\n,1,2\nA,Cmpd1,Cmpd1\nB,Cmpd1,Cmpd1\n")
    assert plate_map.wells["A2"].factor("substrate").conc.to("uM") == pytest.approx(25)
    assert plate_map.wells["B1"].factor("compound").name == "Cmpd1"


def test_exported_long_map_round_trips(example_map, tmp_path):
    """A resolved map written out must read back the same."""
    from enzkin.platemap import read_map
    path = tmp_path / "map.csv"
    example_map.to_csv(path)
    again = read_map(path)
    assert len(again.replicate_groups()) == len(example_map.replicate_groups())
    for well in ("A1", "A11", "F12", "G3", "H1", "H5"):
        before, after = example_map.wells[well], again.wells[well]
        assert before.role == after.role
        for factor in ("protein", "compound", "substrate"):
            first, second = before.factor(factor).conc, after.factor(factor).conc
            assert (first is None) == (second is None)
            if first is not None:
                assert first.canonical == pytest.approx(second.canonical)
