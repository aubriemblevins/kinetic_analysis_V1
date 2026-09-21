import pytest

from enzkin.cli import main


def test_check_reports_the_file(capsys, tmp_path):
    from tests.conftest import EXAMPLE_DATA
    assert main(["check", str(EXAMPLE_DATA)]) == 0
    printed = capsys.readouterr().out
    assert "96 wells" in printed and "121 timepoints" in printed


def test_analyse_writes_results(tmp_path, capsys):
    from tests.conftest import EXAMPLE_DATA, EXAMPLE_LAYOUT
    code = main(["analyse", str(EXAMPLE_DATA), "--map", str(EXAMPLE_LAYOUT),
                 "--out", str(tmp_path), "--no-figures", "--no-excel"])
    assert code == 0
    assert (tmp_path / "condition_means.csv").exists()
    assert (tmp_path / "prism").is_dir()
    assert "43 conditions" in capsys.readouterr().out


def test_starter_layout_round_trips(tmp_path):
    from enzkin.platemap import read_map
    path = tmp_path / "layout.yaml"
    assert main(["new-layout", "-o", str(path), "--plate", "384"]) == 0
    plate_map = read_map(path)
    assert plate_map.plate.size == 384
    assert plate_map.validate() is not None


def test_blank_map_has_a_row_per_well(tmp_path):
    import pandas as pd
    path = tmp_path / "map.csv"
    assert main(["new-map", "-o", str(path), "--plate", "96"]) == 0
    frame = pd.read_csv(path)
    assert len(frame) == 96
    assert "substrate_conc" in frame.columns


def test_a_missing_file_is_reported_not_raised(capsys, tmp_path):
    assert main(["check", str(tmp_path / "nope.csv")]) == 2
    assert "Could not read" in capsys.readouterr().err
