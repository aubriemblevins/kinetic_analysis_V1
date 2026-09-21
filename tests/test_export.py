import pandas as pd
import pytest

from enzkin.export import prism_table, prism_tables, write_outputs


def test_michaelis_menten_layout(example_result):
    table = prism_table(example_result, "substrate", "compound")
    assert table.columns[0] == "Substrate (uM)"
    assert len(table) == 6                      # six substrate concentrations
    assert table.shape[1] == 8                  # X plus seven inhibitor levels
    assert table.iloc[:, 0].is_monotonic_increasing


def test_replicate_subcolumns_sit_side_by_side(example_result):
    table = prism_table(example_result, "compound", "substrate", replicates=True)
    assert table.shape[1] == 1 + 6 * 2          # X plus six datasets of two
    assert table.notna().all().all()


def test_log_x_drops_only_the_zero(example_result):
    plain = prism_table(example_result, "compound", "substrate")
    logged = prism_table(example_result, "compound", "substrate",
                         "percent_activity", log_x=True)
    assert len(logged) == len(plain) - 1
    assert logged.columns[0].startswith("log10")


def test_percent_activity_controls_are_one_hundred(example_result):
    table = prism_table(example_result, "compound", "substrate",
                        "percent_activity")
    controls = table.iloc[0, 1:]
    assert all(abs(value - 100) < 1 for value in controls)


def test_every_layout_is_written(example_result):
    names = prism_tables(example_result)
    assert "prism_substrate_by_compound_mean.csv" in names
    assert "prism_compound_by_substrate_replicates.csv" in names
    assert "prism_compound_by_substrate_log_percent_activity.csv" in names
    assert all(isinstance(table, pd.DataFrame) for table in names.values())


def test_write_outputs_produces_the_whole_folder(example_result, tmp_path):
    written = write_outputs(example_result, tmp_path, figures=True)
    names = {path.name for path in written}
    assert {"well_results.csv", "condition_means.csv", "plate_map_resolved.csv",
            "analysis_report.txt", "plate_curves.png"} <= names
    assert (tmp_path / "prism").is_dir()
    report = (tmp_path / "analysis_report.txt").read_text()
    assert "Linear-range detection" in report
    assert "Wells to look at" in report


def test_well_table_has_a_row_per_analysed_well(example_result):
    table = example_result.well_table()
    assert len(table) == len(example_result.wells)
    assert "rate (RFU/min)" in table.columns
    assert "flags" in table.columns
