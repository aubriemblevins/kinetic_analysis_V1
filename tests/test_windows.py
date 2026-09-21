import numpy as np
import pytest

from enzkin import analyse
from enzkin.platemap import map_from_layout
from enzkin.plates import PlateFormat
from enzkin.readers import KineticData
from enzkin.windows import WindowPolicy, plan_windows

MINUTE = 60.0


def _plate(curves, n=61, seed=2, noise=4.0):
    """Build a plate from {well: callable(minutes) -> signal}."""
    rng = np.random.default_rng(seed)
    time = np.arange(0, n) * MINUTE
    minutes = time / MINUTE
    wells = list(curves)
    values = np.column_stack([
        curves[w](minutes) + rng.normal(0, noise, minutes.size) for w in wells])
    return KineticData(time=time, wells=wells, values=values,
                       plate=PlateFormat.of(96), source="synthetic")


def _linear(rate, start=200.0):
    return lambda m: start + rate * m


def _bends(amplitude, k, start=200.0):
    """Fast, and flattening out - what an uninhibited control does."""
    return lambda m: start + amplitude * (1 - np.exp(-k * m))


# --- the central behaviour -----------------------------------------------

def test_the_control_decides_the_window_not_each_well():
    """A control that bends at ~20 min must pull the slow wells in with it.

    Left to themselves the inhibited wells are straight for the full hour and
    would be fitted over 0-60 min while their control used 0-20 - so the two
    would be measured over different stretches of the same reaction.
    """
    curves = {"G1": _bends(3000, 0.08), "G2": _bends(3000, 0.08),
              "A1": _linear(8.0), "A2": _linear(8.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "substrate": {"conc": "50 uM"},
         "compound": {"name": "X", "conc": "1 uM"}},
        {"wells": "G1,G2", "role": "positive_control",
         "substrate": {"conc": "50 uM"}, "compound": {"name": "DMSO", "conc": 0}},
    ]})

    alone = analyse(data, plate_map, blank_mode="none",
                    window_policy=WindowPolicy(scope="well"))
    assert alone.wells["A1"].fit.end_time > alone.wells["G1"].fit.end_time, \
        "the premise: on its own the slow well would use a longer window"

    shared = analyse(data, plate_map, blank_mode="none")
    windows = {(w.fit.start_index, w.fit.end_index) for w in shared.wells.values()}
    assert len(windows) == 1, "every well must be fitted over the same readings"
    assert shared.wells["A1"].fit.end_time <= alone.wells["G1"].fit.end_time + 1e-6


def test_every_well_on_the_example_plate_shares_one_window(example_result):
    windows = {(w.fit.start_index, w.fit.end_index)
               for w in example_result.wells.values() if w.fit}
    assert len(windows) == 1
    assert example_result.window_plan.is_uniform


def test_controls_are_the_reference_wells(example_result):
    plan = example_result.window_plan
    with_controls = [g for g in plan.groups if g.reference_kind == "controls"]
    assert len(with_controls) == 6          # one per substrate concentration
    for group in with_controls:
        assert all(w.startswith("G") for w in group.reference_wells)


def test_uniform_windows_do_not_cost_precision(example_result):
    """Sharing a window must not make the replicates agree any worse."""
    import math
    cvs = [c.cv for c in example_result.conditions
           if c.role == "sample" and abs(c.mean) > 1 and math.isfinite(c.cv)]
    assert np.median(cvs) < 5 and max(cvs) < 12


# --- scopes ---------------------------------------------------------------

def test_scope_group_matches_on_substrate():
    """Two substrate levels whose controls bend at very different times."""
    curves = {"G1": _bends(3000, 0.30), "G2": _bends(3000, 0.30),   # bends early
              "G3": _bends(3000, 0.02), "G4": _bends(3000, 0.02),   # stays linear
              "A1": _linear(9.0), "A2": _linear(9.0),
              "B1": _linear(4.0), "B2": _linear(4.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "substrate": {"conc": "5 uM"},
         "compound": {"name": "X", "conc": "1 uM"}},
        {"wells": "B1,B2", "substrate": {"conc": "50 uM"},
         "compound": {"name": "X", "conc": "1 uM"}},
        {"wells": "G1,G2", "role": "positive_control",
         "substrate": {"conc": "5 uM"}, "compound": {"conc": 0}},
        {"wells": "G3,G4", "role": "positive_control",
         "substrate": {"conc": "50 uM"}, "compound": {"conc": 0}},
    ]})
    result = analyse(data, plate_map, blank_mode="none",
                     window_policy=WindowPolicy(scope="group"))
    low = result.wells["A1"].fit
    high = result.wells["B1"].fit
    assert low.end_time < high.end_time, \
        "the group whose control bends early must use a shorter window"
    assert result.wells["A1"].fit.end_index == result.wells["G1"].fit.end_index
    assert result.wells["B1"].fit.end_index == result.wells["G3"].fit.end_index


def test_auto_falls_back_to_group_when_no_single_window_suits():
    curves = {"G1": _bends(3000, 0.30), "G2": _bends(3000, 0.30),
              "G3": _bends(3000, 0.02), "G4": _bends(3000, 0.02),
              "A1": _linear(9.0), "B1": _linear(4.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "substrate": {"conc": "5 uM"}, "compound": {"conc": "1 uM"}},
        {"wells": "B1", "substrate": {"conc": "50 uM"}, "compound": {"conc": "1 uM"}},
        {"wells": "G1,G2", "role": "positive_control",
         "substrate": {"conc": "5 uM"}, "compound": {"conc": 0}},
        {"wells": "G3,G4", "role": "positive_control",
         "substrate": {"conc": "50 uM"}, "compound": {"conc": 0}},
    ]})
    plan = plan_windows(data, plate_map)
    assert plan.scope_used == "group"
    assert any("No single window suits the whole plate" in n for n in plan.notes)


def test_scope_plate_forces_one_window_even_then():
    curves = {"G1": _bends(3000, 0.30), "G3": _bends(3000, 0.02),
              "A1": _linear(9.0), "B1": _linear(4.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "substrate": {"conc": "5 uM"}, "compound": {"conc": "1 uM"}},
        {"wells": "B1", "substrate": {"conc": "50 uM"}, "compound": {"conc": "1 uM"}},
        {"wells": "G1", "role": "positive_control", "substrate": {"conc": "5 uM"},
         "compound": {"conc": 0}},
        {"wells": "G3", "role": "positive_control", "substrate": {"conc": "50 uM"},
         "compound": {"conc": 0}},
    ]})
    result = analyse(data, plate_map, blank_mode="none",
                     window_policy=WindowPolicy(scope="plate"))
    windows = {(w.fit.start_index, w.fit.end_index) for w in result.wells.values()}
    assert len(windows) == 1


def test_scope_well_keeps_per_well_ranges(example_data, example_map):
    result = analyse(example_data, example_map,
                     window_policy=WindowPolicy(scope="well"))
    assert result.window_plan.scope_used == "well"
    assert result.window_plan.well_window == {}


# --- references -----------------------------------------------------------

def test_a_group_with_no_control_uses_its_fastest_wells():
    curves = {"A1": _bends(3000, 0.25), "A2": _bends(3000, 0.25),
              "A3": _linear(3.0), "A4": _linear(3.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "substrate": {"conc": "50 uM"},
         "compound": {"name": "X", "conc": "0.1 uM"}},
        {"wells": "A3,A4", "substrate": {"conc": "50 uM"},
         "compound": {"name": "X", "conc": "10 uM"}},
    ]})
    plan = plan_windows(data, plate_map)
    group = plan.groups[0]
    assert group.reference_kind == "fastest wells"
    assert set(group.reference_wells) == {"A1", "A2"}
    assert any("fastest wells in the group were used" in n for n in group.notes)


def test_a_group_with_nothing_to_measure_does_not_constrain_the_plate():
    """Flat blanks must not drag the window in - they carry no information."""
    curves = {"G1": _linear(20.0), "G2": _linear(20.0),
              "H1": _linear(0.0), "H2": _linear(0.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "G1,G2", "role": "positive_control",
         "substrate": {"conc": "50 uM"}, "compound": {"conc": 0}},
        {"wells": "H1,H2", "role": "blank", "substrate": {"conc": 0},
         "compound": {"conc": 0}},
    ]})
    plan = plan_windows(data, plate_map)
    blank_group = next(g for g in plan.groups
                       if g.factors["substrate"].conc.canonical == 0)
    assert "no usable reference" in blank_group.notes
    assert plan.plate_window == (0, data.n_timepoints - 1)


def test_match_factors_can_be_chosen():
    plan = plan_windows(*_simple(), policy=WindowPolicy(match_on=["substrate"]))
    assert plan.match_on == ["substrate"]


def _simple():
    curves = {"G1": _linear(20.0), "A1": _linear(5.0)}
    data = _plate(curves)
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "substrate": {"conc": "50 uM"},
         "compound": {"name": "X", "conc": "1 uM"}},
        {"wells": "G1", "role": "positive_control",
         "substrate": {"conc": "50 uM"}, "compound": {"conc": 0}},
    ]})
    return data, plate_map


# --- reporting ------------------------------------------------------------

def test_plan_rows_describe_every_group(example_result):
    rows = example_result.window_plan.to_rows()
    assert len(rows) == len(example_result.window_plan.groups)
    assert {"group", "reference wells", "applied window (min)",
            "readings used"} <= set(rows[0])


def test_the_report_explains_the_window(example_result, tmp_path):
    from enzkin.export import write_outputs
    write_outputs(example_result, tmp_path, figures=False, excel=False)
    report = (tmp_path / "analysis_report.txt").read_text()
    assert "Window sharing" in report
    assert "The window each matched group is fitted over" in report
    assert (tmp_path / "linear_range_windows.csv").exists()


def test_the_window_figure_renders(example_result, tmp_path):
    from enzkin import plotting
    path = plotting.plot_window_choice(example_result,
                                       tmp_path / "window_choice.png")
    assert path.exists() and path.stat().st_size > 30_000


def test_the_window_figure_refuses_when_there_is_nothing_to_justify(
        example_data, example_map, tmp_path):
    from enzkin import plotting
    result = analyse(example_data, example_map,
                     window_policy=WindowPolicy(scope="well"))
    with pytest.raises(ValueError, match="own range"):
        plotting.plot_window_choice(result, tmp_path / "x.png")
