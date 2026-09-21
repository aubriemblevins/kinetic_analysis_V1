import math

import numpy as np
import pytest

from enzkin import analyse
from enzkin.linearity import DetectionSettings
from enzkin.platemap import map_from_layout
from enzkin.readers import KineticData
from enzkin.plates import PlateFormat


def _synthetic_plate(rates, noise=5.0, seed=4):
    """A plate whose wells rise at rates we chose, so results can be checked."""
    rng = np.random.default_rng(seed)
    time = np.arange(0, 61) * 60.0
    wells = list(rates)
    values = np.column_stack([
        200 + rate * (time / 60) + rng.normal(0, noise, time.size)
        for rate in rates.values()])
    return KineticData(time=time, wells=wells, values=values,
                       plate=PlateFormat.of(96), source="synthetic")


def test_replicate_means_and_spread():
    data = _synthetic_plate({"A1": 20.0, "A2": 22.0, "A3": 10.0, "A4": 10.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "compound": {"name": "X", "conc": "1 uM"}},
        {"wells": "A3,A4", "compound": {"name": "X", "conc": "10 uM"}},
    ]})
    result = analyse(data, plate_map, blank_mode="none")
    by_label = {c.factors["compound"].conc.to("uM"): c for c in result.conditions}
    assert by_label[1.0].n == 2
    assert by_label[1.0].mean == pytest.approx(21.0, rel=0.02)
    assert by_label[1.0].sd == pytest.approx(math.sqrt(2), rel=0.4)
    assert by_label[10.0].mean == pytest.approx(10.0, rel=0.03)
    assert by_label[10.0].cv < 5


def test_blank_rate_is_subtracted():
    data = _synthetic_plate({"A1": 25.0, "A2": 25.0, "H1": 5.0, "H2": 5.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "substrate": {"conc": "50 uM"}},
        {"wells": "H1,H2", "role": "blank", "substrate": {"conc": "50 uM"}},
    ]})
    corrected = analyse(data, plate_map, blank_mode="slope")
    assert corrected.wells["A1"].rate == pytest.approx(20.0, rel=0.05)
    assert corrected.wells["A1"].raw_rate == pytest.approx(25.0, rel=0.05)

    ignored = analyse(data, plate_map, blank_mode="none")
    assert ignored.wells["A1"].rate == pytest.approx(25.0, rel=0.05)


def test_blanks_are_matched_on_the_factor_they_vary_in():
    """Substrate-titrated blanks should correct each sample at its own level."""
    data = _synthetic_plate({"A1": 25.0, "B1": 13.0, "H1": 5.0, "H2": 1.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "substrate": {"conc": "50 uM"}},
        {"wells": "B1", "substrate": {"conc": "5 uM"}},
        {"wells": "H1", "role": "blank", "substrate": {"conc": "50 uM"}},
        {"wells": "H2", "role": "blank", "substrate": {"conc": "5 uM"}},
    ]})
    result = analyse(data, plate_map, blank_mode="slope")
    assert result.wells["A1"].rate == pytest.approx(20.0, rel=0.05)
    assert result.wells["B1"].rate == pytest.approx(12.0, rel=0.06)


def test_percent_activity_uses_the_matching_control():
    data = _synthetic_plate({"A1": 10.0, "A2": 10.0, "G1": 20.0, "G2": 20.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1,A2", "compound": {"name": "X", "conc": "1 uM"},
         "substrate": {"conc": "50 uM"}},
        {"wells": "G1,G2", "role": "positive_control",
         "compound": {"name": "DMSO", "conc": 0}, "substrate": {"conc": "50 uM"}},
    ]})
    result = analyse(data, plate_map, blank_mode="none")
    assert result.wells["A1"].percent_activity == pytest.approx(50, abs=3)
    assert result.wells["A1"].percent_inhibition == pytest.approx(50, abs=3)
    assert result.wells["G1"].percent_activity == pytest.approx(100, abs=3)


def test_rate_unit_scales_the_numbers():
    data = _synthetic_plate({"A1": 60.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "compound": {"conc": "1 uM"}}]})
    per_minute = analyse(data, plate_map, blank_mode="none", rate_unit="min")
    per_second = analyse(data, plate_map, blank_mode="none", rate_unit="s")
    assert per_minute.wells["A1"].rate == pytest.approx(
        per_second.wells["A1"].rate * 60, rel=1e-9)


def test_per_well_override():
    data = _synthetic_plate({"A1": 20.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "A1", "compound": {"conc": "1 uM"}}]})
    result = analyse(data, plate_map, blank_mode="none", well_settings={
        "A1": DetectionSettings(method="fixed", fixed_start=600, fixed_end=1200)})
    assert result.wells["A1"].fit.start_time == pytest.approx(600)
    assert result.wells["A1"].fit.end_time == pytest.approx(1200)


def test_a_map_that_shares_no_wells_with_the_data_says_so():
    data = _synthetic_plate({"A1": 10.0})
    plate_map = map_from_layout({"plate": 96, "blocks": [
        {"wells": "D6", "compound": {"conc": "1 uM"}}]})
    with pytest.raises(ValueError, match="no wells in common|no wells to analyse"):
        analyse(data, plate_map)


# --- against the real example plate --------------------------------------

def test_example_plate_shape(example_result):
    assert len(example_result.conditions) == 43
    assert all(c.n == 2 for c in example_result.conditions)


def test_example_replicates_agree(example_result):
    cvs = [c.cv for c in example_result.conditions
           if c.role == "sample" and c.n == 2 and math.isfinite(c.cv)
           and abs(c.mean) > 1]
    assert max(cvs) < 12, "technical replicates should agree within ~10%"
    assert np.median(cvs) < 5


def test_example_dose_response_is_monotonic(example_result):
    """More inhibitor, less activity - at every substrate concentration."""
    by_substrate: dict[float, list[tuple[float, float]]] = {}
    for condition in example_result.conditions:
        if condition.role not in {"sample", "positive_control"}:
            continue
        substrate = condition.factors["substrate"].conc.canonical
        compound = condition.factors["compound"].conc.canonical
        by_substrate.setdefault(substrate, []).append((compound, condition.mean))
    assert len(by_substrate) == 6
    for series in by_substrate.values():
        rates = [rate for _conc, rate in sorted(series)]
        assert rates == sorted(rates, reverse=True)


def test_example_michaelis_menten_is_monotonic(example_result):
    """More substrate, faster - at every inhibitor concentration.

    Checked with a tolerance tied to each series' own size: at 10 uM inhibitor
    every rate sits at the noise floor (< 0.7 RFU/min), where ordering carries
    no information and demanding it would be testing the noise.
    """
    by_compound: dict[float, list[tuple[float, float]]] = {}
    for condition in example_result.conditions:
        if condition.role != "sample":
            continue
        by_compound.setdefault(
            condition.factors["compound"].conc.canonical, []).append(
            (condition.factors["substrate"].conc.canonical, condition.mean))
    for series in by_compound.values():
        rates = [rate for _conc, rate in sorted(series)]
        tolerance = 0.05 * max(rates)
        assert all(later >= earlier - tolerance
                   for earlier, later in zip(rates, rates[1:])), rates


def test_example_controls_are_the_fastest_wells(example_result):
    controls = [c.mean for c in example_result.conditions
                if c.role == "positive_control"]
    inhibited = [c.mean for c in example_result.conditions
                 if c.role == "sample"]
    assert max(controls) >= max(inhibited)


def test_example_fully_inhibited_wells_are_not_inflated(example_result):
    """The 10 uM column should read near zero, not a noise-driven slope."""
    for well in ("A1", "A2", "F1", "F2"):
        assert abs(example_result.wells[well].rate) < 1.5


def test_example_blank_is_flat(example_result):
    assert abs(example_result.wells["H1"].raw_rate) < 0.5
