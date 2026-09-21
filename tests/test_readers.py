import io

import numpy as np
import pytest

from enzkin.readers import ReaderError, parse_time_value, read_kinetics


@pytest.mark.parametrize("text,seconds", [
    ("0:01:01", 61), ("1:00:00", 3600), ("2:01:01", 7261),
    ("05:30", 330), ("90", 90), ("1.5 min", 90), ("90 s", 90), ("0.5 h", 1800),
])
def test_time_spellings(text, seconds):
    assert parse_time_value(text) == pytest.approx(seconds)


def test_numeric_time_honours_the_stated_unit():
    assert parse_time_value(5, default_unit="min") == pytest.approx(300)


def _csv(text):
    buffer = io.StringIO(text)
    buffer.name = "test.csv"
    return buffer


def test_reads_a_plain_file():
    data = read_kinetics(_csv("Time,A1,A2\n0,10,20\n60,20,30\n120,30,40\n"))
    assert data.wells == ["A1", "A2"]
    assert data.time.tolist() == [0, 60, 120]
    assert data.series("A1").tolist() == [10, 20, 30]


def test_skips_instrument_metadata_above_the_header():
    data = read_kinetics(_csv(
        "Experiment,Run 7\nPlate,Greiner\n\nTime,A1,A2\n"
        "0:00:30,5,6\n0:01:30,7,9\n0:02:30,9,12\n"))
    assert data.n_wells == 2
    assert data.time[0] == pytest.approx(30)
    assert any("metadata" in note for note in data.notes)


def test_handles_a_byte_order_mark():
    data = read_kinetics(_csv("﻿Time,A1\n0,1\n60,2\n120,3\n"))
    assert data.wells == ["A1"]


def test_non_numeric_readings_become_gaps():
    data = read_kinetics(_csv("Time,A1\n0,10\n60,OVRFLW\n120,30\n180,\n240,50\n"))
    series = data.series("A1")
    assert np.isnan(series[1]) and np.isnan(series[3])
    assert any("non-numeric" in note for note in data.notes)


def test_wells_in_rows_are_transposed():
    data = read_kinetics(_csv("Well,0,60,120\nA1,10,20,30\nA2,5,6,7\n"))
    assert data.wells == ["A1", "A2"]
    assert data.series("A1").tolist() == [10, 20, 30]


def test_out_of_order_timepoints_are_sorted():
    data = read_kinetics(_csv("Time,A1\n120,30\n0,10\n60,20\n"))
    assert data.time.tolist() == [0, 60, 120]
    assert data.series("A1").tolist() == [10, 20, 30]


def test_header_unit_is_respected():
    data = read_kinetics(_csv("Time (min),A1\n0,1\n1,2\n2,3\n"))
    assert data.time.tolist() == [0, 60, 120]


def test_a_file_with_no_wells_is_refused():
    with pytest.raises(ReaderError):
        read_kinetics(_csv("Time,Temperature\n0,37\n60,37\n120,37\n"))


def test_too_few_timepoints_is_refused():
    with pytest.raises(ReaderError):
        read_kinetics(_csv("Time,A1\n0,1\n60,2\n"))


def test_example_file(example_data):
    assert example_data.n_wells == 96
    assert example_data.n_timepoints == 121
    assert example_data.plate.size == 96
    assert example_data.duration == pytest.approx(7200)
