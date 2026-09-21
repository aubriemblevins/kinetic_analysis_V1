"""Reading kinetic plate-reader exports.

Real exports are messy: a byte-order mark, a few lines of instrument metadata
above the header, times written ``0:01:01`` or ``1.5`` or ``90 s``, and the
occasional ``OVRFLW`` where a number should be.  This module absorbs all of
that and hands back a tidy :class:`KineticData`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .plates import PlateFormat, is_well, normalise_well, sort_wells
from .units import time_factor

_TIME_NAMES = {"time", "t", "elapsed", "elapsedtime", "time(s)", "times",
               "readtime", "kinetictime", "timepoint", "cycletime"}
_NON_NUMERIC = {"ovrflw", "overflow", "sat", "sat.", "saturated", "####",
                "", "-", "na", "n/a", "nan", "none", "null", "masked", "err"}

_HMS_RE = re.compile(r"^\s*(\d+):([0-5]?\d)(?::([0-5]?\d(?:\.\d+)?))?\s*$")
_NUM_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$")


class ReaderError(ValueError):
    """Raised when a file cannot be understood as kinetic plate data."""


@dataclass
class KineticData:
    """Time-course readings for one plate.

    ``time`` is always in seconds; ``values`` is a ``(n_timepoints, n_wells)``
    float array with NaN where a reading was missing or non-numeric.
    """

    time: np.ndarray
    wells: list[str]
    values: np.ndarray
    plate: PlateFormat
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.time), len(self.wells)):
            raise ReaderError(
                f"values shape {self.values.shape} does not match "
                f"{len(self.time)} timepoints x {len(self.wells)} wells"
            )

    @property
    def n_timepoints(self) -> int:
        return len(self.time)

    @property
    def n_wells(self) -> int:
        return len(self.wells)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) else 0.0

    def index_of(self, well: str) -> int:
        return self.wells.index(normalise_well(well))

    def series(self, well: str) -> np.ndarray:
        return self.values[:, self.index_of(well)]

    def time_in(self, unit: str = "min") -> np.ndarray:
        return self.time / time_factor(unit)

    def to_frame(self, time_unit: str = "min") -> pd.DataFrame:
        frame = pd.DataFrame(self.values, columns=self.wells)
        frame.insert(0, f"Time ({time_unit})", self.time_in(time_unit))
        return frame

    def subset(self, wells) -> "KineticData":
        idx = [self.index_of(w) for w in wells]
        return KineticData(self.time, [self.wells[i] for i in idx],
                           self.values[:, idx], self.plate, self.source,
                           list(self.notes))

    def summary(self) -> str:
        return (f"{self.n_wells} wells x {self.n_timepoints} timepoints, "
                f"{self.duration / 60:.1f} min total "
                f"({self.plate.name} plate)")


# --- time parsing ---------------------------------------------------------

def parse_time_value(value: object, default_unit: str = "s") -> float:
    """Parse one time cell into seconds.

    Accepts ``1:05:00`` (h:mm:ss), ``5:30`` (mm:ss), ``90`` (``default_unit``),
    ``1.5 min``, ``90 s`` and pandas/py datetime-ish values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, np.integer, float, np.floating)):
        return float(value) * time_factor(default_unit)
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if hasattr(value, "hour") and hasattr(value, "minute"):  # datetime.time
        return (value.hour * 3600 + value.minute * 60 + value.second
                + getattr(value, "microsecond", 0) / 1e6)

    text = str(value).strip()
    if text.lower() in _NON_NUMERIC:
        return float("nan")
    match = _HMS_RE.match(text)
    if match:
        hours, minutes, seconds = match.group(1), match.group(2), match.group(3)
        if seconds is None:  # mm:ss - the leading field is minutes, not hours
            return int(hours) * 60 + float(minutes)
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    match = _NUM_UNIT_RE.match(text)
    if match:
        number, unit = float(match.group(1)), match.group(2)
        return number * time_factor(unit or default_unit)
    raise ReaderError(f"could not parse the time value {value!r}")


def _looks_like_time_series(values, default_unit="s") -> bool:
    try:
        parsed = [parse_time_value(v, default_unit) for v in values[:8]]
    except ReaderError:
        return False
    parsed = [p for p in parsed if not np.isnan(p)]
    return len(parsed) >= 2 and all(b > a for a, b in zip(parsed, parsed[1:]))


def _to_number(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, np.integer, float, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text.lower() in _NON_NUMERIC:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        # Trailing flags such as "1234*" or "1234 (sat)" still carry a number.
        match = re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text)
        return float(match.group(0)) if match else float("nan")


# --- file loading ---------------------------------------------------------

def _read_raw(path_or_buffer, sheet: str | int | None = None) -> pd.DataFrame:
    """Load a file with no header interpretation at all."""
    name = getattr(path_or_buffer, "name", str(path_or_buffer))
    suffix = Path(str(name)).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path_or_buffer, sheet_name=sheet or 0,
                             header=None, dtype=object)
    if isinstance(path_or_buffer, (bytes, bytearray)):
        path_or_buffer = io.BytesIO(path_or_buffer)
    sep = "\t" if suffix in {".tsv", ".txt"} else None
    return pd.read_csv(path_or_buffer, header=None, dtype=object, sep=sep,
                       engine="python", encoding="utf-8-sig",
                       skip_blank_lines=False)


def _clean(cell: object) -> str:
    return str(cell).replace("﻿", "").strip() if cell is not None else ""


def _find_header_row(raw: pd.DataFrame, max_scan: int = 60) -> int:
    """The first row that carries several well labels (wide) or a time label."""
    best_row, best_score = None, 0
    for i in range(min(max_scan, len(raw))):
        cells = [_clean(c) for c in raw.iloc[i].tolist()]
        wells = sum(1 for c in cells if is_well(c))
        timey = any(c.lower().replace(" ", "") in _TIME_NAMES
                    or c.lower().startswith("time") for c in cells)
        score = wells + (3 if timey else 0)
        if wells >= 2 and score > best_score:
            best_row, best_score = i, score
    if best_row is None:
        raise ReaderError(
            "no header row with well labels (A1, B2, ...) was found; "
            "check that the file is a plate-reader kinetic export"
        )
    return best_row


def read_kinetics(
    path_or_buffer,
    time_unit: str = "s",
    sheet: str | int | None = None,
    plate: int | str | PlateFormat | None = None,
    orientation: str = "auto",
) -> KineticData:
    """Read a kinetic export into :class:`KineticData`.

    Parameters
    ----------
    time_unit:
        How to interpret a *numeric* time column (``0, 1, 2, ...``).  Clock
        strings like ``0:01:01`` are always unambiguous and ignore this.
    orientation:
        ``"wide"`` (wells in columns, the usual export), ``"tall"`` (wells in
        rows, times across the top) or ``"auto"`` to detect.
    """
    raw = _read_raw(path_or_buffer, sheet)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ReaderError("the file is empty")
    raw = raw.reset_index(drop=True)

    if orientation == "auto":
        first_col = [_clean(c) for c in raw.iloc[:, 0].tolist()]
        well_rows = sum(1 for c in first_col if is_well(c))
        header_wells = 0
        try:
            hdr = _find_header_row(raw)
            header_wells = sum(1 for c in raw.iloc[hdr].tolist() if is_well(_clean(c)))
        except ReaderError:
            hdr = None
        orientation = "tall" if well_rows > header_wells else "wide"
        if hdr is None and well_rows == 0:
            raise ReaderError("no well labels found in the file")

    notes: list[str] = []
    if orientation == "tall":
        raw = raw.T.reset_index(drop=True)
        notes.append("Detected wells in rows; the table was transposed.")

    header_row = _find_header_row(raw)
    if header_row > 0:
        notes.append(f"Skipped {header_row} metadata row(s) above the header.")
    header = [_clean(c) for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)

    well_cols = {i: normalise_well(name) for i, name in enumerate(header)
                 if is_well(name)}
    if not well_cols:
        raise ReaderError("the header row contains no well labels")

    time_col = None
    for i, name in enumerate(header):
        if i in well_cols:
            continue
        if name.lower().replace(" ", "") in _TIME_NAMES or name.lower().startswith("time"):
            time_col = i
            break
    if time_col is None:  # fall back to the first column that reads like a clock
        for i in range(len(header)):
            if i in well_cols:
                continue
            if _looks_like_time_series(body.iloc[:, i].tolist(), time_unit):
                time_col = i
                notes.append(
                    f"No column named 'Time'; used column {i + 1} "
                    f"({header[i] or 'unnamed'}) as the time axis."
                )
                break
    if time_col is None:
        raise ReaderError(
            "no time column found; name it 'Time' or put it first in the file"
        )

    header_unit = _column_time_unit(header[time_col]) or time_unit
    times, keep = [], []
    for i, cell in enumerate(body.iloc[:, time_col].tolist()):
        if _clean(cell) == "":
            continue
        try:
            seconds = parse_time_value(cell, header_unit)
        except ReaderError:
            continue  # trailing footer rows such as "End of run"
        if not np.isnan(seconds):
            times.append(seconds)
            keep.append(i)
    if len(times) < 3:
        raise ReaderError(
            f"only {len(times)} usable timepoints were found - a slope needs at least 3"
        )

    order = sorted(range(len(times)), key=lambda k: times[k])
    if order != list(range(len(times))):
        notes.append("Timepoints were out of order and have been sorted.")
    time = np.array([times[k] for k in order], dtype=float)
    rows = [keep[k] for k in order]

    wells = sort_wells(well_cols.values())
    col_for_well = {well: idx for idx, well in well_cols.items()}
    values = np.full((len(time), len(wells)), np.nan)
    for j, well in enumerate(wells):
        column = body.iloc[rows, col_for_well[well]].tolist()
        values[:, j] = [_to_number(v) for v in column]

    fmt = PlateFormat.of(plate) if plate is not None else PlateFormat.infer(wells)
    missing = int(np.isnan(values).sum())
    if missing:
        notes.append(f"{missing} reading(s) were blank or non-numeric and are ignored.")

    return KineticData(time=time, wells=wells, values=values, plate=fmt,
                       source=str(getattr(path_or_buffer, "name", path_or_buffer)),
                       notes=notes)


def _column_time_unit(header: str) -> str | None:
    """Pull a unit out of a header such as ``Time (min)`` or ``Time_s``."""
    match = re.search(r"[\(\[_\s]\s*(ms|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?)\s*[\)\]]?\s*$",
                      header.strip(), re.IGNORECASE)
    return match.group(1).lower() if match else None
