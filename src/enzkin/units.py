"""Flexible concentration / unit handling.

The guiding rule: a user should never have to convert anything by hand.  A
substrate concentration may be written ``50 uM`` while the inhibitor on the
same plate is written ``3 nM``; both are parsed into a canonical value so they
can be compared, grouped and sorted, and each is rendered back in whatever unit
the user wants to read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Canonical unit per family.  Values are the multiplier to the canonical unit.
_FAMILIES: dict[str, dict[str, float]] = {
    "molar": {
        "M": 1.0,
        "mM": 1e-3,
        "uM": 1e-6,
        "nM": 1e-9,
        "pM": 1e-12,
        "fM": 1e-15,
    },
    "mass_conc": {
        "g/L": 1.0,
        "mg/mL": 1.0,
        "mg/L": 1e-3,
        "ug/mL": 1e-3,
        "ug/uL": 1.0,
        "ug/L": 1e-6,
        "ng/mL": 1e-6,
        "ng/uL": 1e-3,
        "pg/mL": 1e-9,
        "%(w/v)": 10.0,
    },
    "activity": {
        "U/mL": 1.0,
        "mU/mL": 1e-3,
        "U/L": 1e-3,
        "nmol/min/mL": 1e-3,
    },
    "volume_fraction": {
        "%": 1.0,
        "%(v/v)": 1.0,
        "ppm": 1e-4,
    },
    "ratio": {
        "x": 1.0,
        "fold": 1.0,
        "ratio": 1.0,
    },
}

#: Canonical unit symbol for each family.
CANONICAL = {"molar": "M", "mass_conc": "g/L", "activity": "U/mL",
             "volume_fraction": "%", "ratio": "x"}

# Aliases are normalised *before* lookup.  Micro signs, spelled-out names and
# the handful of spellings plate-reader software likes to emit.
_ALIASES = {
    "µ": "u",   # MICRO SIGN
    "μ": "u",   # GREEK SMALL LETTER MU
}
_WORD_UNITS = {
    "molar": "M", "millimolar": "mM", "micromolar": "uM", "nanomolar": "nM",
    "picomolar": "pM", "femtomolar": "fM", "percent": "%", "pct": "%",
    "units/ml": "U/mL", "milliunits/ml": "mU/mL",
}

_UNIT_LOOKUP: dict[str, tuple[str, str]] = {}
for _family, _units in _FAMILIES.items():
    for _u in _units:
        _UNIT_LOOKUP[_u] = (_family, _u)
_UNIT_LOOKUP_CI = {k.lower(): v for k, v in _UNIT_LOOKUP.items()}

_NUMBER_RE = re.compile(
    r"^\s*(?P<num>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>.*?)\s*$"
)

_NULLS = {"", "-", "--", "na", "n/a", "nan", "none", "null", "blank", "empty"}


class UnitError(ValueError):
    """Raised when a unit string cannot be interpreted."""


def normalise_unit(unit: str) -> tuple[str, str]:
    """Return ``(family, canonical_spelling)`` for a unit string.

    Matching is case-sensitive first (so ``mM`` beats ``mm``) and falls back to
    case-insensitive, which is what lets ``50 NM`` or ``5 um`` work.
    """
    raw = (unit or "").strip()
    for bad, good in _ALIASES.items():
        raw = raw.replace(bad, good)
    raw = raw.replace("molar", "molar")  # keep word form for the table below
    if raw == "":
        raise UnitError("empty unit")
    key = raw.replace(" ", "")
    lowered = key.lower()
    if lowered in _WORD_UNITS:
        key = _WORD_UNITS[lowered]
    if key in _UNIT_LOOKUP:
        return _UNIT_LOOKUP[key]
    if key.lower() in _UNIT_LOOKUP_CI:
        return _UNIT_LOOKUP_CI[key.lower()]
    raise UnitError(f"unrecognised unit {unit!r}")


def known_units(family: str | None = None) -> list[str]:
    """List every unit we understand, optionally within one family."""
    if family is None:
        return [u for units in _FAMILIES.values() for u in units]
    return list(_FAMILIES[family])


@dataclass(frozen=True, order=False)
class Quantity:
    """A concentration (or any supported quantity) with its unit remembered.

    ``canonical`` is the value expressed in the family's canonical unit and is
    what grouping, sorting and plotting use internally.
    """

    value: float
    unit: str
    family: str

    @property
    def canonical(self) -> float:
        return self.value * _FAMILIES[self.family][self.unit]

    def to(self, unit: str) -> float:
        """This quantity's magnitude expressed in ``unit``."""
        family, u = normalise_unit(unit)
        if family != self.family:
            raise UnitError(
                f"cannot convert {self.family} quantity to {unit!r} ({family})"
            )
        return self.canonical / _FAMILIES[family][u]

    def format(self, unit: str | None = None, sig: int = 4) -> str:
        unit = unit or self.unit
        try:
            val = self.to(unit)
        except UnitError:
            val, unit = self.value, self.unit
        return f"{_sig(val, sig)} {unit}"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.format()


def _sig(value: float, sig: int = 4) -> str:
    """Format a number with ``sig`` significant digits and no trailing zeros."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    digits = max(0, sig - 1 - math.floor(math.log10(abs(value))))
    if abs(value) >= 1e5 or abs(value) < 1e-4:
        return f"{value:.{max(sig - 1, 0)}e}"
    text = f"{value:.{min(digits, 12)}f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_quantity(
    text: object,
    default_unit: str | None = None,
    family: str | None = None,
) -> Quantity | None:
    """Parse ``"50 uM"``, ``50`` (with ``default_unit``), ``"0"`` or a blank.

    Returns ``None`` for blanks/NA so that "this well has no inhibitor" and
    "I never said" stay distinguishable from a genuine zero.
    """
    if text is None:
        return None
    if isinstance(text, Quantity):
        return text
    if isinstance(text, (int, float)):
        if isinstance(text, float) and math.isnan(text):
            return None
        if default_unit is None:
            if text == 0:  # zero of anything is zero, no unit needed
                fam = family or "molar"
                return Quantity(0.0, CANONICAL[fam], fam)
            raise UnitError(f"{text!r} has no unit and no default was given")
        fam, unit = normalise_unit(default_unit)
        return Quantity(float(text), unit, fam)

    raw = str(text).strip()
    if raw.lower() in _NULLS:
        return None
    match = _NUMBER_RE.match(raw)
    if not match:
        raise UnitError(f"could not parse quantity {text!r}")
    number = float(match.group("num"))
    unit_text = match.group("unit")
    if not unit_text:
        if default_unit is None:
            # A bare 0 is unambiguous: zero of anything is zero.
            if number == 0:
                fam = family or "molar"
                return Quantity(0.0, CANONICAL[fam], fam)
            raise UnitError(f"{text!r} has no unit and no default was given")
        fam, unit = normalise_unit(default_unit)
        return Quantity(number, unit, fam)
    fam, unit = normalise_unit(unit_text)
    if family and fam != family:
        raise UnitError(f"{text!r} is {fam}, expected {family}")
    return Quantity(number, unit, fam)


def canonical_value(q: Quantity | None) -> float | None:
    return None if q is None else q.canonical


def best_display_unit(quantities, family: str | None = None) -> str:
    """Pick the unit that shows a set of concentrations most readably.

    Chooses the unit where the largest non-zero value lands in ``[1, 1000)`` —
    i.e. what a person would have chosen by eye.
    """
    vals = [q.canonical for q in quantities if q is not None and q.canonical > 0]
    fams = {q.family for q in quantities if q is not None}
    if len(fams) == 1:
        family = fams.pop()
    family = family or "molar"
    if not vals:
        return CANONICAL[family]
    top = max(vals)
    best, best_score = CANONICAL[family], math.inf
    for unit, factor in _FAMILIES[family].items():
        shown = top / factor
        if shown <= 0:
            continue
        # Distance (in decades) from the ideal 1-1000 display window.
        score = abs(math.log10(shown) - 1.5)
        if score < best_score:
            best, best_score = unit, score
    return best


# --- time -----------------------------------------------------------------

_TIME_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
               "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
               "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0,
               "hours": 3600.0, "ms": 1e-3, "millisecond": 1e-3,
               "milliseconds": 1e-3}


def time_factor(unit: str) -> float:
    """Seconds per ``unit``."""
    key = (unit or "").strip().lower().lstrip("/")
    if key not in _TIME_UNITS:
        raise UnitError(f"unrecognised time unit {unit!r}")
    return _TIME_UNITS[key]


def convert_time(value: float, frm: str, to: str) -> float:
    return value * time_factor(frm) / time_factor(to)
