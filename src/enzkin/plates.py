"""Plate geometry and well addressing.

Wells are always stored in the canonical ``A1`` form (letter row, un-padded
column) but any common spelling is accepted on input: ``a01``, ``A 1``,
``AA12`` (1536-well), ``A01``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Supported plate formats, keyed by well count.
PLATE_FORMATS: dict[int, tuple[int, int]] = {
    6: (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48),
}

_WELL_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*0*(\d{1,2})\s*$")


class PlateError(ValueError):
    """Raised for malformed wells, ranges or plate formats."""


def row_label(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA' (the 1536-well convention)."""
    if index < 26:
        return chr(ord("A") + index)
    return "A" + chr(ord("A") + index - 26)


def row_index(label: str) -> int:
    label = label.strip().upper()
    if len(label) == 1:
        return ord(label) - ord("A")
    if len(label) == 2 and label[0] == "A":
        return 26 + ord(label[1]) - ord("A")
    raise PlateError(f"unrecognised row label {label!r}")


def parse_well(text: str) -> tuple[int, int]:
    """``'b7'`` -> ``(1, 6)`` as zero-based ``(row, column)``."""
    match = _WELL_RE.match(str(text))
    if not match:
        raise PlateError(f"{text!r} is not a well id")
    col = int(match.group(2))
    if col < 1:
        raise PlateError(f"{text!r} has a zero/negative column")
    return row_index(match.group(1)), col - 1


def is_well(text: object) -> bool:
    try:
        parse_well(str(text))
        return True
    except PlateError:
        return False


def well_name(row: int, col: int) -> str:
    return f"{row_label(row)}{col + 1}"


def normalise_well(text: str) -> str:
    row, col = parse_well(text)
    return well_name(row, col)


@dataclass(frozen=True)
class PlateFormat:
    """A plate's shape.  ``PlateFormat.of(96)`` is the usual way in."""

    n_rows: int
    n_cols: int

    @property
    def size(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def name(self) -> str:
        return f"{self.size}-well"

    @classmethod
    def of(cls, spec: "int | str | PlateFormat") -> "PlateFormat":
        if isinstance(spec, PlateFormat):
            return spec
        if isinstance(spec, str):
            digits = re.sub(r"[^0-9]", "", spec)
            if not digits:
                raise PlateError(f"unrecognised plate format {spec!r}")
            spec = int(digits)
        if spec not in PLATE_FORMATS:
            raise PlateError(
                f"unsupported plate format {spec!r}; known: "
                + ", ".join(str(k) for k in PLATE_FORMATS)
            )
        return cls(*PLATE_FORMATS[spec])

    @classmethod
    def infer(cls, wells) -> "PlateFormat":
        """Smallest standard plate that contains every well given."""
        max_row = max_col = -1
        for well in wells:
            row, col = parse_well(well)
            max_row, max_col = max(max_row, row), max(max_col, col)
        for size, (rows, cols) in sorted(PLATE_FORMATS.items()):
            if max_row < rows and max_col < cols:
                return cls(rows, cols)
        raise PlateError(
            f"wells extend to row {row_label(max_row)} column {max_col + 1}, "
            "which is larger than a 1536-well plate"
        )

    def contains(self, well: str) -> bool:
        row, col = parse_well(well)
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    @property
    def rows(self) -> list[str]:
        return [row_label(i) for i in range(self.n_rows)]

    @property
    def columns(self) -> list[int]:
        return list(range(1, self.n_cols + 1))

    def wells(self, order: str = "row") -> list[str]:
        """Every well, in row-major (``A1, A2, ...``) or column-major order."""
        if order.startswith("col"):
            return [well_name(r, c) for c in range(self.n_cols)
                    for r in range(self.n_rows)]
        return [well_name(r, c) for r in range(self.n_rows)
                for c in range(self.n_cols)]


def expand_wells(spec: str | list[str], plate: PlateFormat | None = None) -> list[str]:
    """Expand a human well specification into canonical well names.

    Understands, comma- or whitespace-separated and in any mixture::

        A1                a single well
        A1:F12            the rectangular block from A1 to F12
        A1-A6             same, with a dash
        A                 an entire row        (needs ``plate``)
        B:D               rows B through D     (needs ``plate``)
        3                 an entire column     (needs ``plate``)
        5:8               columns 5 through 8  (needs ``plate``)
        all               every well           (needs ``plate``)
    """
    if isinstance(spec, (list, tuple, set)):
        parts: list[str] = []
        for item in spec:
            parts.extend(expand_wells(str(item), plate))
        return _dedupe(parts)

    text = str(spec).strip()
    if not text:
        return []
    out: list[str] = []
    for token in re.split(r"[,;]+|\s+", text):
        if not token:
            continue
        out.extend(_expand_token(token, plate))
    return _dedupe(out)


def _need_plate(plate: PlateFormat | None, token: str) -> PlateFormat:
    if plate is None:
        raise PlateError(
            f"{token!r} refers to whole rows/columns, so the plate format must be known"
        )
    return plate


def _expand_token(token: str, plate: PlateFormat | None) -> list[str]:
    token = token.strip()
    if token.lower() in {"all", "*", "plate"}:
        return _need_plate(plate, token).wells()

    if ":" in token or "-" in token:
        sep = ":" if ":" in token else "-"
        start, _, end = token.partition(sep)
        start, end = start.strip(), end.strip()
        if is_well(start) and is_well(end):
            r0, c0 = parse_well(start)
            r1, c1 = parse_well(end)
            return [well_name(r, c)
                    for r in range(min(r0, r1), max(r0, r1) + 1)
                    for c in range(min(c0, c1), max(c0, c1) + 1)]
        fmt = _need_plate(plate, token)
        if start.isdigit() and end.isdigit():  # column range
            c0, c1 = int(start) - 1, int(end) - 1
            return [well_name(r, c) for r in range(fmt.n_rows)
                    for c in range(min(c0, c1), max(c0, c1) + 1)]
        if start.isalpha() and end.isalpha():  # row range
            r0, r1 = row_index(start), row_index(end)
            return [well_name(r, c)
                    for r in range(min(r0, r1), max(r0, r1) + 1)
                    for c in range(fmt.n_cols)]
        raise PlateError(f"could not interpret the range {token!r}")

    if is_well(token):
        return [normalise_well(token)]
    fmt = _need_plate(plate, token)
    if token.isdigit():
        col = int(token) - 1
        if not 0 <= col < fmt.n_cols:
            raise PlateError(f"column {token} is outside a {fmt.name} plate")
        return [well_name(r, col) for r in range(fmt.n_rows)]
    if token.isalpha():
        row = row_index(token)
        if not 0 <= row < fmt.n_rows:
            raise PlateError(f"row {token} is outside a {fmt.name} plate")
        return [well_name(row, c) for c in range(fmt.n_cols)]
    raise PlateError(f"could not interpret the well specification {token!r}")


def _dedupe(wells: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for well in wells:
        if well not in seen:
            seen.add(well)
            out.append(well)
    return out


def sort_wells(wells, order: str = "row") -> list[str]:
    """Sort wells the way a plate reads, not the way strings sort."""
    if order.startswith("col"):
        key = lambda w: tuple(reversed(parse_well(w)))  # noqa: E731
    else:
        key = parse_well
    return sorted(wells, key=key)
