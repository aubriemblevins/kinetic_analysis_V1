"""What is in each well.

Every well carries up to three identities - protein, test compound and
substrate - each with its own concentration in its own unit, plus a role
(sample, blank, control, empty).  Extra factors can be added freely.

A map can be written three ways, whichever suits the person at the bench:

* **layout rules** (YAML/JSON): a handful of blocks with titration series -
  compact, readable, and what the app writes out;
* **a long table** (CSV/XLSX): one row per well - universal, and easy to build
  in Excel;
* **grids** (CSV): 8x12 blocks pasted straight out of a plate template.
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .plates import (PlateError, PlateFormat, expand_wells, is_well,
                     normalise_well, parse_well, sort_wells, well_name)
from .units import Quantity, UnitError, best_display_unit, parse_quantity

#: The three identities the tool always knows about, in display order.
CORE_FACTORS = ("protein", "compound", "substrate")

DEFAULT_LABELS = {"protein": "Protein", "compound": "Test compound",
                  "substrate": "Substrate"}

#: Roles drive how a well is used in the analysis.
ROLES = {
    "sample": "A normal reaction well.",
    "positive_control": "Uninhibited reaction (vehicle only); used as 100% activity.",
    "negative_control": "No-enzyme or fully-inhibited control.",
    "blank": "Background well; can be subtracted from samples.",
    "empty": "Nothing in the well; ignored entirely.",
}

_ROLE_ALIASES = {
    "": "sample", "sample": "sample", "unknown": "sample", "test": "sample",
    "no-inhibitor": "positive_control", "no_inhibitor": "positive_control",
    "uninhibited": "positive_control", "vehicle": "positive_control",
    "dmso": "positive_control", "positive": "positive_control",
    "positive_control": "positive_control", "poscontrol": "positive_control",
    "max": "positive_control", "100%": "positive_control",
    "negative": "negative_control", "negative_control": "negative_control",
    "no-enzyme": "negative_control", "no_enzyme": "negative_control",
    "min": "negative_control", "0%": "negative_control",
    "blank": "blank", "background": "blank", "buffer": "blank",
    "no-substrate": "blank", "no_substrate": "blank",
    "empty": "empty", "none": "empty", "unused": "empty", "skip": "empty",
}

_FACTOR_ALIASES = {
    "protein": "protein", "enzyme": "protein", "target": "protein",
    "kinase": "protein", "protease": "protein", "receptor": "protein",
    "compound": "compound", "test_compound": "compound", "testcompound": "compound",
    "inhibitor": "compound", "drug": "compound", "treatment": "compound",
    "modulator": "compound", "ligand": "compound",
    "substrate": "substrate", "peptide": "substrate", "reporter": "substrate",
}

_CONC_SUFFIXES = ("_conc", "_concentration", "conc", "concentration", "_c")


class PlateMapError(ValueError):
    """Raised when a plate map cannot be built or is inconsistent."""


@dataclass(frozen=True)
class Factor:
    """An identity plus the amount of it, in whatever unit it was written."""

    name: str | None = None
    conc: Quantity | None = None

    @property
    def is_empty(self) -> bool:
        return self.name is None and self.conc is None

    def label(self, unit: str | None = None) -> str:
        if self.is_empty:
            return "-"
        if self.conc is None:
            return self.name or "-"
        amount = self.conc.format(unit)
        return f"{self.name} {amount}" if self.name else amount

    def key(self) -> tuple:
        """Hashable identity used to match technical replicates."""
        conc = None if self.conc is None else _round_sig(self.conc.canonical, 12)
        return (self.name, conc)


def _round_sig(value: float, sig: int = 12) -> float:
    if value == 0 or not math.isfinite(value):
        return 0.0 if value == 0 else value
    return round(value, -int(math.floor(math.log10(abs(value)))) + (sig - 1))


@dataclass
class WellInfo:
    """Everything known about one well."""

    well: str
    factors: dict[str, Factor] = field(default_factory=dict)
    role: str = "sample"
    group: str | None = None
    exclude: bool = False
    notes: str = ""

    def factor(self, name: str) -> Factor:
        return self.factors.get(name, Factor())

    @property
    def is_analysed(self) -> bool:
        return self.role != "empty" and not self.exclude

    def condition_key(self, factors=None) -> tuple:
        if self.group:
            return ("group", self.group)
        names = tuple(factors) if factors else tuple(self.factors)
        return (self.role,) + tuple(self.factor(n).key() for n in names)

    def condition_label(self, factors=None, units=None, labels=None) -> str:
        if self.group:
            return self.group
        units = units or {}
        labels = labels or DEFAULT_LABELS
        names = tuple(factors) if factors else tuple(self.factors)
        parts = []
        for name in names:
            fac = self.factor(name)
            if fac.is_empty:
                continue
            parts.append(f"{labels.get(name, name.title())} {fac.label(units.get(name))}")
        if self.role != "sample":
            parts.append(f"[{self.role.replace('_', ' ')}]")
        return " | ".join(parts) if parts else self.well


@dataclass
class PlateMap:
    """The full picture of a plate: one :class:`WellInfo` per annotated well."""

    plate: PlateFormat
    wells: dict[str, WellInfo] = field(default_factory=dict)
    factor_order: list[str] = field(default_factory=lambda: list(CORE_FACTORS))
    units: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))
    name: str = ""
    notes: list[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    @classmethod
    def blank(cls, plate=96) -> "PlateMap":
        fmt = PlateFormat.of(plate)
        return cls(plate=fmt,
                   wells={w: WellInfo(w, role="empty") for w in fmt.wells()})

    def get(self, well: str) -> WellInfo:
        well = normalise_well(well)
        if well not in self.wells:
            self.wells[well] = WellInfo(well, role="empty")
        return self.wells[well]

    def set(self, well: str, info: WellInfo) -> None:
        self.wells[normalise_well(well)] = info

    @property
    def analysed_wells(self) -> list[str]:
        return sort_wells([w for w, info in self.wells.items() if info.is_analysed])

    def wells_with_role(self, role: str) -> list[str]:
        return sort_wells([w for w, i in self.wells.items() if i.role == role])

    def active_factors(self) -> list[str]:
        """Factors that are actually used somewhere on the plate."""
        used = []
        for name in self.factor_order:
            if any(not info.factor(name).is_empty for info in self.wells.values()):
                used.append(name)
        return used

    def auto_units(self) -> dict[str, str]:
        """Sensible display unit per factor, unless the user fixed one."""
        out = dict(self.units)
        for name in self.factor_order:
            if name in out:
                continue
            quantities = [i.factor(name).conc for i in self.wells.values()
                          if i.is_analysed and i.factor(name).conc is not None]
            if quantities:
                out[name] = best_display_unit(quantities)
        return out

    def label_for(self, factor: str) -> str:
        return self.labels.get(factor, factor.replace("_", " ").title())

    def levels(self, factor: str) -> list[Quantity]:
        """Distinct concentrations of one factor, low to high."""
        seen: dict[float, Quantity] = {}
        for info in self.wells.values():
            if not info.is_analysed:
                continue
            conc = info.factor(factor).conc
            if conc is not None:
                seen.setdefault(_round_sig(conc.canonical, 9), conc)
        return [seen[k] for k in sorted(seen)]

    def replicate_groups(self) -> dict[tuple, list[str]]:
        """Map each condition to the wells that share it (technical replicates)."""
        factors = self.active_factors()
        groups: dict[tuple, list[str]] = {}
        for well in self.analysed_wells:
            key = self.wells[well].condition_key(factors)
            groups.setdefault(key, []).append(well)
        return groups

    def validate(self) -> list[str]:
        """Human-readable warnings; an empty list means the map looks sane."""
        problems = []
        for well, info in self.wells.items():
            if not self.plate.contains(well):
                problems.append(f"{well} is outside a {self.plate.name} plate.")
            if info.role not in ROLES:
                problems.append(f"{well} has unknown role {info.role!r}.")
        if not self.analysed_wells:
            problems.append("No wells are marked for analysis.")
        singletons = [wells[0] for wells in self.replicate_groups().values()
                      if len(wells) == 1]
        if singletons:
            problems.append(
                f"{len(singletons)} condition(s) have no technical replicate "
                f"(e.g. {', '.join(sort_wells(singletons)[:6])})."
            )
        return problems

    # -- serialisation -----------------------------------------------------
    def to_frame(self, units: dict[str, str] | None = None) -> pd.DataFrame:
        """The resolved map as a long table - one row per well."""
        units = units or self.auto_units()
        rows = []
        for well in sort_wells(self.wells):
            info = self.wells[well]
            row, col = parse_well(well)
            record = {"well": well, "row": well[:-len(str(col + 1))],
                      "column": col + 1}
            for name in self.factor_order:
                fac = info.factor(name)
                unit = units.get(name, "")
                record[name] = fac.name
                record[f"{name}_conc"] = (None if fac.conc is None
                                          else fac.conc.to(unit) if unit else fac.conc.value)
                record[f"{name}_unit"] = unit if fac.conc is not None else None
            record.update(role=info.role, exclude=info.exclude,
                          group=info.group, notes=info.notes)
            rows.append(record)
        return pd.DataFrame(rows)

    def to_csv(self, path) -> Path:
        path = Path(path)
        self.to_frame().to_csv(path, index=False)
        return path

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for info in self.wells.values():
            counts[info.role] = counts.get(info.role, 0) + 1
        parts = [f"{n} {role.replace('_', ' ')}" for role, n in sorted(counts.items())]
        groups = self.replicate_groups()
        return (f"{self.plate.name} plate: " + ", ".join(parts)
                + f"; {len(groups)} distinct condition(s)")


# --- long-table maps ------------------------------------------------------

def _canonical_column(name: str) -> tuple[str, str, str | None] | None:
    """Classify a plate-map column: ``('substrate', 'conc', 'uM')`` etc."""
    text = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    text = text.replace("[", "").replace("]", "")
    if text in {"well", "wells", "well_id", "position"}:
        return ("well", "well", None)
    # Where the well sits, not what is in it - ignore, or every well becomes
    # its own condition and nothing is ever a replicate of anything.
    if text in {"row", "column", "col", "plate_row", "plate_column",
                "plate", "plate_id", "index"}:
        return None
    if text in {"role", "well_type", "type", "sample_type"}:
        return ("role", "role", None)
    if text in {"group", "condition", "replicate_group"}:
        return ("group", "group", None)
    if text in {"exclude", "excluded", "ignore", "skip", "omit"}:
        return ("exclude", "exclude", None)
    if text in {"notes", "note", "comment", "comments"}:
        return ("notes", "notes", None)

    # A separate "<factor>_unit" column supplies the unit for "<factor>_conc",
    # which is how this tool writes a resolved map back out.
    for suffix in ("_unit", "_units", "_uom"):
        if text.endswith(suffix) and len(text) > len(suffix):
            stem = text[: -len(suffix)]
            return (_FACTOR_ALIASES.get(stem, stem), "unit", None)

    unit = None
    match = re.search(r"[_(]([a-zA-Zµμ%/]+)\)?$", text)
    if match:
        candidate = match.group(1)
        try:
            from .units import normalise_unit
            # Keep the canonical spelling: the header was lower-cased for
            # matching, and "um" should still display as "uM".
            unit = normalise_unit(candidate)[1]
            text = text[: match.start()].rstrip("_(")
        except UnitError:
            unit = None

    kind = "name"
    for suffix in _CONC_SUFFIXES:
        if text.endswith(suffix):
            kind, text = "conc", text[: -len(suffix)].rstrip("_")
            break
    if text.startswith("conc_"):
        kind, text = "conc", text[5:]

    base = _FACTOR_ALIASES.get(text, text if text else None)
    if base is None:
        return None
    if base.endswith("_name"):
        base = base[:-5]
    return (base, kind, unit)


def read_long_map(path_or_frame, plate=None) -> PlateMap:
    """Read a one-row-per-well plate map (CSV/TSV/XLSX or a DataFrame)."""
    if isinstance(path_or_frame, pd.DataFrame):
        frame = path_or_frame.copy()
    else:
        name = str(getattr(path_or_frame, "name", path_or_frame))
        if Path(name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            frame = pd.read_excel(path_or_frame, dtype=object)
        else:
            sep = "\t" if Path(name).suffix.lower() in {".tsv", ".txt"} else None
            frame = pd.read_csv(path_or_frame, dtype=object, sep=sep,
                                engine="python", encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]

    mapping: dict[str, tuple[str, str, str | None]] = {}
    for column in frame.columns:
        parsed = _canonical_column(column)
        if parsed:
            mapping[column] = parsed
    well_columns = [c for c, m in mapping.items() if m[0] == "well"]
    if not well_columns:
        # Tolerate an unnamed first column holding the well ids.
        first = frame.columns[0]
        if frame[first].map(lambda v: is_well(str(v))).all():
            mapping[first] = ("well", "well", None)
            well_columns = [first]
        else:
            raise PlateMapError(
                "the plate map needs a 'well' column (A1, A2, ...); "
                f"found columns: {', '.join(map(str, frame.columns))}"
            )

    factor_names: list[str] = []
    for _, (base, kind, _u) in mapping.items():
        if kind in {"name", "conc", "unit"} and base not in factor_names:
            factor_names.append(base)
    order = [f for f in CORE_FACTORS if f in factor_names]
    order += [f for f in factor_names if f not in order]

    wells: dict[str, WellInfo] = {}
    units: dict[str, str] = {}
    unit_columns = {base: column for column, (base, kind, _u) in mapping.items()
                    if kind == "unit"}
    for _, row in frame.iterrows():
        well_text = str(row[well_columns[0]]).strip()
        if not well_text or well_text.lower() == "nan":
            continue
        well = normalise_well(well_text)
        info = WellInfo(well)
        factors: dict[str, tuple[str | None, Quantity | None]] = {}
        for column, (base, kind, unit) in mapping.items():
            value = row.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = None
            text = None if value is None else str(value).strip()
            if kind == "well":
                continue
            if kind == "role":
                info.role = normalise_role(text)
            elif kind == "group":
                info.group = text or None
            elif kind == "exclude":
                info.exclude = _truthy(text)
            elif kind == "notes":
                info.notes = text or ""
            elif kind == "name":
                name, conc = factors.get(base, (None, None))
                factors[base] = ((text or None), conc)
            elif kind == "unit":
                continue  # consumed by the matching concentration column
            elif kind == "conc":
                row_unit = unit
                if row_unit is None and base in unit_columns:
                    stated = row.get(unit_columns[base])
                    if stated is not None and str(stated).strip().lower() not in {
                            "", "nan", "none"}:
                        row_unit = str(stated).strip()
                try:
                    quantity = parse_quantity(text, default_unit=row_unit)
                except UnitError as exc:
                    raise PlateMapError(
                        f"{well}: could not read {column!r} value {text!r} ({exc}). "
                        "Write the unit with the number (e.g. '50 uM') or in the "
                        "column header (e.g. 'substrate_conc_uM')."
                    ) from exc
                name, _ = factors.get(base, (None, None))
                factors[base] = (name, quantity)
                if quantity is not None and base not in units and row_unit:
                    units[base] = quantity.unit
        info.factors = {b: Factor(n, c) for b, (n, c) in factors.items()
                        if not (n is None and c is None)}
        if info.role == "sample" and not info.factors and not info.group:
            info.role = "empty"
        wells[well] = info

    fmt = PlateFormat.of(plate) if plate else PlateFormat.infer(wells)
    return PlateMap(plate=fmt, wells=wells, factor_order=order, units=units,
                    name=str(getattr(path_or_frame, "name", "") or ""))


def normalise_role(text: str | None) -> str:
    key = (text or "").strip().lower().replace(" ", "_")
    if key in ROLES:
        return key
    return _ROLE_ALIASES.get(key, "sample" if key else "sample")


def _truthy(text: str | None) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "x", "exclude"}


# --- layout-rule maps -----------------------------------------------------

def _series_values(spec: dict, n_levels: int, what: str) -> list[Quantity | None]:
    """Turn a titration description into one concentration per level."""
    if "values" in spec:
        raw = spec["values"]
        if isinstance(raw, str):
            raw = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
        values = [parse_quantity(v, default_unit=spec.get("unit")) for v in raw]
        if len(values) != n_levels:
            raise PlateMapError(
                f"{what}: {len(values)} values were given but the block has "
                f"{n_levels} level(s).  Adjust 'values', 'replicates' or the well range."
            )
        return values

    top = spec.get("top", spec.get("start", spec.get("high")))
    bottom = spec.get("bottom", spec.get("low"))
    unit = spec.get("unit")
    factor = spec.get("dilution", spec.get("fold"))
    if top is None and bottom is None:
        raise PlateMapError(
            f"{what}: a series needs 'values', or 'top' with 'dilution', "
            "or 'top' and 'bottom'."
        )
    if factor is not None:
        factor = float(factor)
        if factor <= 0:
            raise PlateMapError(f"{what}: 'dilution' must be greater than 0.")
        if top is not None:
            first = parse_quantity(top, default_unit=unit)
            values = [Quantity(first.value / factor ** i, first.unit, first.family)
                      for i in range(n_levels)]
        else:
            last = parse_quantity(bottom, default_unit=unit)
            values = [Quantity(last.value * factor ** (n_levels - 1 - i),
                               last.unit, last.family) for i in range(n_levels)]
    elif top is not None and bottom is not None:
        first = parse_quantity(top, default_unit=unit)
        last = parse_quantity(bottom, default_unit=unit)
        if spec.get("spacing", "log") == "linear" or last.canonical == 0:
            step = (last.canonical - first.canonical) / max(n_levels - 1, 1)
            values = [Quantity((first.canonical + step * i)
                               / (first.canonical / first.value if first.value else 1),
                               first.unit, first.family) for i in range(n_levels)]
        else:
            ratio = (last.canonical / first.canonical) ** (1 / max(n_levels - 1, 1))
            values = [Quantity(first.value * ratio ** i, first.unit, first.family)
                      for i in range(n_levels)]
    else:
        raise PlateMapError(
            f"{what}: give 'dilution' alongside 'top', or give both 'top' and 'bottom'."
        )

    if spec.get("include_zero"):
        values[-1] = Quantity(0.0, values[-1].unit, values[-1].family)
    return values


def _positions(wells: list[str], across: str) -> tuple[list[int], dict[str, int]]:
    """Distinct row (or column) indices in a block, and each well's position."""
    axis = 0 if str(across).lower().startswith("row") else 1
    per_well = {w: parse_well(w)[axis] for w in wells}
    ordered = sorted(set(per_well.values()))
    return ordered, per_well


def _apply_series(spec: dict, wells: list[str], what: str) -> dict[str, Quantity | None]:
    across = spec.get("across", spec.get("axis", "columns"))
    reps = int(spec.get("replicates", spec.get("pairs", 1)))
    if reps < 1:
        raise PlateMapError(f"{what}: 'replicates' must be at least 1.")
    ordered, per_well = _positions(wells, across)
    if spec.get("reverse") or str(spec.get("direction", "")).lower() in {
            "increasing", "up", "ascending"}:
        ordered = list(reversed(ordered))
    n_levels = math.ceil(len(ordered) / reps)
    values = _series_values(spec, n_levels, what)
    level_of = {pos: values[i // reps] for i, pos in enumerate(ordered)}
    return {w: level_of[per_well[w]] for w in wells}


def _factor_from_spec(spec, wells: list[str], what: str) -> dict[str, Factor]:
    """Build a per-well :class:`Factor` from one entry in a layout block."""
    if spec is None:
        return {w: Factor() for w in wells}
    if isinstance(spec, str):
        try:
            quantity = parse_quantity(spec)
            return {w: Factor(None, quantity) for w in wells}
        except UnitError:
            return {w: Factor(spec, None) for w in wells}
    if not isinstance(spec, dict):
        raise PlateMapError(f"{what}: expected a name or a mapping, got {spec!r}")

    name = spec.get("name", spec.get("id", spec.get("identity")))
    if "series" in spec or "titration" in spec:
        series = spec.get("series", spec.get("titration"))
        if "unit" not in series and "unit" in spec:
            series = {**series, "unit": spec["unit"]}
        concs = _apply_series(series, wells, what)
        return {w: Factor(name, concs[w]) for w in wells}
    conc_text = spec.get("conc", spec.get("concentration", spec.get("value")))
    quantity = parse_quantity(conc_text, default_unit=spec.get("unit"))
    return {w: Factor(name, quantity) for w in wells}


def map_from_layout(layout: dict) -> PlateMap:
    """Build a :class:`PlateMap` from layout rules (the YAML/JSON form)."""
    if not isinstance(layout, dict):
        raise PlateMapError("a layout must be a mapping with a 'blocks' list")
    fmt = PlateFormat.of(layout.get("plate", layout.get("format", 96)))
    labels = {**DEFAULT_LABELS, **(layout.get("labels") or {})}
    units = dict(layout.get("units") or {})
    defaults = layout.get("defaults") or {}

    plate_map = PlateMap(plate=fmt, wells={}, units=units, labels=labels,
                         name=str(layout.get("name", "")))
    for well in fmt.wells():
        plate_map.wells[well] = WellInfo(well, role="empty")

    factor_order: list[str] = [f for f in CORE_FACTORS]
    blocks = layout.get("blocks") or layout.get("regions") or []
    if not blocks:
        raise PlateMapError("the layout has no 'blocks' - nothing to map")

    for index, block in enumerate(blocks):
        what = f"block {index + 1}" + (f" ({block['name']})" if block.get("name") else "")
        wells_spec = block.get("wells", block.get("range"))
        if not wells_spec:
            raise PlateMapError(f"{what}: no 'wells' given")
        try:
            wells = expand_wells(wells_spec, fmt)
        except PlateError as exc:
            raise PlateMapError(f"{what}: {exc}") from exc
        outside = [w for w in wells if not fmt.contains(w)]
        if outside:
            raise PlateMapError(
                f"{what}: {', '.join(outside[:5])} fall outside a {fmt.name} plate"
            )

        role = normalise_role(block.get("role"))
        if block.get("role") is None:
            role = "empty" if _block_is_empty(block, defaults) else "sample"

        merged = {**{k: v for k, v in defaults.items() if k not in _BLOCK_KEYS},
                  **{k: v for k, v in block.items() if k not in _BLOCK_KEYS}}
        if role == "empty":
            merged = {}  # an empty well holds nothing, defaults included
        per_well: dict[str, dict[str, Factor]] = {w: {} for w in wells}
        for factor_name, spec in merged.items():
            canonical = _FACTOR_ALIASES.get(factor_name.lower(), factor_name.lower())
            if canonical not in factor_order:
                factor_order.append(canonical)
            built = _factor_from_spec(spec, wells, f"{what} / {factor_name}")
            for well in wells:
                if not built[well].is_empty:
                    per_well[well][canonical] = built[well]

        for well in wells:
            plate_map.wells[well] = WellInfo(
                well=well, factors=per_well[well], role=role,
                group=block.get("group"), exclude=bool(block.get("exclude", False)),
                notes=str(block.get("notes", block.get("name", "")) or ""),
            )

    plate_map.factor_order = factor_order
    return plate_map


_BLOCK_KEYS = {"wells", "range", "role", "name", "group", "exclude", "notes"}


def _block_is_empty(block: dict, defaults: dict) -> bool:
    keys = (set(block) | set(defaults)) - _BLOCK_KEYS
    return not keys


def read_layout(path) -> dict:
    """Load layout rules from YAML or JSON."""
    text = Path(path).read_text(encoding="utf-8-sig") if not hasattr(path, "read") \
        else path.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
        raise PlateMapError("PyYAML is required to read YAML layouts") from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise PlateMapError("the layout file did not contain a mapping")
    return loaded


def _suffix_of(path) -> str:
    """The file extension, whether given a path, a Path or an upload."""
    if hasattr(path, "read"):
        return Path(str(getattr(path, "name", ""))).suffix.lower()
    return Path(str(path)).suffix.lower()


def read_map(path, plate=None) -> PlateMap:
    """Read a plate map from any supported file type."""
    suffix = _suffix_of(path)
    if suffix in {".yaml", ".yml", ".json"}:
        return map_from_layout(read_layout(path))
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return read_long_map(path, plate)
    if suffix in {".csv", ".tsv", ".txt"}:
        if hasattr(path, "read"):
            raw = path.read()
        else:
            raw = Path(str(path)).read_text(encoding="utf-8-sig")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        # A leading "# field" line means plate-shaped grids rather than a table.
        if re.search(r"^\s*#\s*\w+", raw, re.MULTILINE):
            return read_grid_map(raw, plate)
        buffer = io.StringIO(raw)
        buffer.name = str(getattr(path, "name", path))
        return read_long_map(buffer, plate)
    raise PlateMapError(
        f"unsupported plate-map file type: {suffix or path!r}. "
        "Use .yaml/.json for layout rules, or .csv/.xlsx for a table.")


# --- grid maps ------------------------------------------------------------

def read_grid_map(text_or_path, plate=None) -> PlateMap:
    """Read plate-shaped grids, one per field.

    Each grid is introduced by a ``# field`` line, then a header row of column
    numbers and one row per plate row::

        # substrate_conc_uM
        ,1,2,3
        A,50,50,25
        B,25,25,12.5
    """
    if hasattr(text_or_path, "read"):
        text = text_or_path.read()
    elif "\n" in str(text_or_path):
        text = str(text_or_path)
    else:
        text = Path(text_or_path).read_text(encoding="utf-8-sig")
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")

    sections: dict[str, list[list[str]]] = {}
    current: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            current = line.lstrip("# \t").strip()
            sections[current] = []
            continue
        if current is None:
            continue
        sections[current].append([c.strip() for c in line.split(",")])
    if not sections:
        raise PlateMapError(
            "no '# field' sections found; a grid map needs a '# substrate_conc' "
            "style header above each grid"
        )

    per_well: dict[str, dict] = {}
    units: dict[str, str] = {}
    factor_order: list[str] = []
    for title, rows in sections.items():
        parsed = _canonical_column(title)
        if parsed is None:
            raise PlateMapError(f"unrecognised grid field {title!r}")
        base, kind, unit = parsed
        if kind in {"name", "conc"} and base not in factor_order:
            factor_order.append(base)
        if not rows:
            continue
        header, *body = rows
        columns = []
        for cell in header[1:]:
            digits = re.sub(r"[^0-9]", "", cell)
            columns.append(int(digits) if digits else None)
        for row_cells in body:
            if not row_cells or not row_cells[0]:
                continue
            row_label_text = row_cells[0].strip()
            for position, cell in enumerate(row_cells[1:]):
                if position >= len(columns) or columns[position] is None:
                    continue
                if not cell:
                    continue
                well = normalise_well(f"{row_label_text}{columns[position]}")
                record = per_well.setdefault(well, {})
                if kind == "conc":
                    record[(base, "conc")] = parse_quantity(cell, default_unit=unit)
                    if unit:
                        units.setdefault(base, unit)
                elif kind == "name":
                    record[(base, "name")] = cell
                else:
                    record[kind] = cell

    wells: dict[str, WellInfo] = {}
    for well, record in per_well.items():
        factors: dict[str, Factor] = {}
        for (base, kind), value in [(k, v) for k, v in record.items()
                                    if isinstance(k, tuple)]:
            existing = factors.get(base, Factor())
            factors[base] = (replace(existing, name=value) if kind == "name"
                             else replace(existing, conc=value))
        info = WellInfo(well, factors=factors,
                        role=normalise_role(record.get("role")),
                        group=record.get("group"),
                        exclude=_truthy(record.get("exclude")),
                        notes=record.get("notes", "") or "")
        if not factors and record.get("role") is None:
            info.role = "empty"
        wells[well] = info

    fmt = PlateFormat.of(plate) if plate else PlateFormat.infer(wells)
    order = [f for f in CORE_FACTORS if f in factor_order]
    order += [f for f in factor_order if f not in order]
    return PlateMap(plate=fmt, wells=wells, factor_order=order, units=units)


def layout_from_map(plate_map: PlateMap) -> dict:
    """Round-trip a resolved map back to a (well-by-well) layout dict."""
    blocks = []
    for well in sort_wells(plate_map.wells):
        info = plate_map.wells[well]
        if info.role == "empty" and not info.factors:
            continue
        block = {"wells": well, "role": info.role}
        for name, fac in info.factors.items():
            entry = {}
            if fac.name:
                entry["name"] = fac.name
            if fac.conc is not None:
                entry["conc"] = fac.conc.format()
            if entry:
                block[name] = entry
        if info.notes:
            block["notes"] = info.notes
        blocks.append(block)
    return {"plate": plate_map.plate.size, "labels": plate_map.labels,
            "units": plate_map.auto_units(), "blocks": blocks}
