"""Writing results out, in the shapes Prism and Excel want.

The aim is that nothing needs rearranging after this: open the file, select
all, paste into the matching Prism table, fit.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import AnalysisResult
from .units import _sig

#: Prism table shapes this module can produce.
PRISM_VALUES = {
    "rate": "Fitted rate per well",
    "percent_activity": "Rate as a percentage of the matching no-inhibitor control",
    "percent_inhibition": "100 - % activity",
}


def prism_table(
    result: AnalysisResult,
    x_factor: str,
    series_factor: str | None = None,
    value: str = "rate",
    replicates: bool = False,
    log_x: bool = False,
) -> pd.DataFrame:
    """An XY table: ``x_factor`` down the rows, ``series_factor`` across.

    With ``replicates=True`` each dataset spreads into side-by-side subcolumns,
    which is what a Prism XY table with replicate subcolumns expects.
    """
    if value not in PRISM_VALUES:
        raise ValueError(f"value must be one of {', '.join(PRISM_VALUES)}")
    units = result.units
    x_unit = units.get(x_factor, "")
    series_unit = units.get(series_factor, "") if series_factor else ""

    usable = [c for c in result.conditions
              if c.role in {"sample", "positive_control"}
              and c.factors.get(x_factor) is not None
              and c.factors[x_factor].conc is not None]
    if not usable:
        raise ValueError(f"no conditions carry a {x_factor} concentration")

    x_values = sorted({c.factors[x_factor].conc.canonical for c in usable})
    if log_x:
        x_values = [v for v in x_values if v > 0]
        if not x_values:
            raise ValueError("a log X axis needs at least one non-zero concentration")

    series_keys = sorted(
        {(c.factors[series_factor].conc.canonical
          if series_factor and c.factors.get(series_factor)
          and c.factors[series_factor].conc is not None else None)
         for c in usable},
        key=lambda k: (k is None, k))

    lookup: dict[tuple, list] = {}
    for condition in usable:
        x_key = condition.factors[x_factor].conc.canonical
        series_key = None
        if series_factor and condition.factors.get(series_factor) is not None \
                and condition.factors[series_factor].conc is not None:
            series_key = condition.factors[series_factor].conc.canonical
        lookup[(x_key, series_key)] = condition

    width = max((len(c.rates) for c in usable), default=1) if replicates else 1
    x_label = f"{result.plate_map.label_for(x_factor)} ({x_unit})"
    if log_x:
        x_label = f"log10 [{result.plate_map.label_for(x_factor)} ({x_unit})]"

    columns: dict[str, list] = {}
    first = next(iter(usable))
    scale = (first.factors[x_factor].conc.to(x_unit)
             / first.factors[x_factor].conc.canonical) if x_unit else 1.0
    columns[x_label] = [math.log10(v * scale) if log_x else v * scale
                        for v in x_values]

    for series_key in series_keys:
        name = _series_name(result, series_factor, series_key, series_unit)
        if replicates:
            for index in range(width):
                header = f"{name}" if index == 0 else f"{name} ({index + 1})"
                columns[header] = [
                    _replicate(result, lookup.get((x, series_key)), index, value)
                    for x in x_values]
        else:
            columns[name] = [_aggregate(lookup.get((x, series_key)), value)
                             for x in x_values]
    return pd.DataFrame(columns)


def _series_name(result, series_factor, key, unit) -> str:
    if series_factor is None:
        return "Y"
    label = result.plate_map.label_for(series_factor)
    if key is None:
        return f"{label} (none)"
    from .units import Quantity, normalise_unit
    family, _canonical = normalise_unit(unit) if unit else ("molar", "M")
    quantity = Quantity(key, {"molar": "M", "mass_conc": "g/L", "activity": "U/mL",
                              "volume_fraction": "%", "ratio": "x"}[family], family)
    return f"{_sig(quantity.to(unit), 3)} {unit}" if unit else _sig(key, 3)


def _aggregate(condition, value):
    if condition is None:
        return None
    return {"rate": condition.mean,
            "percent_activity": condition.percent_activity,
            "percent_inhibition": condition.percent_inhibition}[value]


def _replicate(result, condition, index, value):
    """One replicate of a condition - the raw rate, or its normalised form."""
    if condition is None or index >= len(condition.rates):
        return None
    if value == "rate":
        return condition.rates[index]
    if index >= len(condition.wells):
        return None
    outcome = result.wells.get(condition.wells[index])
    if outcome is None:
        return None
    return (outcome.percent_activity if value == "percent_activity"
            else outcome.percent_inhibition)


def prism_tables(result: AnalysisResult) -> dict[str, pd.DataFrame]:
    """Every sensible Prism layout for this plate, keyed by file name."""
    factors = [f for f in result.factors() if len(result.plate_map.levels(f)) > 1]
    tables: dict[str, pd.DataFrame] = {}
    if not factors:
        return tables

    pairs: list[tuple[str, str | None]] = []
    if len(factors) == 1:
        pairs.append((factors[0], None))
    else:
        for x_factor in factors:
            for series_factor in factors:
                if x_factor != series_factor:
                    pairs.append((x_factor, series_factor))

    for x_factor, series_factor in pairs:
        stem = f"prism_{x_factor}" + (f"_by_{series_factor}" if series_factor else "")
        try:
            tables[f"{stem}_mean.csv"] = prism_table(
                result, x_factor, series_factor, "rate")
            tables[f"{stem}_replicates.csv"] = prism_table(
                result, x_factor, series_factor, "rate", replicates=True)
        except ValueError:
            continue
        # A compound on the X axis is a dose-response, which is fitted on a log
        # axis against normalised activity.
        if x_factor == "compound":
            try:
                tables[f"{stem}_percent_activity.csv"] = prism_table(
                    result, x_factor, series_factor, "percent_activity")
                tables[f"{stem}_log_percent_activity.csv"] = prism_table(
                    result, x_factor, series_factor, "percent_activity", log_x=True)
            except ValueError:
                pass
    return tables


# --- the whole output folder ---------------------------------------------

def write_outputs(
    result: AnalysisResult,
    out_dir,
    figures: bool = True,
    theme: str = "light",
    plate_scale: str = "free",
    excel: bool = True,
) -> list[Path]:
    """Write tables, Prism layouts, figures and a record of the settings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write_csv(frame: pd.DataFrame, name: str) -> Path:
        path = out_dir / name
        frame.to_csv(path, index=False)
        written.append(path)
        return path

    wells = result.well_table()
    conditions = result.condition_table()
    write_csv(wells, "well_results.csv")
    write_csv(conditions, "condition_means.csv")
    write_csv(result.plate_map.to_frame(result.units), "plate_map_resolved.csv")
    if result.window_plan is not None and result.window_plan.groups:
        write_csv(pd.DataFrame(result.window_plan.to_rows()),
                  "linear_range_windows.csv")

    prism_dir = out_dir / "prism"
    prism_dir.mkdir(exist_ok=True)
    for name, table in prism_tables(result).items():
        path = prism_dir / name
        table.to_csv(path, index=False)
        written.append(path)

    written.append(_write_report(result, out_dir / "analysis_report.txt"))

    if excel:
        try:
            path = out_dir / "kinetics_results.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                wells.to_excel(writer, sheet_name="wells", index=False)
                conditions.to_excel(writer, sheet_name="conditions", index=False)
                result.plate_map.to_frame(result.units).to_excel(
                    writer, sheet_name="plate map", index=False)
                for name, table in prism_tables(result).items():
                    sheet = name.replace("prism_", "").replace(".csv", "")[:31]
                    table.to_excel(writer, sheet_name=sheet, index=False)
            written.append(path)
        except Exception as exc:  # pragma: no cover - openpyxl is optional
            print(f"  (Excel workbook skipped: {exc})")

    if figures:
        from . import plotting
        figure_dir = out_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        written.append(plotting.plot_plate_curves(
            result, figure_dir / "plate_curves.png", scale=plate_scale,
            theme=theme))
        try:
            written.append(plotting.plot_window_choice(
                result, figure_dir / "window_choice.png", theme=theme))
        except ValueError:
            pass  # per-well windows, or no reference well with enough signal
        try:
            written.append(plotting.plot_rate_heatmap(
                result, figure_dir / "rate_heatmap.png", theme=theme))
        except ValueError:
            pass
        factors = [f for f in result.factors()
                   if len(result.plate_map.levels(f)) > 1]
        for x_factor in factors:
            series = next((f for f in factors if f != x_factor), None)
            for value in (["rate", "percent_activity"]
                          if x_factor == "compound" else ["rate"]):
                try:
                    written.append(plotting.plot_rate_vs_concentration(
                        result, x_factor, series, value=value,
                        path=figure_dir / f"{value}_vs_{x_factor}.png",
                        theme=theme))
                except ValueError:
                    continue
    return written


def _write_report(result: AnalysisResult, path: Path) -> Path:
    """A plain-text record of what was done - the provenance of the numbers."""
    settings = asdict(result.settings)
    lines = [
        "Enzyme kinetics analysis",
        "=" * 24,
        "",
        f"Data file      : {result.data.source}",
        f"Plate          : {result.plate_map.plate.name}",
        f"Readings       : {result.data.n_timepoints} timepoints over "
        f"{result.data.duration / 60:.1f} min",
        f"Rate units     : {result.rate_label}",
        "",
        result.plate_map.summary(),
        result.summary(),
        "",
        "Linear-range detection",
        "-" * 22,
    ]
    described = {
        "auto": ("longest straight window anchored at the start of the reaction; "
                 "straightness judged against each well's own noise"),
        "max_slope": f"steepest window of {settings['window_points']} readings",
        "fixed": (f"fixed window "
                  f"{_fmt(settings['fixed_start'])}-{_fmt(settings['fixed_end'])} s"),
        "initial": f"first {settings['window_points']} readings",
        "all": "the whole curve, no window selection",
    }[settings["method"]]
    lines.append(f"Method         : {settings['method']} - {described}")
    plan = result.window_plan
    if plan is not None:
        lines.append(f"Window sharing : {plan.summary()}")
        if plan.match_on:
            lines.append("Matched on     : "
                         + ", ".join(result.plate_map.label_for(f)
                                     for f in plan.match_on))
        for note in plan.notes:
            lines.append(f"                 {note}")
    lines.append(f"Scatter allowed: {settings['residual_tolerance']:g} x the "
                 f"well's noise (at {settings['confidence']:.0%} confidence)")
    lines.append("Minimum window : "
                 + (f"{settings['min_points']} readings"
                    if settings["min_points"] else "chosen automatically"))
    lines.append(f"Rate floor     : a window must beat "
                 f"{settings['noise_sigma_threshold']:g} x noise to count")
    if settings["search_start"] or settings["search_end"]:
        lines.append(f"Search limited to "
                     f"{_fmt(settings['search_start'])}-"
                     f"{_fmt(settings['search_end'])} s")
    lines.append(f"Background     : {result.blank_summary or result.blank_mode}")

    if plan is not None and plan.groups:
        lines += ["", "The window each matched group is fitted over", "-" * 43,
                  "  (reference wells decide it; see figures/window_choice.png)"]
        for row in plan.to_rows():
            lines.append(f"  {row['applied window (min)']:>16s} min  "
                         f"{row['readings used']:>4d} readings  "
                         f"[refs {row['reference wells'] or 'none'}]  "
                         f"{row['group']}")
            if row["notes"]:
                lines.append(f"      note: {row['notes']}")

    flagged = result.flagged_wells()
    lines += ["", "Wells to look at", "-" * 16]
    if flagged:
        for well in flagged:
            outcome = result.wells[well]
            note = ", ".join(outcome.concerns) or outcome.error
            lines.append(f"  {well:5s} {note}")
    else:
        lines.append("  none - every well fitted cleanly.")

    noisy = [c for c in result.conditions
             if c.n > 1 and math.isfinite(c.cv) and c.cv > 20 and c.role == "sample"]
    if noisy:
        lines += ["", "Conditions whose replicates disagree by more than 20%",
                  "-" * 52]
        for condition in sorted(noisy, key=lambda c: -c.cv):
            lines.append(f"  CV {condition.cv:5.1f}%  {condition.label} "
                         f"({', '.join(condition.wells)})")

    if result.warnings:
        lines += ["", "Notes", "-" * 5] + [f"  {w}" for w in result.warnings]

    lines += ["", "Flag meanings", "-" * 13]
    from .linearity import FLAG_DESCRIPTIONS
    seen = {f for outcome in result.wells.values() for f in outcome.flags}
    for flag in sorted(seen):
        lines.append(f"  {flag:18s} {FLAG_DESCRIPTIONS.get(flag, '')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value) -> str:
    return "start" if value is None else f"{value:g}"
