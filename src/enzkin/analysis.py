"""Turning a plate of progress curves into rates, and rates into conditions.

The pipeline is: fit every well's linear range, subtract background if there
are blanks, average technical replicates, and express each sample against its
matching no-inhibitor control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .linearity import (INFORMATIONAL_FLAGS, DetectionSettings, LinearFit,
                        LinearityError, detect_linear_range)
from .platemap import Factor, PlateMap, WellInfo
from .plates import sort_wells
from .readers import KineticData
from .units import time_factor
from .windows import WindowPlan, WindowPolicy, fixed_settings, plan_windows

#: How background wells are used.
BLANK_MODES = {
    "none": "Ignore blank wells.",
    "slope": "Subtract the mean blank rate from every fitted rate.",
    "trace": "Subtract the blank readings point by point before fitting.",
}


@dataclass
class WellResult:
    """One well: what was in it, the fit, and the rate that came out."""

    well: str
    info: WellInfo
    fit: LinearFit | None
    rate: float                   # signal units per `rate_unit`, blank-corrected
    raw_rate: float               # before blank correction
    rate_se: float
    blank_rate: float = 0.0
    percent_activity: float | None = None
    percent_inhibition: float | None = None
    error: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def role(self) -> str:
        return self.info.role

    @property
    def usable(self) -> bool:
        return self.fit is not None and not self.error

    @property
    def concerns(self) -> list[str]:
        """Flags worth a look, as opposed to ones that just describe the fit."""
        return [f for f in self.flags if f not in INFORMATIONAL_FLAGS]


@dataclass
class ConditionResult:
    """One condition and the technical replicates that measured it."""

    key: tuple
    label: str
    wells: list[str]
    rates: list[float]
    role: str
    factors: dict[str, Factor]
    mean: float
    sd: float
    sem: float
    cv: float
    n: int
    percent_activity: float | None = None
    percent_inhibition: float | None = None
    flags: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    sort_key: tuple = ()

    @property
    def concerns(self) -> list[str]:
        return [f for f in self.flags if f not in INFORMATIONAL_FLAGS]


@dataclass
class AnalysisResult:
    """Everything the tool worked out about one plate."""

    data: KineticData
    plate_map: PlateMap
    settings: DetectionSettings
    wells: dict[str, WellResult]
    conditions: list[ConditionResult]
    rate_unit: str = "min"
    signal_label: str = "RFU"
    blank_mode: str = "slope"
    blank_summary: str = ""
    units: dict[str, str] = field(default_factory=dict)
    z_prime: float | None = None
    warnings: list[str] = field(default_factory=list)
    window_plan: WindowPlan | None = None

    @property
    def rate_label(self) -> str:
        return f"{self.signal_label}/{self.rate_unit}"

    def factors(self) -> list[str]:
        return self.plate_map.active_factors()

    # -- tables ------------------------------------------------------------
    def well_table(self) -> pd.DataFrame:
        """One row per well: conditions, rate, fit window and QC flags."""
        rows = []
        for well in sort_wells(self.wells):
            result = self.wells[well]
            fit = result.fit
            row: dict = {"well": well, "role": result.role}
            for name in self.factors():
                factor = result.info.factor(name)
                unit = self.units.get(name, "")
                row[self.plate_map.label_for(name)] = factor.name
                row[f"{self.plate_map.label_for(name)} ({unit})"] = (
                    None if factor.conc is None else factor.conc.to(unit) if unit
                    else factor.conc.value)
            row[f"rate ({self.rate_label})"] = result.rate
            row[f"rate SE ({self.rate_label})"] = result.rate_se
            if self.blank_mode != "none":
                row[f"rate before blank ({self.rate_label})"] = result.raw_rate
            row["% activity"] = result.percent_activity
            row["% inhibition"] = result.percent_inhibition
            row["R2"] = None if fit is None else fit.r2
            row["fit start (min)"] = None if fit is None else fit.start_time / 60
            row["fit end (min)"] = None if fit is None else fit.end_time / 60
            row["fit points"] = None if fit is None else fit.n_points
            row["SNR"] = None if fit is None else (
                None if not math.isfinite(fit.snr) else round(fit.snr, 1))
            row["flags"] = ", ".join(result.flags)
            row["note"] = result.error
            rows.append(row)
        return pd.DataFrame(rows)

    def condition_table(self) -> pd.DataFrame:
        """One row per condition: the replicate mean and its spread."""
        rows = []
        for condition in self.conditions:
            row: dict = {"condition": condition.label, "role": condition.role}
            for name in self.factors():
                factor = condition.factors.get(name, Factor())
                unit = self.units.get(name, "")
                row[self.plate_map.label_for(name)] = factor.name
                row[f"{self.plate_map.label_for(name)} ({unit})"] = (
                    None if factor.conc is None else factor.conc.to(unit) if unit
                    else factor.conc.value)
            row.update({
                "wells": ", ".join(condition.wells),
                "n": condition.n,
                f"mean rate ({self.rate_label})": condition.mean,
                f"SD ({self.rate_label})": condition.sd,
                f"SEM ({self.rate_label})": condition.sem,
                "CV %": condition.cv,
                "% activity": condition.percent_activity,
                "% inhibition": condition.percent_inhibition,
                "flags": ", ".join(condition.flags),
            })
            for i, rate in enumerate(condition.rates, start=1):
                row[f"replicate {i}"] = rate
            rows.append(row)
        return pd.DataFrame(rows)

    def flagged_wells(self) -> list[str]:
        """Wells whose fit needs a human to look at it."""
        return sort_wells([w for w, r in self.wells.items()
                           if r.concerns or r.error])

    def summary(self) -> str:
        n_ok = sum(1 for r in self.wells.values() if r.usable and not r.concerns)
        lines = [
            f"{len(self.wells)} wells analysed, {len(self.conditions)} conditions, "
            f"{n_ok} wells with no flags.",
        ]
        if self.blank_summary:
            lines.append(self.blank_summary)
        if self.z_prime is not None:
            lines.append(f"Z' factor: {self.z_prime:.2f}")
        # Only judge spread where a rate is big enough for a CV to mean
        # anything - a blank scattering around zero always looks terrible.
        rates = [abs(c.mean) for c in self.conditions if c.role == "sample"]
        floor = 0.05 * max(rates, default=0.0)
        worst = sorted((c for c in self.conditions
                        if c.n > 1 and c.role == "sample"
                        and math.isfinite(c.cv) and abs(c.mean) > floor),
                       key=lambda c: -c.cv)[:1]
        if worst:
            lines.append(f"Worst replicate CV: {worst[0].cv:.1f}% ({worst[0].label}).")
        return "\n".join(lines)


# --- the pipeline ---------------------------------------------------------

def analyse(
    data: KineticData,
    plate_map: PlateMap,
    settings: DetectionSettings | None = None,
    blank_mode: str = "slope",
    rate_unit: str = "min",
    signal_label: str = "RFU",
    units: dict[str, str] | None = None,
    outlier_sd: float = 2.5,
    cv_warn: float = 20.0,
    well_settings: dict[str, DetectionSettings] | None = None,
    window_policy: WindowPolicy | None = None,
) -> AnalysisResult:
    """Fit every well, correct for background, and average replicates.

    By default the fitting window is chosen once - from the no-inhibitor
    controls - and shared by every well that matches them on protein and
    substrate, so that slopes are comparable.  ``window_policy`` controls that;
    see :mod:`enzkin.windows`.

    ``well_settings`` overrides everything for named wells, which is how a
    hand-picked window for one awkward curve is applied without disturbing the
    rest of the plate.
    """
    settings = settings or DetectionSettings()
    if blank_mode not in BLANK_MODES:
        raise ValueError(
            f"blank_mode must be one of {', '.join(BLANK_MODES)}, got {blank_mode!r}")
    units = {**plate_map.auto_units(), **(units or {})}
    warnings: list[str] = list(data.notes)

    wells = [w for w in plate_map.analysed_wells if w in data.wells]
    missing = [w for w in plate_map.analysed_wells if w not in data.wells]
    if missing:
        warnings.append(
            f"{len(missing)} mapped well(s) are not in the data file "
            f"(e.g. {', '.join(missing[:6])}) and were skipped.")
    unmapped = [w for w in data.wells if w not in plate_map.wells
                or plate_map.wells[w].role == "empty"]
    if unmapped:
        warnings.append(
            f"{len(unmapped)} well(s) in the data file are not mapped "
            f"(e.g. {', '.join(unmapped[:6])}) and were ignored.")
    if not wells:
        raise ValueError(
            "no wells to analyse - the plate map and the data file have no wells "
            "in common.  Check that the map uses the same well ids as the data.")

    blank_wells = [w for w in plate_map.wells_with_role("blank") if w in data.wells]
    blank_trace, blank_summary = None, ""
    if blank_mode == "trace" and blank_wells:
        blank_trace = np.nanmean(
            np.column_stack([data.series(w) for w in blank_wells]), axis=1)
        blank_summary = (f"Blank trace subtracted point by point "
                         f"(mean of {', '.join(blank_wells)}).")
    elif blank_mode != "none" and not blank_wells:
        blank_summary = "No blank wells on this plate; no background subtracted."

    # -- choose the window, then fit every well over it ---------------------
    factor_names = plate_map.active_factors()
    per_second_to_unit = time_factor(rate_unit)
    plan = plan_windows(data, plate_map, settings, window_policy)
    warnings.extend(plan.notes)

    def signal_for(well: str) -> np.ndarray:
        series = data.series(well)
        if blank_trace is not None and plate_map.wells[well].role != "blank":
            return series - blank_trace
        return series

    def settings_for(well: str) -> DetectionSettings:
        override = (well_settings or {}).get(well)
        if override is not None:
            return override
        window = plan.window_for(well)
        return settings if window is None else fixed_settings(
            settings, data.time, window)

    results: dict[str, WellResult] = {}
    for well in wells:
        info = plate_map.wells[well]
        try:
            fit = detect_linear_range(data.time, signal_for(well),
                                      settings_for(well))
        except LinearityError as exc:
            results[well] = WellResult(well, info, None, float("nan"),
                                       float("nan"), float("nan"),
                                       error=str(exc), flags=["not_fitted"])
            continue
        rate = fit.slope * per_second_to_unit
        results[well] = WellResult(
            well=well, info=info, fit=fit, rate=rate, raw_rate=rate,
            rate_se=fit.se_slope * per_second_to_unit, flags=list(fit.flags))

    # -- background correction --------------------------------------------
    if blank_mode == "slope" and blank_wells:
        usable_blanks = [w for w in blank_wells
                         if w in results and results[w].usable]
        if usable_blanks:
            match_on = _blank_match_factors(plate_map, blank_wells, factor_names)
            by_key: dict[tuple, list[str]] = {}
            for well in usable_blanks:
                by_key.setdefault(
                    _factor_key(results[well].info, match_on), []).append(well)

            for well, result in results.items():
                if result.info.role == "blank" or not result.usable:
                    continue
                matched = by_key.get(_factor_key(result.info, match_on),
                                     usable_blanks)
                # Measure the background over the same readings as the well it
                # corrects: a rate subtracted from one stretch of the run has
                # to have been measured over that same stretch.
                blank = _blank_rate(data, matched, result.fit, settings,
                                    per_second_to_unit)
                if blank is None:
                    continue
                result.blank_rate = blank
                result.rate = result.raw_rate - blank
            described = (f"matched on {', '.join(plate_map.label_for(f) for f in match_on)}"
                         if match_on else "plate mean")
            blank_summary = (
                f"Background subtracted as a rate ({described}, measured over "
                f"each well's own fitted window) from {len(usable_blanks)} "
                f"blank well(s): {', '.join(usable_blanks)}.")

    # -- controls: % activity and % inhibition ----------------------------
    control_keys = _control_match_factors(factor_names)
    controls = _control_lookup(results, plate_map, control_keys)
    for result in results.values():
        if not result.usable or result.info.role in {"blank", "empty"}:
            continue
        reference = controls.get(_factor_key(result.info, control_keys))
        if reference is None or reference == 0 or not math.isfinite(reference):
            continue
        # Rounded so that a control divided by itself reads 0, not -1.4e-14.
        result.percent_activity = round(100.0 * result.rate / reference, 6)
        result.percent_inhibition = round(100.0 - result.percent_activity, 6)

    # -- group technical replicates ---------------------------------------
    conditions: list[ConditionResult] = []
    for key, group_wells in plate_map.replicate_groups().items():
        members = [results[w] for w in sort_wells(group_wells) if w in results]
        usable = [m for m in members if m.usable and math.isfinite(m.rate)]
        if not members:
            continue
        rates = [m.rate for m in usable]
        info = members[0].info
        flags = sorted({f for m in members for f in m.flags})
        outliers: list[str] = []
        if len(rates) >= 3:
            values = np.array(rates)
            centre, spread = float(values.mean()), float(values.std(ddof=1))
            if spread > 0:
                outliers = [m.well for m, v in zip(usable, values)
                            if abs(v - centre) > outlier_sd * spread]
        mean = float(np.mean(rates)) if rates else float("nan")
        sd = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
        sem = sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
        cv = 100.0 * sd / abs(mean) if rates and mean not in (0.0,) else float("nan")
        group_flags = list(flags)
        if len(rates) > 1 and math.isfinite(cv) and cv > cv_warn:
            group_flags.append("replicates_disagree")
        if len(rates) < len(members):
            group_flags.append("some_wells_unfitted")
        activities = [m.percent_activity for m in usable
                      if m.percent_activity is not None]
        conditions.append(ConditionResult(
            key=key,
            label=info.condition_label(factor_names, units, plate_map.labels),
            wells=[m.well for m in members], rates=rates, role=info.role,
            factors={n: info.factor(n) for n in factor_names},
            mean=mean, sd=sd, sem=sem, cv=cv, n=len(rates),
            percent_activity=(round(float(np.mean(activities)), 6)
                              if activities else None),
            percent_inhibition=(round(100.0 - float(np.mean(activities)), 6)
                                if activities else None),
            flags=sorted(set(group_flags)), outliers=outliers,
            sort_key=_condition_sort_key(info, factor_names),
        ))
    # Read in the order a person would expect: samples first, then each factor
    # by concentration rather than by the alphabet ("12.5" before "3.125").
    conditions.sort(key=lambda c: (c.role != "sample", c.sort_key, c.label))

    result = AnalysisResult(
        data=data, plate_map=plate_map, settings=settings, wells=results,
        conditions=conditions, rate_unit=rate_unit, signal_label=signal_label,
        blank_mode=blank_mode, blank_summary=blank_summary, units=units,
        z_prime=_z_prime(results), warnings=warnings, window_plan=plan,
    )
    return result


def _condition_sort_key(info: WellInfo, names) -> tuple:
    key: list = []
    for name in names:
        factor = info.factor(name)
        key.append(factor.name or "")
        key.append(-1.0 if factor.conc is None else factor.conc.canonical)
    return tuple(key)


def _factor_key(info: WellInfo, names) -> tuple:
    return tuple(info.factor(n).key() for n in names)


def _blank_match_factors(plate_map: PlateMap, blank_wells, factor_names) -> list[str]:
    """Match blanks to samples on any factor the blanks actually vary in.

    A plate whose blanks are a substrate titration should have each sample
    corrected by the blank at its own substrate concentration; a plate with one
    kind of blank should use them all.
    """
    varying = []
    for name in factor_names:
        keys = {plate_map.wells[w].factor(name).key() for w in blank_wells}
        if len(keys) > 1:
            varying.append(name)
    return varying


def _blank_rate(data, blank_wells, fit, settings, per_second_to_unit):
    """Mean blank rate over exactly the readings ``fit`` used."""
    if fit is None:
        return None
    window = (fit.start_index, fit.end_index)
    rates = []
    for well in blank_wells:
        try:
            blank_fit = detect_linear_range(
                data.time, data.series(well),
                fixed_settings(settings, data.time, window))
        except LinearityError:
            continue
        rates.append(blank_fit.slope * per_second_to_unit)
    return float(np.mean(rates)) if rates else None


def _control_match_factors(factor_names) -> list[str]:
    """Everything except the test compound - that is what the control varies."""
    return [n for n in factor_names if n != "compound"]


def _control_lookup(results, plate_map: PlateMap, match_on) -> dict[tuple, float]:
    grouped: dict[tuple, list[float]] = {}
    for result in results.values():
        if result.info.role != "positive_control" or not result.usable:
            continue
        grouped.setdefault(_factor_key(result.info, match_on), []).append(result.rate)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _z_prime(results) -> float | None:
    """Assay window, when the plate carries both kinds of control."""
    high = [r.rate for r in results.values()
            if r.info.role == "positive_control" and r.usable]
    low = [r.rate for r in results.values()
           if r.info.role in {"negative_control", "blank"} and r.usable]
    if len(high) < 3 or len(low) < 3:
        return None
    separation = abs(float(np.mean(high)) - float(np.mean(low)))
    if separation == 0:
        return None
    spread = 3 * (float(np.std(high, ddof=1)) + float(np.std(low, ddof=1)))
    return float(1 - spread / separation)
