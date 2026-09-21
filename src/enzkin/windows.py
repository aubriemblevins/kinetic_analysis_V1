"""Choosing one linear range and applying it to matched wells.

A slope is only comparable with another slope if both came from the same
stretch of the reaction.  Fitting well A over 0-30 min and well B over 0-60 min
biases the comparison in a known direction: on a decelerating curve the longer
window averages in more of the slow tail, so B reads slower than it is, for
reasons that have nothing to do with what is in the well.

So the window is chosen once and shared.  It is chosen from the **no-inhibitor
controls**, because they are the fastest wells on the plate and therefore the
first to bend: a window over which the controls are straight is a window over
which every inhibited well sharing their protein and substrate is also
straight.  Wells are matched on protein and substrate concentration, since how
long a reaction stays linear depends on how fast it consumes its substrate.

Scopes
------
``plate``  one window for every well - the most comparable.
``group``  one window per protein/substrate combination - needed when the
           substrate range is wide enough that no single window suits all of it.
``auto``   plate-wide when that still leaves a usable window, otherwise
           per-group, saying which it chose and why (the default).
``well``   every well fitted over its own range.  Not comparable across wells;
           kept for inspecting a single curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .linearity import (INFORMATIONAL_FLAGS, DetectionSettings, LinearFit,
                        LinearityError, detect_linear_range)
from .platemap import Factor, PlateMap, WellInfo
from .plates import sort_wells
from .readers import KineticData

SCOPES = ("auto", "plate", "group", "well")
REFERENCES = ("controls", "fastest", "all")
CONSENSUS = ("intersection", "median")

#: A reference well tells us nothing about linearity if it never went anywhere.
_UNINFORMATIVE = {"low_signal", "no_linear_range", "few_points"}


@dataclass
class WindowPolicy:
    """How the fitting window is chosen and how widely it is shared."""

    scope: str = "auto"
    reference: str = "controls"
    consensus: str = "intersection"
    match_on: list[str] | None = None      # None -> every factor but the compound
    #: A plate-wide window is only used if it keeps this much of the typical
    #: per-group window; below that, per-group windows are used instead.
    plate_min_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name, value, allowed in (("scope", self.scope, SCOPES),
                                     ("reference", self.reference, REFERENCES),
                                     ("consensus", self.consensus, CONSENSUS)):
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {', '.join(allowed)}, got {value!r}")


@dataclass
class WindowGroup:
    """One set of wells that share a window, and the evidence for it."""

    key: tuple
    label: str
    wells: list[str]
    factors: dict[str, Factor]
    reference_wells: list[str] = field(default_factory=list)
    reference_fits: dict[str, LinearFit] = field(default_factory=dict)
    start_index: int = 0
    end_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    reference_kind: str = "controls"
    notes: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return self.end_index - self.start_index + 1

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def describe(self) -> str:
        return (f"{self.start_time / 60:.1f}-{self.end_time / 60:.1f} min "
                f"({self.n_points} readings)")


@dataclass
class WindowPlan:
    """The window every well will be fitted over, and why."""

    policy: WindowPolicy
    groups: list[WindowGroup]
    well_window: dict[str, tuple[int, int]]
    scope_used: str
    time: np.ndarray
    plate_window: tuple[int, int] | None = None
    match_on: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_uniform(self) -> bool:
        return len(set(self.well_window.values())) <= 1

    def window_for(self, well: str) -> tuple[int, int] | None:
        return self.well_window.get(well)

    def group_for(self, well: str) -> WindowGroup | None:
        for group in self.groups:
            if well in group.wells:
                return group
        return None

    def summary(self) -> str:
        if self.scope_used == "well":
            return "Each well was fitted over its own linear range."
        if self.is_uniform and self.plate_window is not None:
            start, end = self.plate_window
            return (f"One window for the whole plate: "
                    f"{self.time[start] / 60:.1f}-{self.time[end] / 60:.1f} min "
                    f"({end - start + 1} readings), chosen from "
                    f"{self._reference_description()}.")
        spans = [g.describe() for g in self.groups]
        return (f"{len(set(spans))} window(s) across {len(self.groups)} matched "
                f"group(s), each chosen from {self._reference_description()}.")

    def _reference_description(self) -> str:
        # Only groups that actually constrained the window get a say in how it
        # is described; a group with nothing to measure constrained nothing.
        kinds = {g.reference_kind for g in self.groups
                 if "no usable reference" not in g.notes}
        kinds = kinds or {g.reference_kind for g in self.groups}
        if kinds == {"controls"}:
            return "the no-inhibitor controls"
        if "controls" in kinds:
            return "the no-inhibitor controls (fastest wells where none exist)"
        return "the fastest wells in each group"

    def to_rows(self) -> list[dict]:
        """One row per group - the table behind the figure."""
        rows = []
        for group in self.groups:
            applied = [self.well_window[w] for w in group.wells
                       if w in self.well_window]
            start, end = applied[0] if applied else (group.start_index,
                                                     group.end_index)
            rows.append({
                "group": group.label,
                "wells": len(group.wells),
                "reference wells": ", ".join(group.reference_wells),
                "reference basis": group.reference_kind,
                "group window (min)": (f"{group.start_time / 60:.1f}-"
                                       f"{group.end_time / 60:.1f}"),
                "applied window (min)": (f"{self.time[start] / 60:.1f}-"
                                         f"{self.time[end] / 60:.1f}"),
                "readings used": end - start + 1,
                "notes": "; ".join(group.notes),
            })
        return rows


def default_match_factors(plate_map: PlateMap) -> list[str]:
    """Match on everything except the test compound - that is what varies."""
    return [f for f in plate_map.active_factors() if f != "compound"]


def fixed_settings(settings: DetectionSettings, time: np.ndarray,
                   window: tuple[int, int]) -> DetectionSettings:
    """Settings that fit exactly the readings in ``window``."""
    start, end = window
    return replace(settings, method="fixed",
                   fixed_start=float(time[start]), fixed_end=float(time[end]))


def plan_windows(
    data: KineticData,
    plate_map: PlateMap,
    settings: DetectionSettings | None = None,
    policy: WindowPolicy | None = None,
) -> WindowPlan:
    """Work out which readings each well should be fitted over."""
    settings = settings or DetectionSettings()
    policy = policy or WindowPolicy()
    match_on = (list(policy.match_on) if policy.match_on is not None
                else default_match_factors(plate_map))

    wells = [w for w in plate_map.analysed_wells if w in data.wells]
    if policy.scope == "well" or not wells:
        return WindowPlan(policy=policy, groups=[], well_window={},
                          scope_used="well", time=data.time, match_on=match_on)

    usable = np.flatnonzero(np.isfinite(data.time))
    full = (int(usable[0]), int(usable[-1]))
    min_points = settings.resolved_min_points(len(data.time))

    groups: list[WindowGroup] = []
    grouped: dict[tuple, list[str]] = {}
    for well in wells:
        info = plate_map.wells[well]
        grouped.setdefault(_key(info, match_on), []).append(well)

    units = plate_map.auto_units()
    for key, members in grouped.items():
        info = plate_map.wells[members[0]]
        group = WindowGroup(
            key=key, label=_group_label(info, match_on, units, plate_map),
            wells=sort_wells(members),
            factors={name: info.factor(name) for name in match_on})
        _choose_group_window(group, data, plate_map, settings, policy,
                             min_points, full)
        groups.append(group)
    groups.sort(key=lambda g: _sort_key(g, match_on))

    informative = [g for g in groups if "no usable reference" not in g.notes]
    pool = informative or groups
    plate_window = (max(g.start_index for g in pool),
                    min(g.end_index for g in pool))
    if plate_window[1] - plate_window[0] + 1 < max(3, min(min_points, 3)):
        plate_window = full

    scope_used, notes = _decide_scope(policy, groups, plate_window, data.time,
                                      min_points)

    well_window: dict[str, tuple[int, int]] = {}
    for group in groups:
        window = plate_window if scope_used == "plate" else (group.start_index,
                                                             group.end_index)
        for well in group.wells:
            well_window[well] = window

    return WindowPlan(policy=policy, groups=groups, well_window=well_window,
                      scope_used=scope_used, time=data.time,
                      plate_window=plate_window, match_on=match_on, notes=notes)


def _decide_scope(policy, groups, plate_window, time, min_points):
    """Plate-wide if that still leaves a usable window; otherwise per group."""
    notes: list[str] = []
    plate_points = plate_window[1] - plate_window[0] + 1
    plate_span = time[plate_window[1]] - time[plate_window[0]]
    spans = [g.duration for g in groups] or [plate_span]
    typical = float(np.median(spans))

    if policy.scope == "plate":
        if plate_points < min_points:
            notes.append(
                f"A plate-wide window leaves only {plate_points} reading(s); "
                "it was used as asked, but check the figure.")
        return "plate", notes
    if policy.scope == "group":
        return "group", notes

    if len({(g.start_index, g.end_index) for g in groups}) == 1:
        return "plate", notes
    if plate_points >= min_points and (
            typical <= 0 or plate_span >= policy.plate_min_fraction * typical):
        notes.append(
            f"One window suits the whole plate: it keeps "
            f"{plate_span / typical:.0%} of the typical matched-group window.")
        return "plate", notes
    notes.append(
        f"No single window suits the whole plate - a plate-wide window would "
        f"keep only {plate_span / typical:.0%} of the typical matched-group "
        f"window, so each protein/substrate group uses its own.")
    return "group", notes


def _choose_group_window(group, data, plate_map, settings, policy, min_points,
                         full) -> None:
    references, kind = _references(group, data, plate_map, settings, policy)
    group.reference_kind = kind

    fits: dict[str, LinearFit] = {}
    for well in references:
        try:
            fit = detect_linear_range(data.time, data.series(well), settings)
        except LinearityError:
            continue
        fits[well] = fit

    informative = {w: f for w, f in fits.items()
                   if not (_UNINFORMATIVE & set(f.flags))}
    if not informative:
        group.reference_wells = sort_wells(fits)
        group.reference_fits = fits
        group.start_index, group.end_index = full
        group.notes.append("no usable reference")
        group.notes.append(
            "No reference well here had enough signal to show where linearity "
            "ends, so the whole run was kept and this group does not constrain "
            "the plate-wide window.")
        _stamp_times(group, data.time)
        return

    group.reference_wells = sort_wells(informative)
    group.reference_fits = fits
    starts = [f.start_index for f in informative.values()]
    ends = [f.end_index for f in informative.values()]

    if policy.consensus == "median":
        start, end = int(np.median(starts)), int(np.median(ends))
    else:
        start, end = max(starts), min(ends)
        if end - start + 1 < min_points and len(starts) > 1:
            start, end = int(np.median(starts)), int(np.median(ends))
            group.notes.append(
                "The reference wells' ranges barely overlap, so the median of "
                "them was used rather than the strict overlap.")
    if end <= start:
        start, end = full
        group.notes.append(
            "The reference wells' ranges did not overlap at all; the whole run "
            "was kept.  Look at these curves.")
    group.start_index, group.end_index = int(start), int(end)
    _stamp_times(group, data.time)


def _stamp_times(group: WindowGroup, time: np.ndarray) -> None:
    group.start_time = float(time[group.start_index])
    group.end_time = float(time[group.end_index])


def _references(group, data, plate_map, settings, policy):
    """Which wells get to decide this group's window."""
    if policy.reference == "all":
        return group.wells, "all wells"

    if policy.reference == "controls":
        controls = [w for w in group.wells
                    if plate_map.wells[w].role == "positive_control"]
        if controls:
            return controls, "controls"

    fastest = _fastest_wells(group.wells, data, settings)
    if not fastest:
        return group.wells, "all wells"
    if policy.reference == "controls":
        group.notes.append(
            "No no-inhibitor control shares these conditions, so the fastest "
            "wells in the group were used instead.")
    return fastest, "fastest wells"


def _fastest_wells(wells, data, settings, keep: float = 0.25):
    """The quickest wells in a group - they bend before the others do."""
    rates: list[tuple[float, str]] = []
    for well in wells:
        try:
            fit = detect_linear_range(data.time, data.series(well), settings)
        except LinearityError:
            continue
        if _UNINFORMATIVE & set(fit.flags):
            continue
        rates.append((abs(fit.slope), well))
    if not rates:
        return []
    rates.sort(reverse=True)
    count = max(2, math.ceil(keep * len(rates)))
    return sort_wells(w for _rate, w in rates[:count])


def _key(info: WellInfo, names) -> tuple:
    return tuple(info.factor(name).key() for name in names)


def _sort_key(group: WindowGroup, names) -> tuple:
    key: list = []
    for name in names:
        factor = group.factors.get(name, Factor())
        key.append(factor.name or "")
        key.append(-1.0 if factor.conc is None else factor.conc.canonical)
    return tuple(key)


def _group_label(info: WellInfo, names, units, plate_map: PlateMap) -> str:
    parts = []
    for name in names:
        factor = info.factor(name)
        if factor.is_empty:
            continue
        parts.append(f"{plate_map.label_for(name)} {factor.label(units.get(name))}")
    return " | ".join(parts) or "all wells"
