"""Figures: raw curves with the fitted range shown, and rate-vs-concentration.

Every figure is built so the fit can be checked by eye - the point of the raw
plate view is to let someone confirm in one glance that the tool picked the
range they would have picked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .analysis import AnalysisResult  # noqa: E402
from .plates import parse_well, row_label  # noqa: E402
from .units import _sig, time_factor  # noqa: E402


@dataclass(frozen=True)
class Theme:
    """Chart colours.  Both modes are chosen for their own surface."""

    surface: str
    page: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    raw: str
    fit: str
    flag_tint: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]

    @property
    def rc(self) -> dict:
        return {
            "figure.facecolor": self.page,
            "axes.facecolor": self.surface,
            "savefig.facecolor": self.page,
            "text.color": self.ink,
            "axes.labelcolor": self.secondary,
            "axes.edgecolor": self.axis,
            "xtick.color": self.muted,
            "ytick.color": self.muted,
            "grid.color": self.grid,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }


LIGHT = Theme(
    surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", secondary="#52514e",
    muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
    raw="#898781", fit="#2a78d6", flag_tint="#fdf3de",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
            "#4a3aa7", "#e34948"),
    sequential=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                "#0d366b"),
)

DARK = Theme(
    surface="#1a1a19", page="#0d0d0d", ink="#ffffff", secondary="#c3c2b7",
    muted="#898781", grid="#2c2c2a", axis="#383835",
    raw="#898781", fit="#3987e5", flag_tint="#3a3320",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
            "#9085e9", "#e66767"),
    sequential=("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
                "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
                "#cde2fb"),
)

THEMES = {"light": LIGHT, "dark": DARK}

#: Amber, paired with a glyph - a status colour never carries meaning alone.
WARNING = "#fab219"

#: More than this many series on one chart and identity stops being readable.
MAX_SERIES = 8


def _theme(theme: str | Theme) -> Theme:
    return theme if isinstance(theme, Theme) else THEMES[str(theme).lower()]


def _sequential_color(theme: Theme, fraction: float) -> str:
    ramp = theme.sequential
    index = int(round(min(max(fraction, 0.0), 1.0) * (len(ramp) - 1)))
    return ramp[index]


def _save(fig, path, dpi: int = 150):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --- the plate view -------------------------------------------------------

def plot_plate_curves(
    result: AnalysisResult,
    path=None,
    scale: str = "free",
    time_unit: str = "min",
    theme: str | Theme = "light",
    show_fit: bool = True,
    title: str | None = None,
):
    """Every well's raw curve in plate layout, with the fitted range picked out.

    ``scale`` is ``"free"`` (each well autoscaled - best for checking the shape
    of a fit), ``"shared"`` (one scale for the plate - best for comparing
    magnitudes) or ``"row"``/``"column"``.
    """
    palette = _theme(theme)
    data, plate = result.data, result.plate_map.plate
    n_rows, n_cols = plate.n_rows, plate.n_cols
    x = data.time / time_factor(time_unit)

    width = max(8.0, min(1.35 * n_cols, 30.0))
    height = max(5.0, min(1.05 * n_rows + 1.0, 24.0))
    with plt.rc_context(palette.rc):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height),
                                 sharex=True, squeeze=False)

        drawn = [w for w in data.wells if plate.contains(w)]
        limits = _scale_limits(result, drawn, scale, n_rows, n_cols)

        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row][col]
                well = f"{row_label(row)}{col + 1}"
                _draw_well(ax, well, result, x, palette, show_fit, time_unit)
                low, high = limits(row, col, well)
                if low is not None and high > low:
                    ax.set_ylim(low, high)
                ax.set_xlim(float(x.min()), float(x.max()))
                ax.tick_params(length=0, labelsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(palette.grid)
                    spine.set_linewidth(0.6)
                if row == 0:
                    ax.set_title(str(col + 1), color=palette.muted, fontsize=8,
                                 pad=4)
                if col == 0:
                    ax.set_ylabel(row_label(row), color=palette.muted,
                                  fontsize=8, rotation=0, ha="right", va="center",
                                  labelpad=8)

        heading = title or (
            f"Raw progress curves - {Path(result.data.source).name or 'plate'}")
        scale_text = ("each well autoscaled" if scale == "free"
                      else f"{scale} scale")
        subtitle = (f"{result.signal_label} against time ({time_unit}), "
                    f"{scale_text}; the number in each well is its fitted rate "
                    f"in {result.rate_label}.")
        fig.suptitle(heading, color=palette.ink, fontsize=13, y=1.0, x=0.005,
                     ha="left", va="top")
        fig.text(0.005, 0.975, subtitle, color=palette.secondary, fontsize=9,
                 ha="left", va="top")
        handles = [
            Line2D([], [], color=palette.raw, lw=1.0, label="reading"),
            Line2D([], [], color=palette.fit, lw=2.0, label="fitted linear range"),
            Patch(facecolor=palette.flag_tint, edgecolor=WARNING, lw=1.0,
                  label="! flagged - check this well"),
        ]
        fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.995, 1.0), labelcolor=palette.secondary)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def _scale_limits(result: AnalysisResult, wells, scale: str, n_rows, n_cols):
    """Return a callable giving the y-limits for each cell."""
    data = result.data

    def span(subset):
        values = [data.series(w) for w in subset if w in data.wells]
        if not values:
            return None, None
        stacked = np.concatenate(values)
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size == 0:
            return None, None
        low, high = float(stacked.min()), float(stacked.max())
        pad = max((high - low) * 0.08, 1e-9)
        return low - pad, high + pad

    scale = (scale or "free").lower()
    if scale == "shared":
        low, high = span(wells)
        return lambda row, col, well: (low, high)
    if scale.startswith("row"):
        cache = {r: span([w for w in wells if parse_well(w)[0] == r])
                 for r in range(n_rows)}
        return lambda row, col, well: cache.get(row, (None, None))
    if scale.startswith("col"):
        cache = {c: span([w for w in wells if parse_well(w)[1] == c])
                 for c in range(n_cols)}
        return lambda row, col, well: cache.get(col, (None, None))
    return lambda row, col, well: (None, None)


def _draw_well(ax, well, result: AnalysisResult, x, palette: Theme,
               show_fit: bool, time_unit: str) -> None:
    data = result.data
    if well not in data.wells:
        ax.set_facecolor(palette.page)
        ax.text(0.5, 0.5, well, transform=ax.transAxes, ha="center", va="center",
                fontsize=6, color=palette.muted)
        return

    y = data.series(well)
    outcome = result.wells.get(well)
    if outcome is None:  # mapped as empty, or not mapped at all
        ax.plot(x, y, color=palette.grid, lw=0.8)
        ax.text(0.04, 0.9, well, transform=ax.transAxes, fontsize=6,
                color=palette.muted, va="top")
        return

    if outcome.concerns:
        ax.set_facecolor(palette.flag_tint)
    ax.plot(x, y, color=palette.raw, lw=0.9, solid_capstyle="round")

    fit = outcome.fit
    if show_fit and fit is not None:
        lo, hi = fit.start_index, fit.end_index + 1
        ax.plot(x[lo:hi], y[lo:hi], color=palette.fit, lw=2.0,
                solid_capstyle="round", zorder=3)
        xs = np.array([fit.start_time, fit.end_time]) / time_factor(time_unit)
        ax.plot(xs, fit.predict([fit.start_time, fit.end_time]),
                color=palette.ink, lw=0.9, ls="--", zorder=4, alpha=0.75)

    label = well if outcome.role == "sample" else f"{well} {_role_tag(outcome.role)}"
    ax.text(0.04, 0.94, label, transform=ax.transAxes, fontsize=6,
            color=palette.secondary, va="top", fontweight="bold")
    if outcome.concerns:
        # Bottom-left: the only corner free of the well label and the rate.
        ax.text(0.04, 0.06, "!", transform=ax.transAxes, fontsize=8,
                color=palette.ink, va="bottom", ha="left", fontweight="bold")
    if outcome.usable and math.isfinite(outcome.rate):
        ax.text(0.96, 0.08, _sig(outcome.rate, 3), transform=ax.transAxes,
                fontsize=6, color=palette.secondary, va="bottom", ha="right",
                bbox={"facecolor": ax.get_facecolor(), "edgecolor": "none",
                      "pad": 0.8, "alpha": 0.8})


def _role_tag(role: str) -> str:
    return {"positive_control": "ctrl", "negative_control": "neg",
            "blank": "blank", "empty": ""}.get(role, "")


# --- one well, large ------------------------------------------------------

def plot_well(result: AnalysisResult, well: str, path=None,
              time_unit: str = "min", theme: str | Theme = "light"):
    """One curve at full size, with the fitted range and its statistics."""
    palette = _theme(theme)
    data = result.data
    x = data.time / time_factor(time_unit)
    y = data.series(well)
    outcome = result.wells.get(well)

    with plt.rc_context(palette.rc):
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.plot(x, y, color=palette.raw, lw=1.2, marker="o", ms=3,
                mfc=palette.surface, mec=palette.raw, mew=0.8, label="reading")
        if outcome is not None and outcome.fit is not None:
            fit = outcome.fit
            lo, hi = fit.start_index, fit.end_index + 1
            ax.plot(x[lo:hi], y[lo:hi], color=palette.fit, lw=2.5, zorder=3,
                    label="fitted linear range")
            ax.axvspan(x[lo], x[hi - 1], color=palette.fit, alpha=0.07, zorder=0)
            ends = np.array([fit.start_time, fit.end_time])
            ax.plot(ends / time_factor(time_unit), fit.predict(ends),
                    color=palette.ink, lw=1.1, ls="--", zorder=4,
                    label=f"slope {outcome.rate:,.3g} {result.rate_label}")
            info = (f"R² = {fit.r2:.4f}    {fit.n_points} points    "
                    f"{fit.start_time / 60:.1f}-{fit.end_time / 60:.1f} min")
            if outcome.concerns:
                info += "\n! " + ", ".join(outcome.concerns)
            ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top",
                    fontsize=9, color=palette.secondary)
        label = result.plate_map.wells[well].condition_label(
            result.factors(), result.units, result.plate_map.labels
        ) if well in result.plate_map.wells else ""
        ax.set_title(f"{well}  ·  {label}", color=palette.ink, loc="left")
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel(result.signal_label)
        ax.grid(True, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=9, labelcolor=palette.secondary)
        fig.tight_layout()
    return _save(fig, path)


# --- rate against concentration ------------------------------------------

def plot_rate_vs_concentration(
    result: AnalysisResult,
    x_factor: str = "substrate",
    series_factor: str | None = "compound",
    value: str = "rate",
    path=None,
    theme: str | Theme = "light",
    log_x: bool | None = None,
    title: str | None = None,
):
    """Replicate means against concentration - the Prism preview.

    ``x_factor`` on the x-axis, one series per level of ``series_factor``.
    Swapping the two turns a Michaelis-Menten panel into an IC50 panel.
    ``value`` is ``"rate"``, ``"percent_activity"`` or ``"percent_inhibition"``.
    """
    palette = _theme(theme)
    units = result.units
    conditions = [c for c in result.conditions
                  if c.role in {"sample", "positive_control"}
                  and c.factors.get(x_factor) is not None
                  and c.factors[x_factor].conc is not None]
    if not conditions:
        raise ValueError(
            f"no conditions carry a {x_factor} concentration to plot against")

    x_unit = units.get(x_factor, "M")
    series_unit = units.get(series_factor, "M") if series_factor else None

    grouped: dict[float | None, list] = {}
    for condition in conditions:
        key = None
        if series_factor:
            factor = condition.factors.get(series_factor)
            key = None if factor is None or factor.conc is None else factor.conc.canonical
        grouped.setdefault(key, []).append(condition)

    levels = sorted(grouped, key=lambda k: (k is None, k))
    note = ""
    if len(levels) > MAX_SERIES:
        picked = np.linspace(0, len(levels) - 1, MAX_SERIES).round().astype(int)
        kept = [levels[i] for i in sorted(set(picked))]
        note = (f"Showing {len(kept)} of {len(levels)} "
                f"{result.plate_map.label_for(series_factor)} levels, evenly "
                f"spaced; every level is in the exported tables.")
        levels = kept

    getter = {
        "rate": lambda c: (c.mean, c.sem),
        "percent_activity": lambda c: (c.percent_activity, None),
        "percent_inhibition": lambda c: (c.percent_inhibition, None),
    }[value]
    y_label = {"rate": f"Rate ({result.rate_label})",
               "percent_activity": "Activity (% of control)",
               "percent_inhibition": "Inhibition (%)"}[value]

    with plt.rc_context(palette.rc):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        all_x: list[float] = []
        real_x = [c.factors[x_factor].conc.to(x_unit) for c in conditions
                  if c.factors[x_factor].conc.to(x_unit) > 0]
        zero_x = (min(real_x) / 4.0
                  if real_x and any(c.factors[x_factor].conc.canonical == 0
                                    for c in conditions) else None)
        for index, level in enumerate(levels):
            members = sorted(
                grouped[level],
                key=lambda c: c.factors[x_factor].conc.canonical)
            xs, ys, errs = [], [], []
            for condition in members:
                y_value, err = getter(condition)
                if y_value is None or not math.isfinite(y_value):
                    continue
                value_x = condition.factors[x_factor].conc.to(x_unit)
                xs.append(zero_x if (value_x == 0 and zero_x is not None)
                          else value_x)
                ys.append(y_value)
                errs.append(err if err is not None else 0.0)
            if not xs:
                continue
            all_x.extend(xs)
            color = palette.series[index % len(palette.series)]
            label = _series_label(result, series_factor, level, series_unit)
            ax.errorbar(xs, ys, yerr=errs if any(errs) else None, color=color,
                        lw=2.0, marker="o", ms=6, mfc=palette.surface,
                        mec=color, mew=2.0, capsize=3, elinewidth=1.2,
                        label=label, zorder=3 + index)

        positive = [v for v in all_x if v > 0]
        if log_x is None:
            log_x = bool(positive) and (max(positive) / min(positive) >= 50)
        if log_x and positive:
            ax.set_xscale("log")
            if zero_x is not None:
                # A log axis cannot show zero, and dropping the no-inhibitor
                # control would quietly remove the most important point on the
                # chart.  Park it to the left of the lowest real concentration
                # and label the tick 0, the way a dose-response is normally read.
                ax.set_xlim(left=zero_x / 2.0)
                ticks = [t for t in ax.get_xticks()
                         if min(positive) <= t <= max(positive)]
                ax.set_xticks([zero_x] + list(ticks))
                ax.set_xticklabels(["0"] + [_axis_tick(t) for t in ticks])
                ax.axvline(zero_x * 1.7, color=palette.grid, lw=1.0, ls=(0, (2, 3)),
                           zorder=0)

        ax.set_xlabel(f"{result.plate_map.label_for(x_factor)} ({x_unit})")
        ax.set_ylabel(y_label)
        heading = title or (
            f"{y_label.split(' (')[0]} against "
            f"{result.plate_map.label_for(x_factor).lower()}")
        ax.set_title(heading, color=palette.ink, loc="left")
        ax.grid(True, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        if len(levels) > 1 and series_factor:
            ax.legend(title=f"{result.plate_map.label_for(series_factor)} "
                            f"({series_unit})",
                      fontsize=9, labelcolor=palette.secondary,
                      title_fontproperties={"size": 9, "weight": "bold"},
                      loc="best")
        if note:
            fig.text(0.01, -0.02, note, fontsize=8, color=palette.muted)
        fig.tight_layout()
    return _save(fig, path)


def _axis_tick(value: float) -> str:
    return _sig(value, 3)


def _series_label(result, series_factor, level, unit) -> str:
    if series_factor is None or level is None:
        return "control"
    from .units import Quantity, normalise_unit
    family, canonical = normalise_unit(unit)
    quantity = Quantity(level, "M" if family == "molar" else canonical, family)
    try:
        return _sig(quantity.to(unit), 3)
    except Exception:  # pragma: no cover - defensive
        return _sig(level, 3)


# --- plate heatmap --------------------------------------------------------

def plot_rate_heatmap(result: AnalysisResult, path=None,
                      theme: str | Theme = "light", title: str | None = None):
    """Fitted rate per well, laid out as the plate - a one-glance overview."""
    palette = _theme(theme)
    plate = result.plate_map.plate
    grid = np.full((plate.n_rows, plate.n_cols), np.nan)
    for well, outcome in result.wells.items():
        if outcome.usable and math.isfinite(outcome.rate):
            row, col = parse_well(well)
            grid[row, col] = outcome.rate

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("no fitted rates to draw")
    low, high = float(finite.min()), float(finite.max())

    with plt.rc_context(palette.rc):
        fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * plate.n_cols + 2.2),
                                        max(4.0, 0.52 * plate.n_rows + 2.0)))
        colours = matplotlib.colors.LinearSegmentedColormap.from_list(
            "enzkin_sequential", list(palette.sequential))
        image = ax.imshow(grid, cmap=colours, vmin=low, vmax=high,
                          aspect="auto")
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(range(plate.n_cols),
                      [str(c) for c in plate.columns], fontsize=8)
        ax.set_yticks(range(plate.n_rows), plate.rows, fontsize=8)
        ax.set_xticks(np.arange(-0.5, plate.n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, plate.n_rows, 1), minor=True)
        ax.grid(which="minor", color=palette.surface, lw=2)
        ax.tick_params(which="both", length=0)
        ax.set_title(title or f"Fitted rate per well ({result.rate_label})",
                     color=palette.ink, loc="left", pad=24)

        if plate.size <= 96:  # labels fit, so print them rather than rely on hue
            for row in range(plate.n_rows):
                for col in range(plate.n_cols):
                    value = grid[row, col]
                    if not math.isfinite(value):
                        continue
                    fraction = (value - low) / (high - low) if high > low else 0.5
                    shown = _sig(value, 3) if abs(value) >= 5e-4 else "0"
                    ax.text(col, row, shown, ha="center", va="center",
                            fontsize=6,
                            color="#ffffff" if fraction > 0.55 else palette.ink)
        bar = fig.colorbar(image, ax=ax, shrink=0.85)
        bar.ax.tick_params(labelsize=8, length=0)
        bar.set_label(f"rate ({result.rate_label})", color=palette.secondary,
                      fontsize=9)
        bar.outline.set_visible(False)
        fig.tight_layout()
    return _save(fig, path)


# --- why this window ------------------------------------------------------

def _rolling_rate(time: np.ndarray, signal: np.ndarray, width: int):
    """Local slope through a sliding window - the curve's rate over time."""
    finite = np.isfinite(time) & np.isfinite(signal)
    x, y = time[finite], signal[finite]
    if x.size < width + 1:
        return np.array([]), np.array([])
    half = width // 2
    centres, slopes = [], []
    for start in range(0, x.size - width + 1):
        xs, ys = x[start:start + width], y[start:start + width]
        var = np.sum((xs - xs.mean()) ** 2)
        if var <= 0:
            continue
        centres.append(xs[half] if width % 2 else xs.mean())
        slopes.append(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var)
    return np.array(centres), np.array(slopes)


def plot_window_choice(
    result: AnalysisResult,
    path=None,
    theme: str | Theme = "light",
    time_unit: str = "min",
    tolerance: float = 10.0,
):
    """Show why the fitting window is where it is.

    Top: the window every matched group ended up with, against the readings its
    reference wells were individually straight over.  Below, per group: the
    reference curves with the applied window shaded, and the same curves'
    *local* rate over time - the direct argument, since a flat local rate is
    what "still linear" means.  Where it rolls off is where the window ends.
    """
    palette = _theme(theme)
    plan = result.window_plan
    if plan is None or plan.scope_used == "well":
        raise ValueError(
            "no shared window to justify - each well was fitted over its own "
            "range (window scope 'well')")

    factor = time_factor(time_unit)
    groups = plan.groups
    detailed = [g for g in groups if g.reference_fits
                and "no usable reference" not in g.notes]
    if not detailed:
        raise ValueError("no reference wells had enough signal to show a window")

    n = len(detailed)
    width = max(9.0, min(2.9 * n + 1.5, 26.0))
    with plt.rc_context(palette.rc):
        fig = plt.figure(figsize=(width, 9.2))
        spec = fig.add_gridspec(3, n, height_ratios=[1.25, 1.0, 1.0],
                                hspace=0.42, wspace=0.28)

        strip = fig.add_subplot(spec[0, :])
        _window_strip(strip, result, groups, palette, factor, time_unit)

        for column, group in enumerate(detailed):
            curve_ax = fig.add_subplot(spec[1, column])
            rate_ax = fig.add_subplot(spec[2, column], sharex=curve_ax)
            _reference_panel(curve_ax, rate_ax, result, group, palette, factor,
                             time_unit, tolerance)
            if column == 0:
                curve_ax.set_ylabel(f"{result.signal_label}\n(reference wells)")
                rate_ax.set_ylabel("local rate\n(% of fitted rate)")
            rate_ax.set_xlabel(f"Time ({time_unit})")

        handles = [
            Line2D([], [], color=palette.series[0], lw=2.0, label="reference well"),
            Patch(facecolor=palette.grid, alpha=0.9, label="readings left out"),
            Line2D([], [], color=palette.fit, lw=1.4, ls=(0, (4, 2)),
                   label="edge of the window used for every well"),
            Line2D([], [], color=palette.muted, lw=1.2, ls="--",
                   label=f"±{tolerance:g}% of the fitted rate"),
        ]
        fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=9,
                   bbox_to_anchor=(0.995, 0.995), labelcolor=palette.secondary)
        fig.suptitle("Why this window", color=palette.ink, fontsize=13,
                     x=0.006, y=0.995, ha="left", va="top")
        fig.text(0.006, 0.966, plan.summary(), color=palette.secondary,
                 fontsize=9, ha="left", va="top")
        fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.07)
    return _save(fig, path)


def _shade_excluded(ax, x, start_t, end_t, palette: Theme) -> None:
    """Grey the readings outside the window, and mark its edges."""
    left, right = float(np.min(x)), float(np.max(x))
    if start_t > left:
        ax.axvspan(left, start_t, color=palette.grid, alpha=0.9, zorder=0)
    if end_t < right:
        ax.axvspan(end_t, right, color=palette.grid, alpha=0.9, zorder=0)
    for edge, shown in ((start_t, start_t > left), (end_t, end_t < right)):
        if shown:
            ax.axvline(edge, color=palette.fit, lw=1.4, ls=(0, (4, 2)),
                       zorder=5)


def _window_strip(ax, result, groups, palette: Theme, factor, time_unit):
    """One bar per matched group: the readings it is fitted over."""
    plan = result.window_plan
    time = plan.time
    labels = []
    for row, group in enumerate(reversed(groups)):
        applied = plan.well_window.get(group.wells[0])
        start, end = applied if applied else (group.start_index, group.end_index)
        ax.barh(row, (time[end] - time[start]) / factor,
                left=time[start] / factor, height=0.5,
                color=palette.fit, alpha=0.85, zorder=3)
        # What each reference well was individually straight over.
        for offset, (well, fit) in enumerate(sorted(group.reference_fits.items())):
            ax.plot([fit.start_time / factor, fit.end_time / factor],
                    [row + 0.34 + offset * 0.1] * 2, color=palette.muted,
                    lw=1.4, solid_capstyle="butt", zorder=4)
        if "no usable reference" in group.notes:
            ax.text(time[end] / factor + 1, row, "no signal to judge by",
                    va="center", fontsize=7.5, color=palette.muted)
        labels.append(group.label.replace(" | ", "\n"))

    ax.set_yticks(range(len(groups)), labels, fontsize=7.5)
    ax.set_xlim(float(time.min()) / factor, float(time.max()) / factor * 1.02)
    ax.set_ylim(-0.6, len(groups) - 0.25)
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_title("The window each matched group is fitted over "
                 "(grey ticks: where each reference well was straight on its own)",
                 color=palette.ink, loc="left", fontsize=10, pad=8)
    ax.grid(True, axis="x", lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _reference_panel(curve_ax, rate_ax, result, group, palette: Theme, factor,
                     time_unit, tolerance):
    data = result.data
    plan = result.window_plan
    applied = plan.well_window.get(group.wells[0],
                                   (group.start_index, group.end_index))
    start_t = float(plan.time[applied[0]]) / factor
    end_t = float(plan.time[applied[1]]) / factor
    x = data.time / factor
    colour = palette.series[0]

    width = max(5, min(int(round(0.12 * data.n_timepoints)), 25))
    for index, well in enumerate(group.reference_wells):
        if well not in data.wells:
            continue
        y = data.series(well)
        curve_ax.plot(x, y, color=colour, lw=1.4, alpha=0.9 - 0.2 * index,
                      zorder=3)
        outcome = result.wells.get(well)
        if outcome is not None and outcome.fit is not None:
            ends = np.array([outcome.fit.start_time, outcome.fit.end_time])
            curve_ax.plot(ends / factor, outcome.fit.predict(ends),
                          color=palette.ink, lw=0.9, ls="--", zorder=4,
                          alpha=0.7)
            reference_rate = outcome.fit.slope
        else:
            reference_rate = None

        centres, slopes = _rolling_rate(data.time, y, width)
        if centres.size and reference_rate:
            rate_ax.plot(centres / factor, 100 * slopes / reference_rate,
                         color=colour, lw=1.4, alpha=0.9 - 0.2 * index, zorder=3)

    for ax in (curve_ax, rate_ax):
        # Grey out what was dropped rather than tinting what was kept: when the
        # window is most of the run, tinting it colours the whole panel and the
        # thing worth seeing - what got left out - disappears.
        _shade_excluded(ax, x, start_t, end_t, palette)
        ax.grid(True, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.set_xlim(float(x.min()), float(x.max()))
    curve_ax.tick_params(labelbottom=False)

    rate_ax.axhline(100, color=palette.muted, lw=1.0, zorder=2)
    for edge in (100 - tolerance, 100 + tolerance):
        rate_ax.axhline(edge, color=palette.muted, lw=1.1, ls="--", zorder=2)
    low, high = rate_ax.get_ylim()
    rate_ax.set_ylim(max(0.0, min(low, 100 - 2.2 * tolerance)),
                     min(220.0, max(high, 100 + 2.2 * tolerance)))

    title = group.label.split(" | ")[-1]
    curve_ax.set_title(f"{title}\n{', '.join(group.reference_wells)}",
                       color=palette.secondary, fontsize=9, loc="left")
