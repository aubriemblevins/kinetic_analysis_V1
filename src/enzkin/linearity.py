"""Finding the linear range of a progress curve and fitting its slope.

A progress curve usually has three parts: a short lag while the reaction gets
going, a straight initial-velocity region, and a tail that bends over as
substrate is consumed or the detector saturates.  Only the straight part should
be fitted, and this module finds it without anyone dragging a mouse.

How the default method decides
------------------------------
The initial rate lives at the *start* of the reaction, so the search is
anchored there rather than roaming the whole curve.  For each plausible start
(the first reading, or a later one if there is a lag) the window is extended
for as long as the leftover scatter stays consistent with that well's own
noise.  Among those windows the fastest ones are kept and the longest is
returned.

Two details matter more than they look:

* **Straightness is judged against the well's own noise, not a fixed R^2.**
  On a dose-response plate the rates span a hundredfold; a well running at
  1 RFU/min can be perfectly straight and still never reach R^2 = 0.98, because
  its R^2 ceiling is set by the reader's noise.  Comparing the residual scatter
  with the noise asks the right question - "is there curvature here beyond what
  the instrument does anyway?" - at every signal level.  R^2 is still reported
  for every fit, and can still be imposed as an extra filter.
* **The search is anchored, and the candidate windows are long.**  Taking the
  steepest of many short windows is an upward-biased estimate: with enough
  candidates, some stretch of noise always drifts upwards.  That bias falls
  hardest on slow wells, which is where an inflated rate does the most damage
  to an IC50.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Selection methods understood by :func:`detect_linear_range`.
METHODS = ("auto", "max_slope", "fixed", "initial", "all")

FLAG_DESCRIPTIONS = {
    "low_signal": "Signal change is within noise - the slope is not meaningful.",
    "no_linear_range": ("Signal over the whole run barely beats the noise, so no "
                        "window stands out; the whole curve was fitted."),
    "poor_fit": "R^2 is below the reporting threshold - inspect this curve.",
    "relaxed": "No window was straight enough; the criteria had to be relaxed.",
    "short_window": "The fitted window is shorter than the requested minimum.",
    "lag_phase": "The fit starts after the first reading (a lag was skipped).",
    "curve_bends": "The fit ends before the last reading (the curve bent over).",
    "uses_whole_curve": "The whole curve was straight and was fitted end to end.",
    "saturated": "Repeated identical maximum readings - the detector may be saturated.",
    "few_points": "Fewer than 4 usable readings.",
    "decreasing": "Signal falls with time; slopes are negative by design.",
    "fast_reaction": ("The curve has largely levelled off by the end of the run, so "
                      "even the earliest readings are past the true initial rate - "
                      "read sooner, read faster, or dilute the enzyme."),
    "missing_data": "Some readings were blank or non-numeric and were skipped.",
}


class LinearityError(ValueError):
    """Raised when a curve cannot be fitted at all."""


@dataclass
class LinearFit:
    """The straight-line fit chosen for one well."""

    slope: float                 # signal units per second
    intercept: float
    r2: float
    se_slope: float              # standard error of the slope
    start_index: int             # index into the original (unfiltered) arrays
    end_index: int               # inclusive
    start_time: float            # seconds
    end_time: float              # seconds
    n_points: int
    noise: float = 0.0           # robust point-to-point noise estimate
    residual_se: float = 0.0     # scatter left once the line is removed
    final_rate_ratio: float = 1.0 # rate at the end of the run / fitted rate
    signal_change: float = 0.0   # fitted rise across the window
    curve_amplitude: float = 0.0 # total rise of the whole curve
    method: str = "auto"
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def fraction_of_curve(self) -> float:
        """How much of the curve's total rise happens inside the fit window."""
        if self.curve_amplitude <= 0:
            return 0.0
        return min(abs(self.signal_change) / self.curve_amplitude, 1.0)

    @property
    def snr(self) -> float:
        return abs(self.signal_change) / self.noise if self.noise > 0 else math.inf

    def rate(self, per: str = "min") -> float:
        """Slope expressed per minute (default), second or hour."""
        from .units import time_factor
        return self.slope * time_factor(per)

    def rate_se(self, per: str = "min") -> float:
        from .units import time_factor
        return self.se_slope * time_factor(per)

    def predict(self, x) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    @property
    def ok(self) -> bool:
        """True when nothing about this fit needs a human to look at it."""
        return not ({"low_signal", "no_linear_range", "poor_fit", "few_points",
                     "relaxed", "saturated", "fast_reaction"} & set(self.flags))


@dataclass
class DetectionSettings:
    """Knobs for linear-range detection.  The defaults suit most assays."""

    method: str = "auto"
    min_points: int | None = None        # None -> 10% of the curve, at least 5
    residual_tolerance: float = 1.25     # how much scatter counts as straight
    confidence: float = 0.99             # how sure before calling it curvature
    min_fraction_of_max_slope: float = 0.90
    max_lag_fraction: float = 0.25       # most of the curve that may be skipped
    r2_min: float | None = None          # optional extra straightness filter
    flag_r2_below: float = 0.90          # below this the fit is flagged for review
    window_points: int = 10              # for method="max_slope" / "initial"
    fixed_start: float | None = None     # seconds, for method="fixed"
    fixed_end: float | None = None
    search_start: float | None = None    # ignore readings before this (seconds)
    search_end: float | None = None
    direction: str = "auto"              # auto | increasing | decreasing
    smooth_points: int = 0               # smooth for *detection* only
    noise_sigma_threshold: float = 3.0   # signal must beat this many sigma
    low_signal_snr: float = 8.0          # below this, fit the whole curve
    slowdown_flag_ratio: float = 0.5     # flag if the run ends this much slower

    def resolved_min_points(self, n: int) -> int:
        """Floor on the window length, not a target.

        The search always extends a window as far as it stays straight, so this
        only needs to be short enough that a fast reaction - one largely over
        within a few readings - still has a linear region to find.
        """
        if self.min_points:
            return max(3, min(int(self.min_points), n))
        return max(4, min(n, min(int(round(0.05 * n)), 8)))


# --- basic fitting --------------------------------------------------------

def fit_line(x, y) -> tuple[float, float, float, float]:
    """Least-squares fit; returns ``(slope, intercept, r2, se_slope)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise LinearityError("a line needs at least two points")
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx == 0:
        raise LinearityError("all timepoints are identical")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - (intercept + slope * x)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 and ss_res > 0 else 0.0
    return slope, intercept, r2, se


def estimate_noise(y) -> float:
    """Robust point-to-point noise (from median absolute successive differences).

    Successive differences cancel the trend, and the median shrugs off a few
    wild points, so this measures the reader rather than the reaction.
    """
    y = np.asarray(y, dtype=float)
    diffs = np.diff(y[np.isfinite(y)])
    if diffs.size == 0:
        return 0.0
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sigma = 1.4826 * mad / math.sqrt(2)
    if sigma > 0:
        return sigma
    spread = float(np.std(diffs) / math.sqrt(2))
    return spread


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _chi2_over_df(df: np.ndarray | float, confidence: float) -> np.ndarray:
    """Upper critical value of chi-square divided by its degrees of freedom.

    Wilson-Hilferty approximation - accurate to well under a percent for the
    degrees of freedom seen here, and it keeps SciPy out of the dependency list.
    """
    df = np.maximum(np.asarray(df, dtype=float), 1.0)
    z = _normal_quantile(confidence)
    return (1.0 - 2.0 / (9.0 * df) + z * np.sqrt(2.0 / (9.0 * df))) ** 3


def _smooth(y: np.ndarray, points: int) -> np.ndarray:
    if points and points > 1:
        kernel = np.ones(int(points)) / float(int(points))
        padded = np.pad(y, (int(points) // 2, int(points) // 2), mode="edge")
        return np.convolve(padded, kernel, mode="same")[
            int(points) // 2: int(points) // 2 + y.size]
    return y


class _Cumulative:
    """Prefix sums that make any window's regression a constant-time lookup."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y
        self.cx = np.concatenate(([0.0], np.cumsum(x)))
        self.cy = np.concatenate(([0.0], np.cumsum(y)))
        self.cxx = np.concatenate(([0.0], np.cumsum(x * x)))
        self.cxy = np.concatenate(([0.0], np.cumsum(x * y)))
        self.cyy = np.concatenate(([0.0], np.cumsum(y * y)))

    def stats(self, starts, ends):
        """Regression stats for windows ``[start, end]`` (end inclusive)."""
        i = np.asarray(starts)
        j = np.asarray(ends) + 1
        m = (j - i).astype(float)
        sx = self.cx[j] - self.cx[i]
        sy = self.cy[j] - self.cy[i]
        sxx = self.cxx[j] - self.cxx[i]
        sxy = self.cxy[j] - self.cxy[i]
        syy = self.cyy[j] - self.cyy[i]
        var_x = sxx - sx * sx / m
        cov = sxy - sx * sy / m
        var_y = syy - sy * sy / m
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(var_x > 0, cov / var_x, np.nan)
            ss_res = np.maximum(var_y - slope * cov, 0.0)
            r2 = np.where(var_y > 0, 1.0 - ss_res / var_y, 1.0)
            df = np.maximum(m - 2.0, 1.0)
            resid = np.sqrt(ss_res / df)
            se = np.where(var_x > 0, resid / np.sqrt(var_x), np.inf)
        return slope, r2, resid, se, m


def _direction_sign(x: np.ndarray, y: np.ndarray, setting: str) -> int:
    if setting.startswith("inc"):
        return 1
    if setting.startswith("dec"):
        return -1
    third = max(1, y.size // 3)
    delta = float(np.median(y[-third:]) - np.median(y[:third]))
    if delta == 0:
        slope, *_ = fit_line(x, y)
        delta = slope
    return -1 if delta < 0 else 1


def _saturation_run(y: np.ndarray) -> int:
    """Longest run of identical readings at the curve's maximum."""
    if y.size == 0:
        return 0
    top = float(np.nanmax(y))
    best = run = 0
    for value in y:
        run = run + 1 if value == top else 0
        best = max(best, run)
    return best


# --- the main entry point -------------------------------------------------

def detect_linear_range(time, signal, settings: DetectionSettings | None = None) -> LinearFit:
    """Choose the linear region of one progress curve and fit it.

    ``time`` is in seconds and ``signal`` in whatever the reader produced.
    Missing readings are skipped.  The returned :class:`LinearFit` records the
    window used, its quality, and any flags worth a second look.
    """
    settings = settings or DetectionSettings()
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.shape != signal.shape:
        raise LinearityError("time and signal must be the same length")

    usable = np.isfinite(time) & np.isfinite(signal)
    dropped = int(np.count_nonzero(~usable))
    if settings.search_start is not None:
        usable &= time >= settings.search_start
    if settings.search_end is not None:
        usable &= time <= settings.search_end
    index_map = np.flatnonzero(usable)
    x, y = time[usable], signal[usable]
    n = x.size

    if n < 2:
        raise LinearityError(
            "fewer than two usable readings - check the data and the time range"
        )

    noise = estimate_noise(y)
    amplitude = float(np.nanmax(y) - np.nanmin(y))
    flags: list[str] = []
    if dropped:
        flags.append("missing_data")
    if n < 4:
        flags.append("few_points")

    sign = _direction_sign(x, y, settings.direction)
    if sign < 0:
        flags.append("decreasing")
    if _saturation_run(y) >= 3:
        flags.append("saturated")

    method = settings.method
    if method == "all" or n < 4:
        start_idx, end_idx, method_used = 0, n - 1, "all"
    elif method == "fixed":
        start_idx, end_idx = _fixed_window(x, settings)
        method_used = "fixed"
    elif method == "initial":
        count = max(3, min(settings.window_points, n))
        start_idx, end_idx, method_used = 0, count - 1, "initial"
    elif method == "max_slope":
        start_idx, end_idx = _max_slope_window(x, y, settings, sign)
        method_used = "max_slope"
    else:
        start_idx, end_idx, auto_flags = _auto_window(x, y, settings, sign, noise)
        flags.extend(auto_flags)
        method_used = "auto"

    xs, ys = x[start_idx:end_idx + 1], y[start_idx:end_idx + 1]
    slope, intercept, r2, se = fit_line(xs, ys)
    change = slope * (xs[-1] - xs[0])
    residual_se = math.sqrt(
        max(float(np.sum((ys - (intercept + slope * xs)) ** 2)), 0.0)
        / max(xs.size - 2, 1))

    # How fast is the reaction still going at the end of the run?  A curve that
    # has flattened out was already past its initial rate when the first
    # reading was taken, and no choice of window can recover that.
    tail = max(4, min(n, int(round(0.25 * n))))
    final_ratio = 1.0
    if n >= 6 and abs(slope) > 0:
        try:
            tail_slope, *_ = fit_line(x[-tail:], y[-tail:])
            final_ratio = float(tail_slope / slope)
        except LinearityError:
            final_ratio = 1.0
    if (final_ratio < settings.slowdown_flag_ratio
            and "low_signal" not in flags and "no_linear_range" not in flags):
        flags.append("fast_reaction")

    if xs.size < settings.resolved_min_points(n) and method_used == "auto":
        flags.append("short_window")
    if start_idx > 0:
        flags.append("lag_phase")
    if end_idx < n - 1:
        flags.append("curve_bends")
    elif start_idx == 0 and method_used == "auto":
        flags.append("uses_whole_curve")
    if r2 < settings.flag_r2_below:
        flags.append("poor_fit")
    if noise > 0 and abs(change) < settings.noise_sigma_threshold * noise:
        flags.append("low_signal")

    return LinearFit(
        slope=float(slope), intercept=float(intercept), r2=float(r2),
        se_slope=float(se),
        start_index=int(index_map[start_idx]), end_index=int(index_map[end_idx]),
        start_time=float(xs[0]), end_time=float(xs[-1]), n_points=int(xs.size),
        noise=float(noise), residual_se=float(residual_se),
        final_rate_ratio=float(final_ratio),
        signal_change=float(change), curve_amplitude=float(amplitude),
        method=method_used, flags=sorted(set(flags)),
    )


def _fixed_window(x: np.ndarray, settings: DetectionSettings) -> tuple[int, int]:
    start = settings.fixed_start if settings.fixed_start is not None else x[0]
    end = settings.fixed_end if settings.fixed_end is not None else x[-1]
    inside = np.flatnonzero((x >= start - 1e-9) & (x <= end + 1e-9))
    if inside.size < 2:
        raise LinearityError(
            f"the fixed window {start:g}-{end:g} s contains "
            f"{inside.size} reading(s); widen it"
        )
    return int(inside[0]), int(inside[-1])


def _max_slope_window(x, y, settings: DetectionSettings, sign: int) -> tuple[int, int]:
    """Steepest window of a fixed width - the classic 'Vmax points' rule."""
    n = x.size
    width = max(3, min(settings.window_points, n))
    cum = _Cumulative(x, y)
    starts = np.arange(0, n - width + 1)
    slope, r2, _resid, _se, _m = cum.stats(starts, starts + width - 1)
    oriented = slope * sign
    order = np.argsort(-np.nan_to_num(oriented, nan=-np.inf))
    best = int(order[0])
    if settings.r2_min is not None:
        for candidate in order:
            if r2[candidate] >= settings.r2_min:
                best = int(candidate)
                break
    start = int(starts[best])
    return start, start + width - 1


def _straight_end(cum: _Cumulative, start: int, n: int, min_pts: int,
                  noise: float, settings: DetectionSettings,
                  tolerance: float) -> int | None:
    """Furthest reading the window from ``start`` can reach and stay straight.

    Straight means the residual scatter is no larger than this well's own
    noise, allowing for the fact that a short window's scatter is itself a
    noisy estimate (hence the chi-square critical value).
    """
    last = n - 1
    ends = np.arange(start + min_pts - 1, last + 1)
    if ends.size == 0:
        return None
    _slope, r2, resid, _se, m = cum.stats(np.full(ends.shape, start), ends)
    limit = noise * tolerance * np.sqrt(_chi2_over_df(m - 2.0, settings.confidence))
    ok = np.isfinite(resid) & (resid <= limit)
    if settings.r2_min is not None:
        ok &= r2 >= settings.r2_min
    if not np.any(ok):
        return None
    return int(ends[np.flatnonzero(ok)[-1]])


def _auto_window(x, y, settings: DetectionSettings, sign: int, noise: float):
    """Longest straight window anchored at the start of the reaction."""
    n = x.size
    flags: list[str] = []
    detect_y = _smooth(y, settings.smooth_points)
    min_pts = settings.resolved_min_points(n)

    global_slope, _, _, _ = fit_line(x, detect_y)
    global_change = abs(global_slope * (x[-1] - x[0]))
    if noise > 0 and global_change < settings.low_signal_snr * noise:
        # The whole run barely moves.  Any "steepest window" here is a stretch
        # of noise that happened to drift, and that inflates the rate of
        # exactly the wells - fully inhibited, no-enzyme, empty - where an
        # inflated rate does the most damage.  Fit everything instead: for a
        # flat well that is the honest answer, and for a genuinely slow but
        # clean well the whole curve *is* the linear range.
        return 0, n - 1, ["no_linear_range"]
    if noise <= 0:  # a perfectly noiseless (usually synthetic) curve
        noise = max(abs(global_change), 1.0) * 1e-9

    cum = _Cumulative(x, detect_y)
    max_start = max(0, min(int(settings.max_lag_fraction * n), n - min_pts))
    starts = np.arange(0, max_start + 1)

    for attempt, (tolerance, floor_pts) in enumerate(
            [(settings.residual_tolerance, min_pts),
             (settings.residual_tolerance * 1.5, max(4, min_pts // 2)),
             (settings.residual_tolerance * 3.0, 4)]):
        candidates = []
        for start in starts:
            end = _straight_end(cum, int(start), n, floor_pts, noise, settings,
                                tolerance)
            if end is not None:
                candidates.append((int(start), int(end)))
        if candidates:
            if attempt:
                flags.append("relaxed")
            break
    else:
        return 0, n - 1, ["relaxed", "poor_fit"]

    # Every candidate is long and straight, so their slopes carry little
    # sampling noise and the fastest of them is a fair reference point.
    starts_arr = np.array([c[0] for c in candidates])
    ends_arr = np.array([c[1] for c in candidates])
    slopes, _r2, _resid, _se, _m = cum.stats(starts_arr, ends_arr)
    oriented = slopes * sign
    best_slope = float(np.nanmax(oriented))
    if best_slope <= 0:
        return 0, n - 1, flags + ["low_signal"]

    keep = np.flatnonzero(oriented >= settings.min_fraction_of_max_slope * best_slope)
    lengths = ends_arr - starts_arr + 1
    # Longest wins; ties go to the steeper window, then the earlier one.
    chosen = max(keep, key=lambda k: (lengths[k], oriented[k], -starts_arr[k]))
    return int(starts_arr[chosen]), int(ends_arr[chosen]), flags
