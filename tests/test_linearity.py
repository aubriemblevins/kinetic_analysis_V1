import numpy as np
import pytest

from enzkin.linearity import (DetectionSettings, LinearityError,
                              detect_linear_range, estimate_noise, fit_line)

MINUTE = 60.0


def _progress_curve(amplitude, k_per_min, noise, n=121, lag_min=0.0, seed=3):
    """An exponential approach to a plateau - the usual shape of a real one."""
    rng = np.random.default_rng(seed)
    time = np.arange(1, n + 1) * MINUTE
    minutes = np.maximum(time / MINUTE - lag_min, 0.0)
    signal = 1000 + amplitude * (1 - np.exp(-k_per_min * minutes))
    return time, signal + rng.normal(0, noise, time.size), amplitude * k_per_min


def test_a_straight_line_is_recovered_exactly(straight_curve):
    time, signal, true_rate = straight_curve
    fit = detect_linear_range(time, signal)
    assert fit.rate("min") == pytest.approx(true_rate, rel=0.02)
    assert fit.r2 > 0.99


def test_a_straight_line_uses_the_whole_curve(straight_curve):
    time, signal, _ = straight_curve
    fit = detect_linear_range(time, signal)
    assert "uses_whole_curve" in fit.flags
    assert fit.n_points == time.size


def test_a_lag_is_skipped():
    time, signal, _ = _progress_curve(6000, 0.02, 20, lag_min=15)
    fit = detect_linear_range(time, signal)
    assert "lag_phase" in fit.flags
    assert fit.start_time / MINUTE >= 10


def test_a_bending_curve_is_trimmed():
    time, signal, _ = _progress_curve(6000, 0.06, 20)
    fit = detect_linear_range(time, signal)
    assert "curve_bends" in fit.flags
    assert fit.end_time < time[-1]


def test_trimming_beats_fitting_the_whole_curve():
    """The point of detection: a windowed fit is much closer to the truth."""
    time, signal, truth = _progress_curve(6000, 0.06, 20)
    windowed = detect_linear_range(time, signal).rate("min")
    whole = detect_linear_range(
        time, signal, DetectionSettings(method="all")).rate("min")
    assert abs(windowed - truth) < abs(whole - truth) / 2


def test_a_flat_well_reports_about_zero_and_says_so():
    rng = np.random.default_rng(5)
    time = np.arange(1, 122) * MINUTE
    signal = np.full(121, 7.0) + rng.normal(0, 5, 121)
    fit = detect_linear_range(time, signal)
    assert abs(fit.rate("min")) < 0.2
    assert "low_signal" in fit.flags or "no_linear_range" in fit.flags


def test_a_noisy_flat_well_is_not_given_a_steep_window():
    """Without a noise floor, the steepest stretch of noise wins - and inflates
    exactly the fully-inhibited wells that set an IC50's bottom plateau."""
    rng = np.random.default_rng(11)
    time = np.arange(1, 122) * MINUTE
    signal = 1400 + 0.5 * (time / MINUTE) + rng.normal(0, 18, 121)
    fit = detect_linear_range(time, signal)
    assert fit.rate("min") == pytest.approx(0.5, abs=0.4)


def test_a_falling_curve_is_handled():
    rng = np.random.default_rng(7)
    time = np.arange(1, 61) * MINUTE
    signal = 5000 - 30 * (time / MINUTE) + rng.normal(0, 8, 60)
    fit = detect_linear_range(time, signal)
    assert fit.rate("min") == pytest.approx(-30, rel=0.05)
    assert "decreasing" in fit.flags


def test_a_reaction_too_fast_for_the_sampling_is_flagged():
    time, signal, _ = _progress_curve(4000, 0.4, 15)
    fit = detect_linear_range(time, signal)
    assert "fast_reaction" in fit.flags


def test_fixed_window_uses_exactly_that_window(straight_curve):
    time, signal, true_rate = straight_curve
    fit = detect_linear_range(time, signal, DetectionSettings(
        method="fixed", fixed_start=10 * MINUTE, fixed_end=30 * MINUTE))
    assert fit.start_time == pytest.approx(10 * MINUTE)
    assert fit.end_time == pytest.approx(30 * MINUTE)
    assert fit.rate("min") == pytest.approx(true_rate, rel=0.15)


def test_max_slope_window_has_the_requested_width(straight_curve):
    time, signal, _ = straight_curve
    fit = detect_linear_range(time, signal, DetectionSettings(
        method="max_slope", window_points=8))
    assert fit.n_points == 8


def test_missing_readings_are_skipped(straight_curve):
    time, signal, true_rate = straight_curve
    signal = signal.copy()
    signal[[3, 4, 20]] = np.nan
    fit = detect_linear_range(time, signal)
    assert fit.rate("min") == pytest.approx(true_rate, rel=0.03)
    assert "missing_data" in fit.flags


def test_search_window_limits_the_data_used(straight_curve):
    time, signal, _ = straight_curve
    fit = detect_linear_range(time, signal, DetectionSettings(
        search_start=20 * MINUTE))
    assert fit.start_time >= 20 * MINUTE


def test_too_few_readings_is_refused():
    with pytest.raises(LinearityError):
        detect_linear_range(np.array([0.0]), np.array([1.0]))


def test_noise_estimate_ignores_the_trend():
    rng = np.random.default_rng(2)
    time = np.arange(0, 200) * MINUTE
    assert estimate_noise(3 * time / MINUTE + rng.normal(0, 10, 200)) == \
        pytest.approx(10, rel=0.25)


def test_fit_line_matches_numpy():
    x = np.arange(10.0)
    y = 3 * x + 7
    slope, intercept, r2, _ = fit_line(x, y)
    assert (slope, intercept) == pytest.approx((3, 7))
    assert r2 == pytest.approx(1.0)
