import matplotlib
import pytest

from enzkin import plotting

matplotlib.use("Agg")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_plate_figure_renders_in_both_themes(example_result, tmp_path, theme):
    path = plotting.plot_plate_curves(
        example_result, tmp_path / f"plate_{theme}.png", theme=theme)
    assert path.exists() and path.stat().st_size > 20_000


@pytest.mark.parametrize("scale", ["free", "shared", "row", "column"])
def test_every_scaling_mode_works(example_result, tmp_path, scale):
    path = plotting.plot_plate_curves(
        example_result, tmp_path / f"plate_{scale}.png", scale=scale)
    assert path.exists()


def test_single_well_figure(example_result, tmp_path):
    assert plotting.plot_well(example_result, "A5", tmp_path / "well.png").exists()


def test_dose_response_figure(example_result, tmp_path):
    path = plotting.plot_rate_vs_concentration(
        example_result, "compound", "substrate", value="percent_activity",
        path=tmp_path / "ic50.png")
    assert path.exists()


def test_michaelis_menten_figure(example_result, tmp_path):
    path = plotting.plot_rate_vs_concentration(
        example_result, "substrate", "compound", path=tmp_path / "mm.png")
    assert path.exists()


def test_heatmap(example_result, tmp_path):
    assert plotting.plot_rate_heatmap(example_result, tmp_path / "heat.png").exists()


def test_a_factor_with_no_concentrations_is_refused(example_result, tmp_path):
    with pytest.raises(ValueError):
        plotting.plot_rate_vs_concentration(example_result, "protein_missing")


def test_palette_is_the_validated_order():
    """Slot order is the colour-vision safety mechanism, not decoration."""
    assert plotting.LIGHT.series[:3] == ("#2a78d6", "#eb6834", "#1baf7a")
    assert len(plotting.LIGHT.series) == len(plotting.DARK.series) == 8
