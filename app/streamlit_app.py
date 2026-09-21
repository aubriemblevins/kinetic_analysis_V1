"""Point-and-click front end for enzkin.

Upload a kinetic export, describe the plate once, and get replicate-averaged
slopes plus Prism-ready tables - without dragging anything in a spreadsheet.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enzkin import plotting                                    # noqa: E402
from enzkin.analysis import BLANK_MODES, analyse               # noqa: E402
from enzkin.export import prism_tables, write_outputs          # noqa: E402
from enzkin.linearity import (FLAG_DESCRIPTIONS,               # noqa: E402
                              DetectionSettings)
from enzkin.platemap import (CORE_FACTORS, DEFAULT_LABELS,     # noqa: E402
                             PlateMapError, map_from_layout, read_map)
from enzkin.plates import PLATE_FORMATS, PlateFormat, expand_wells  # noqa: E402
from enzkin.readers import ReaderError, read_kinetics          # noqa: E402
from enzkin.templates import starter_layout                    # noqa: E402
from enzkin.units import known_units                           # noqa: E402

EXAMPLE_DATA = Path(__file__).resolve().parent.parent / "examples" / "Example_data.csv"
EXAMPLE_LAYOUT = (Path(__file__).resolve().parent.parent / "examples"
                  / "example_layout.yaml")

ROLE_CHOICES = ["sample", "positive_control", "negative_control", "blank", "empty"]
ROLE_HELP = {
    "sample": "A normal reaction well.",
    "positive_control": "No inhibitor - the 100% activity reference.",
    "negative_control": "No enzyme, or fully inhibited.",
    "blank": "Background; can be subtracted from the samples.",
    "empty": "Nothing in the well - ignored entirely.",
}
ROLE_TINT = {"sample": "#e8f1fd", "positive_control": "#e3f6ee",
             "negative_control": "#fbe7e7", "blank": "#fdf3de",
             "empty": "#f2f1ed"}

st.set_page_config(page_title="Enzyme kinetics", page_icon="🧪", layout="wide")


# --- state ----------------------------------------------------------------

def _state():
    st.session_state.setdefault("blocks", [])
    st.session_state.setdefault("overrides", {})
    st.session_state.setdefault("block_counter", 0)
    return st.session_state


def _new_block(plate: PlateFormat) -> dict:
    state = _state()
    state["block_counter"] += 1
    return {
        "id": state["block_counter"],
        "name": f"Block {state['block_counter']}",
        "wells": f"A1:{plate.rows[min(5, plate.n_rows - 1)]}{plate.n_cols}",
        "role": "sample",
        "factors": {
            name: {"use": name != "protein", "name": "", "mode": "constant",
                   "conc": "", "unit": "uM", "across": "columns", "replicates": 2,
                   "top": "10 uM", "dilution": 3.0, "reverse": False,
                   "values": ""}
            for name in CORE_FACTORS
        },
    }


def _block_to_layout(block: dict) -> dict:
    entry: dict = {"name": block["name"], "wells": block["wells"],
                   "role": block["role"]}
    if block["role"] == "empty":
        return entry
    for factor, spec in block["factors"].items():
        if not spec.get("use"):
            continue
        item: dict = {}
        if spec.get("name"):
            item["name"] = spec["name"]
        if spec["mode"] == "constant":
            if str(spec.get("conc", "")).strip():
                item["conc"] = spec["conc"]
        elif spec["mode"] == "list":
            values = [v.strip() for v in str(spec.get("values", "")).split(",")
                      if v.strip()]
            if values:
                item["series"] = {"across": spec["across"],
                                  "replicates": int(spec["replicates"]),
                                  "values": values,
                                  "reverse": bool(spec["reverse"])}
        else:
            item["series"] = {"across": spec["across"],
                              "replicates": int(spec["replicates"]),
                              "top": spec["top"],
                              "dilution": float(spec["dilution"]),
                              "reverse": bool(spec["reverse"])}
        if item:
            entry[factor] = item
    return entry


def _layout_from_state(plate: PlateFormat, labels, units) -> dict:
    return {
        "name": st.session_state.get("experiment_name", "My experiment"),
        "plate": plate.size,
        "labels": labels,
        "units": units,
        "blocks": [_block_to_layout(b) for b in _state()["blocks"]],
    }


# --- sidebar --------------------------------------------------------------

def sidebar():
    st.sidebar.title("🧪 Enzyme kinetics")
    st.sidebar.caption("Raw curves → linear range → slopes → replicate means → Prism")

    st.sidebar.subheader("1 · Data")
    upload = st.sidebar.file_uploader(
        "Kinetic export", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
        help="A Time column plus one column per well (A1, A2, ...). "
             "Wells in rows work too.")
    use_example = st.sidebar.checkbox("Use the example plate", value=upload is None)

    plate_choice = st.sidebar.selectbox(
        "Plate format", ["auto"] + [str(k) for k in sorted(PLATE_FORMATS)],
        index=0)
    time_unit = st.sidebar.selectbox(
        "Unit of a numeric time column", ["s", "min", "h", "ms"], index=0,
        help="Clock times such as 0:01:30 are read as written; this only "
             "applies when the time column is a plain number.")

    st.sidebar.subheader("2 · Linear range")
    method = st.sidebar.selectbox(
        "How to choose it",
        ["auto", "fixed", "max_slope", "initial", "all"],
        format_func=lambda m: {
            "auto": "Automatic (recommended)",
            "fixed": "A fixed time window",
            "max_slope": "Steepest window of N readings",
            "initial": "First N readings",
            "all": "The whole curve",
        }[m])
    settings_kwargs: dict = {"method": method}
    if method == "auto":
        settings_kwargs["residual_tolerance"] = st.sidebar.slider(
            "Scatter allowed (× the well's noise)", 0.75, 3.0, 1.25, 0.05,
            help="Higher keeps longer windows; lower trims at the first hint "
                 "of curvature.")
        settings_kwargs["min_points"] = st.sidebar.number_input(
            "Fewest readings in a window", min_value=0, max_value=500, value=0,
            help="0 lets the tool choose.") or None
    elif method == "fixed":
        span = st.sidebar.slider("Window (minutes)", 0.0, 600.0, (0.0, 30.0), 0.5)
        settings_kwargs["fixed_start"] = span[0] * 60
        settings_kwargs["fixed_end"] = span[1] * 60
    elif method in {"max_slope", "initial"}:
        settings_kwargs["window_points"] = st.sidebar.number_input(
            "Readings per window", 3, 500, 10)

    with st.sidebar.expander("More detection options"):
        limit = st.checkbox("Ignore part of the run")
        if limit:
            bounds = st.slider("Use only this stretch (minutes)", 0.0, 600.0,
                               (0.0, 600.0), 0.5)
            settings_kwargs["search_start"] = bounds[0] * 60
            settings_kwargs["search_end"] = bounds[1] * 60
        settings_kwargs["direction"] = st.selectbox(
            "Signal direction", ["auto", "increasing", "decreasing"],
            help="Decreasing suits absorbance assays such as NADH consumption.")
        settings_kwargs["smooth_points"] = st.number_input(
            "Smooth over N readings when detecting", 0, 25, 0,
            help="Detection only - the slope is always fitted to the raw data.")

    st.sidebar.subheader("3 · Reporting")
    blank_mode = st.sidebar.selectbox(
        "Background wells", list(BLANK_MODES), index=1,
        format_func=lambda m: {"none": "Ignore them",
                               "slope": "Subtract their rate",
                               "trace": "Subtract their readings"}[m])
    rate_unit = st.sidebar.selectbox("Rate per", ["min", "s", "h"], index=0)
    signal = st.sidebar.text_input("Signal name", "RFU")
    theme = st.sidebar.selectbox("Figure theme", ["light", "dark"], index=0)

    return dict(upload=upload, use_example=use_example, plate_choice=plate_choice,
                time_unit=time_unit, settings=DetectionSettings(**settings_kwargs),
                blank_mode=blank_mode, rate_unit=rate_unit, signal=signal,
                theme=theme)


@st.cache_data(show_spinner=False)
def _load(raw: bytes, name: str, time_unit: str, plate: str | None):
    buffer = io.BytesIO(raw)
    buffer.name = name
    return read_kinetics(buffer, time_unit=time_unit,
                         plate=None if plate in (None, "auto") else plate)


# --- plate map preview ----------------------------------------------------

def plate_preview(plate_map, colour_by: str):
    """An HTML plate grid; hover a well to see everything in it."""
    plate = plate_map.plate
    units = plate_map.auto_units()
    levels = plate_map.levels(colour_by) if colour_by != "role" else []
    order = {round(q.canonical, 12): i for i, q in enumerate(levels)}

    def cell_colour(info):
        if colour_by == "role" or not levels:
            return ROLE_TINT.get(info.role, "#f2f1ed")
        if info.role == "empty":
            return "#f2f1ed"
        conc = info.factor(colour_by).conc
        if conc is None:
            return "#f2f1ed"
        ramp = plotting.LIGHT.sequential
        position = order.get(round(conc.canonical, 12), 0)
        index = int(round(position / max(len(levels) - 1, 1) * (len(ramp) - 1)))
        return ramp[index]

    head = "".join(f"<th>{c}</th>" for c in plate.columns)
    rows = []
    for row in plate.rows:
        cells = []
        for col in plate.columns:
            well = f"{row}{col}"
            info = plate_map.wells.get(well)
            if info is None:
                cells.append('<td class="ek-empty"></td>')
                continue
            tooltip = info.condition_label(plate_map.active_factors(), units,
                                           plate_map.labels).replace('"', "'")
            dark = colour_by != "role" and cell_colour(info) in {
                "#1c5cab", "#184f95", "#104281", "#0d366b", "#256abf"}
            cells.append(
                f'<td style="background:{cell_colour(info)};'
                f'color:{"#fff" if dark else "#0b0b0b"}" '
                f'title="{well} — {tooltip}">{well}</td>')
        rows.append(f"<tr><th>{row}</th>{''.join(cells)}</tr>")

    st.markdown(f"""
<style>
.ek-plate {{border-collapse:separate;border-spacing:2px;font-size:11px;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
.ek-plate td {{width:46px;height:26px;text-align:center;border-radius:4px}}
.ek-plate th {{color:#898781;font-weight:600;padding:0 4px}}
.ek-plate td.ek-empty {{background:#f2f1ed}}
</style>
<table class="ek-plate"><tr><th></th>{head}</tr>{''.join(rows)}</table>
""", unsafe_allow_html=True)
    st.caption("Hover a well to see what is in it.")


# --- the plate-map tab ----------------------------------------------------

def map_tab(data, config):
    plate = (PlateFormat.of(config["plate_choice"])
             if config["plate_choice"] != "auto" else data.plate)
    st.subheader("Describe the plate")
    source = st.radio(
        "Where does the layout come from?",
        ["Build it here", "Upload a map file", "Use the example layout"],
        horizontal=True, label_visibility="collapsed")

    if source == "Upload a map file":
        uploaded = st.file_uploader(
            "Layout YAML/JSON, or a one-row-per-well CSV/XLSX",
            type=["yaml", "yml", "json", "csv", "tsv", "xlsx"])
        if uploaded is None:
            st.info("Upload a plate map, or switch to **Build it here**.")
            return None
        try:
            return read_map(uploaded, plate=plate.size)
        except (PlateMapError, ValueError) as exc:
            st.error(f"Could not read that map: {exc}")
            return None

    if source == "Use the example layout":
        st.caption(f"Reading `{EXAMPLE_LAYOUT.name}` - "
                   "an inhibitor dose-response crossed with a substrate titration.")
        return read_map(EXAMPLE_LAYOUT, plate=plate.size)

    return build_map(plate)


def build_map(plate: PlateFormat):
    state = _state()
    st.text_input("Experiment name", key="experiment_name",
                  value=st.session_state.get("experiment_name", "My experiment"))

    with st.expander("Names and units", expanded=False):
        columns = st.columns(3)
        labels, units = {}, {}
        for index, factor in enumerate(CORE_FACTORS):
            with columns[index]:
                labels[factor] = st.text_input(
                    f"Call “{factor}”", DEFAULT_LABELS[factor], key=f"label_{factor}")
                units[factor] = st.selectbox(
                    "shown in", known_units("molar") + known_units("mass_conc"),
                    index=3 if factor == "protein" else 2, key=f"unit_{factor}")

    if not state["blocks"]:
        state["blocks"].append(_new_block(plate))

    remove = None
    for position, block in enumerate(state["blocks"]):
        with st.expander(f"**{block['name']}** — {block['wells']} "
                         f"({block['role'].replace('_', ' ')})",
                         expanded=len(state["blocks"]) <= 2):
            top = st.columns([3, 3, 3, 1])
            block["name"] = top[0].text_input(
                "Name", block["name"], key=f"n{block['id']}")
            block["wells"] = top[1].text_input(
                "Wells", block["wells"], key=f"w{block['id']}",
                help="A1:F12 · A1,B2 · A · A:F · 3 · 5:8 · all")
            block["role"] = top[2].selectbox(
                "Role", ROLE_CHOICES, index=ROLE_CHOICES.index(block["role"]),
                key=f"r{block['id']}",
                help="\n".join(f"{k}: {v}" for k, v in ROLE_HELP.items()))
            if top[3].button("🗑", key=f"d{block['id']}", help="Remove this block"):
                remove = position

            try:
                wells = expand_wells(block["wells"], plate)
                st.caption(f"{len(wells)} well(s): {', '.join(wells[:8])}"
                           f"{' …' if len(wells) > 8 else ''}")
            except Exception as exc:
                st.error(f"Well range: {exc}")

            if block["role"] != "empty":
                tabs = st.tabs([DEFAULT_LABELS[f] for f in CORE_FACTORS])
                for tab, factor in zip(tabs, CORE_FACTORS):
                    with tab:
                        _factor_editor(block, factor)

    if remove is not None:
        state["blocks"].pop(remove)
        st.rerun()

    buttons = st.columns([1, 1, 4])
    if buttons[0].button("➕ Add block"):
        state["blocks"].append(_new_block(plate))
        st.rerun()
    if buttons[1].button("↺ Start over"):
        state["blocks"] = []
        st.rerun()

    layout = _layout_from_state(plate, labels, units)
    try:
        return map_from_layout(layout)
    except (PlateMapError, ValueError) as exc:
        st.error(f"That layout does not resolve: {exc}")
        return None


def _factor_editor(block: dict, factor: str) -> None:
    spec = block["factors"][factor]
    key = f"{block['id']}_{factor}"
    spec["use"] = st.checkbox("This well contains it", spec["use"], key=f"u{key}")
    if not spec["use"]:
        return
    columns = st.columns([2, 2])
    spec["name"] = columns[0].text_input("Identity", spec["name"], key=f"i{key}",
                                         placeholder="e.g. Compound-1")
    spec["mode"] = columns[1].selectbox(
        "Concentration", ["constant", "series", "list"],
        index=["constant", "series", "list"].index(spec["mode"]), key=f"m{key}",
        format_func=lambda m: {"constant": "The same everywhere",
                               "series": "Serial dilution",
                               "list": "A list of values"}[m])

    if spec["mode"] == "constant":
        spec["conc"] = st.text_input(
            "Concentration", spec["conc"], key=f"c{key}",
            placeholder="50 uM · 3 nM · 0 · 2 mg/mL",
            help="Write the unit with the number - no conversions needed.")
        return

    row = st.columns([2, 2, 2])
    spec["across"] = row[0].selectbox(
        "Varies across", ["columns", "rows"],
        index=0 if spec["across"] == "columns" else 1, key=f"a{key}")
    spec["replicates"] = row[1].number_input(
        "Wells per concentration", 1, 24, int(spec["replicates"]), key=f"p{key}",
        help="2 means adjacent column pairs (or row pairs) share a concentration.")
    spec["reverse"] = row[2].checkbox(
        "Lowest first", spec["reverse"], key=f"v{key}",
        help="Tick when column 1 (or row A) holds the *lowest* concentration.")

    if spec["mode"] == "series":
        pair = st.columns(2)
        spec["top"] = pair[0].text_input("Highest concentration", spec["top"],
                                         key=f"t{key}", placeholder="10 uM")
        spec["dilution"] = pair[1].number_input(
            "Dilution factor", 1.0001, 1000.0, float(spec["dilution"]),
            key=f"f{key}", help="3 means a 3-fold serial dilution.")
    else:
        spec["values"] = st.text_input(
            "Values, highest first", spec["values"], key=f"l{key}",
            placeholder="10 uM, 3 uM, 1 uM, 0.3 uM",
            help="One per concentration level, separated by commas.")


# --- results --------------------------------------------------------------

def results_tab(result, config):
    left, right = st.columns([2, 3])
    with left:
        st.metric("Conditions", len(result.conditions))
        st.metric("Wells analysed", len(result.wells))
        worst = max((c.cv for c in result.conditions
                     if c.n > 1 and c.role == "sample" and c.cv == c.cv),
                    default=float("nan"))
        st.metric("Worst replicate CV", f"{worst:.1f}%" if worst == worst else "-")
    with right:
        st.info(result.summary())

    st.subheader("Replicate means")
    st.caption("One row per condition - this is the table to paste into Prism "
               "if you only want the averages.")
    st.dataframe(result.condition_table(), use_container_width=True, height=330)

    factors = [f for f in result.factors()
               if len(result.plate_map.levels(f)) > 1]
    if factors:
        st.subheader("Rates against concentration")
        columns = st.columns([1, 1, 1])
        x_factor = columns[0].selectbox(
            "X axis", factors, index=len(factors) - 1,
            format_func=result.plate_map.label_for)
        others = [f for f in factors if f != x_factor]
        series = columns[1].selectbox(
            "One line per", ["(none)"] + others,
            index=1 if others else 0,
            format_func=lambda f: ("(none)" if f == "(none)"
                                   else result.plate_map.label_for(f)))
        value = columns[2].selectbox(
            "Y axis", ["rate", "percent_activity", "percent_inhibition"],
            format_func=lambda v: {"rate": f"Rate ({result.rate_label})",
                                   "percent_activity": "% of control",
                                   "percent_inhibition": "% inhibition"}[v])
        try:
            figure = plotting.plot_rate_vs_concentration(
                result, x_factor, None if series == "(none)" else series,
                value=value, theme=config["theme"])
            st.pyplot(figure, use_container_width=False)
        except ValueError as exc:
            st.warning(str(exc))

    st.subheader("Take it away")
    st.caption("The Prism folder holds the same numbers in every layout Prism "
               "expects - means, replicate subcolumns, normalised, log X.")
    tables = {"condition_means.csv": result.condition_table(),
              "well_results.csv": result.well_table(),
              **{f"prism/{name}": table for name, table in prism_tables(result).items()}}
    chosen = st.selectbox("Preview a file", list(tables))
    st.dataframe(tables[chosen], use_container_width=True, height=240)
    columns = st.columns(2)
    columns[0].download_button(
        f"⬇ {Path(chosen).name}", tables[chosen].to_csv(index=False).encode(),
        file_name=Path(chosen).name, mime="text/csv", use_container_width=True)
    columns[1].download_button(
        "⬇ Everything (zip)", _zip_everything(result, config),
        file_name="kinetics_results.zip", mime="application/zip",
        use_container_width=True, type="primary")


def _zip_everything(result, config) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        written = write_outputs(result, folder, theme=config["theme"])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in written:
                archive.write(path, Path(path).relative_to(folder))
        return buffer.getvalue()


def curves_tab(result, config):
    st.subheader("Every curve, with the range that was fitted")
    columns = st.columns([1, 1, 2])
    scale = columns[0].selectbox(
        "Y scaling", ["free", "shared", "row", "column"],
        format_func=lambda s: {"free": "Each well autoscaled",
                               "shared": "One scale for the plate",
                               "row": "One scale per row",
                               "column": "One scale per column"}[s])
    if columns[1].button("Redraw", use_container_width=True):
        st.cache_data.clear()
    flagged = result.flagged_wells()
    if flagged:
        columns[2].warning(f"{len(flagged)} well(s) flagged: "
                           f"{', '.join(flagged[:10])}"
                           f"{' …' if len(flagged) > 10 else ''}")
    st.pyplot(plotting.plot_plate_curves(result, scale=scale,
                                         theme=config["theme"]))

    st.subheader("Look at one well")
    wells = list(result.wells)
    choice = st.selectbox("Well", wells,
                          index=wells.index(flagged[0]) if flagged else 0)
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plotting.plot_well(result, choice, theme=config["theme"]))
    with right:
        outcome = result.wells[choice]
        if outcome.fit is not None:
            fit = outcome.fit
            st.markdown(
                f"**Rate** {outcome.rate:,.4g} {result.rate_label}  \n"
                f"**R²** {fit.r2:.4f}  ·  **points** {fit.n_points}  \n"
                f"**Window** {fit.start_time / 60:.1f}–{fit.end_time / 60:.1f} min  \n"
                f"**Signal / noise** {fit.snr:,.0f}")
        for flag in outcome.flags:
            st.caption(f"`{flag}` — {FLAG_DESCRIPTIONS.get(flag, '')}")

        st.markdown("**Override this well's window**")
        override = st.session_state["overrides"].get(choice)
        limits = st.slider(
            "Minutes", 0.0, float(result.data.time[-1] / 60),
            (float(outcome.fit.start_time / 60) if outcome.fit else 0.0,
             float(outcome.fit.end_time / 60) if outcome.fit else 1.0),
            0.5, key=f"ov_{choice}")
        buttons = st.columns(2)
        if buttons[0].button("Apply", key=f"apply_{choice}",
                             use_container_width=True):
            st.session_state["overrides"][choice] = DetectionSettings(
                method="fixed", fixed_start=limits[0] * 60, fixed_end=limits[1] * 60)
            st.rerun()
        if override and buttons[1].button("Clear", key=f"clear_{choice}",
                                          use_container_width=True):
            st.session_state["overrides"].pop(choice, None)
            st.rerun()
    if st.session_state["overrides"]:
        st.caption("Hand-picked windows: "
                   + ", ".join(sorted(st.session_state["overrides"])))


# --- main -----------------------------------------------------------------

def main():
    _state()
    config = sidebar()

    if config["upload"] is not None:
        raw, name = config["upload"].getvalue(), config["upload"].name
    elif config["use_example"] and EXAMPLE_DATA.exists():
        raw, name = EXAMPLE_DATA.read_bytes(), EXAMPLE_DATA.name
    else:
        st.title("Enzyme kinetics")
        st.markdown(
            "Upload a plate-reader export in the sidebar, or tick "
            "**Use the example plate** to see how it works.\n\n"
            "The file needs a **Time** column and one column per well "
            "(`A1`, `A2`, …).  Times may be `0:01:30` or plain numbers; "
            "wells in rows instead of columns are detected automatically.")
        st.stop()

    try:
        data = _load(raw, name, config["time_unit"], config["plate_choice"])
    except (ReaderError, ValueError) as exc:
        st.error(f"Could not read that file: {exc}")
        st.stop()

    st.title("Enzyme kinetics")
    st.caption(f"**{name}** — {data.summary()}")
    for note in data.notes:
        st.caption(f"ℹ️ {note}")

    tab_map, tab_curves, tab_results, tab_data = st.tabs(
        ["1 · Plate map", "2 · Curves & fits", "3 · Results", "Raw data"])

    with tab_map:
        plate_map = map_tab(data, config)
        if plate_map is not None:
            st.divider()
            colour_by = st.selectbox(
                "Colour the plate by",
                ["role"] + plate_map.active_factors(),
                format_func=lambda f: ("Role" if f == "role"
                                       else plate_map.label_for(f)))
            plate_preview(plate_map, colour_by)
            for problem in plate_map.validate():
                st.warning(problem)
            st.download_button(
                "⬇ Save this layout (YAML)",
                _layout_yaml(plate_map),
                file_name="layout.yaml", mime="text/yaml")

    if plate_map is None:
        st.stop()

    try:
        result = analyse(
            data, plate_map, config["settings"], blank_mode=config["blank_mode"],
            rate_unit=config["rate_unit"], signal_label=config["signal"],
            well_settings=st.session_state["overrides"])
    except ValueError as exc:
        with tab_results:
            st.error(str(exc))
        st.stop()

    with tab_curves:
        curves_tab(result, config)
    with tab_results:
        results_tab(result, config)
    with tab_data:
        st.dataframe(data.to_frame(), use_container_width=True, height=440)


def _layout_yaml(plate_map) -> str:
    import yaml
    from enzkin.platemap import layout_from_map
    blocks = st.session_state.get("blocks")
    if blocks:
        layout = _layout_from_state(plate_map.plate, plate_map.labels,
                                    plate_map.auto_units())
    else:
        layout = layout_from_map(plate_map)
    return yaml.safe_dump(layout, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
