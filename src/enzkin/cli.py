"""Command line interface: ``enzkin ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import BLANK_MODES, analyse
from .export import write_outputs
from .linearity import METHODS, DetectionSettings
from .platemap import PlateMap, PlateMapError, read_map
from .plates import PlateFormat
from .readers import ReaderError, read_kinetics
from .templates import blank_long_map, starter_layout


def _add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("linear range")
    group.add_argument("--method", choices=METHODS, default="auto",
                       help="how to choose the linear range (default: auto)")
    group.add_argument("--min-points", type=int, default=None,
                       help="fewest readings a fitted window may contain")
    group.add_argument("--tolerance", type=float, default=1.25,
                       metavar="X", dest="residual_tolerance",
                       help="scatter allowed, as a multiple of the well's own "
                            "noise (default: 1.25)")
    group.add_argument("--r2-min", type=float, default=None,
                       help="also require this R^2 (off by default; straightness "
                            "is judged against the noise instead)")
    group.add_argument("--window-points", type=int, default=10,
                       help="window width for --method max_slope / initial")
    group.add_argument("--fixed-start", type=float, default=None, metavar="MIN",
                       help="start of the window for --method fixed, in minutes")
    group.add_argument("--fixed-end", type=float, default=None, metavar="MIN",
                       help="end of the window for --method fixed, in minutes")
    group.add_argument("--from", type=float, default=None, metavar="MIN",
                       dest="search_start",
                       help="ignore readings before this time (minutes)")
    group.add_argument("--until", type=float, default=None, metavar="MIN",
                       dest="search_end",
                       help="ignore readings after this time (minutes)")
    group.add_argument("--direction", choices=["auto", "increasing", "decreasing"],
                       default="auto",
                       help="whether signal rises or falls (default: auto)")
    group.add_argument("--smooth", type=int, default=0, dest="smooth_points",
                       help="smooth over N readings when detecting (not when fitting)")


def _settings_from(args) -> DetectionSettings:
    minute = 60.0
    return DetectionSettings(
        method=args.method,
        min_points=args.min_points,
        residual_tolerance=args.residual_tolerance,
        r2_min=args.r2_min,
        window_points=args.window_points,
        fixed_start=None if args.fixed_start is None else args.fixed_start * minute,
        fixed_end=None if args.fixed_end is None else args.fixed_end * minute,
        search_start=None if args.search_start is None else args.search_start * minute,
        search_end=None if args.search_end is None else args.search_end * minute,
        direction=args.direction,
        smooth_points=args.smooth_points,
    )


def _parse_units(pairs) -> dict[str, str]:
    units: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--unit expects factor=unit, got {item!r}")
        factor, unit = item.split("=", 1)
        units[factor.strip()] = unit.strip()
    return units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzkin",
        description="Enzyme kinetics from plate-reader time courses: find each "
                    "well's linear range, fit its slope, average technical "
                    "replicates, and write Prism-ready tables.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "analyse", aliases=["analyze", "run"],
        help="analyse a kinetic file against a plate map")
    run.add_argument("data", help="kinetic export (CSV/TSV/XLSX)")
    run.add_argument("-m", "--map", required=True,
                     help="plate map: layout YAML/JSON, or a long CSV/XLSX")
    run.add_argument("-o", "--out", default="results",
                     help="output folder (default: results)")
    run.add_argument("--plate", default=None,
                     help="plate format, e.g. 96 or 384 (default: inferred)")
    run.add_argument("--time-unit", default="s",
                     help="unit of a numeric time column (default: s; clock "
                          "times like 0:01:30 are read as written)")
    run.add_argument("--rate-unit", default="min",
                     help="report rates per this unit (default: min)")
    run.add_argument("--signal", default="RFU",
                     help="what the reader measured (default: RFU)")
    run.add_argument("--blank", choices=list(BLANK_MODES), default="slope",
                     help="how to use blank wells (default: slope)")
    run.add_argument("--unit", action="append", metavar="FACTOR=UNIT",
                     help="display unit for a factor, e.g. --unit compound=nM")
    run.add_argument("--sheet", default=None,
                     help="worksheet name or index, for Excel input")
    run.add_argument("--orientation", choices=["auto", "wide", "tall"],
                     default="auto",
                     help="wells in columns (wide) or rows (tall)")
    run.add_argument("--theme", choices=["light", "dark"], default="light")
    run.add_argument("--plate-scale", choices=["free", "shared", "row", "column"],
                     default="free",
                     help="y-axis scaling in the plate figure (default: free)")
    run.add_argument("--no-figures", action="store_true", help="skip the figures")
    run.add_argument("--no-excel", action="store_true", help="skip the workbook")
    run.add_argument("--quiet", action="store_true")
    _add_detection_arguments(run)

    check = subparsers.add_parser(
        "check", help="report what is in a kinetic file, without analysing it")
    check.add_argument("data")
    check.add_argument("--time-unit", default="s")
    check.add_argument("--sheet", default=None)

    layout = subparsers.add_parser(
        "new-layout", help="write a starter layout file to fill in")
    layout.add_argument("-o", "--out", default="layout.yaml")
    layout.add_argument("--plate", default="96")

    table = subparsers.add_parser(
        "new-map", help="write a blank one-row-per-well plate map for Excel")
    table.add_argument("-o", "--out", default="plate_map.csv")
    table.add_argument("--plate", default="96")

    app = subparsers.add_parser("app", help="open the point-and-click app")
    app.add_argument("--port", type=int, default=8501)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "app":
        return _launch_app(args.port)

    if args.command == "new-layout":
        path = Path(args.out)
        path.write_text(starter_layout(args.plate), encoding="utf-8")
        print(f"Wrote {path}")
        print("Edit it to describe your plate, then run:")
        print(f"  enzkin analyse <data.csv> --map {path} --out results")
        return 0

    if args.command == "new-map":
        path = Path(args.out)
        blank_long_map(PlateFormat.of(args.plate)).to_csv(path, index=False)
        print(f"Wrote {path}")
        print("Fill it in (one row per well; concentrations may be written "
              "'50 uM', '3 nM', ...), then run:")
        print(f"  enzkin analyse <data.csv> --map {path} --out results")
        return 0

    if args.command == "check":
        try:
            data = read_kinetics(args.data, time_unit=args.time_unit,
                                 sheet=args.sheet)
        except (ReaderError, OSError) as exc:
            print(f"Could not read {args.data}: {exc}", file=sys.stderr)
            return 2
        print(data.summary())
        print(f"Time: {data.time[0]:.0f} s to {data.time[-1]:.0f} s "
              f"({data.time[1] - data.time[0]:.0f} s between readings)")
        print(f"Wells: {', '.join(data.wells[:12])}"
              f"{' ...' if data.n_wells > 12 else ''}")
        for note in data.notes:
            print(f"  note: {note}")
        return 0

    # -- analyse -----------------------------------------------------------
    try:
        data = read_kinetics(args.data, time_unit=args.time_unit,
                             sheet=args.sheet, plate=args.plate,
                             orientation=args.orientation)
    except (ReaderError, OSError) as exc:
        print(f"Could not read {args.data}: {exc}", file=sys.stderr)
        return 2
    try:
        plate_map: PlateMap = read_map(args.map, plate=args.plate)
    except (PlateMapError, OSError, ValueError) as exc:
        print(f"Could not read the plate map {args.map}: {exc}", file=sys.stderr)
        return 2

    problems = plate_map.validate()
    for problem in problems:
        print(f"  plate map: {problem}", file=sys.stderr)

    try:
        result = analyse(
            data, plate_map, _settings_from(args), blank_mode=args.blank,
            rate_unit=args.rate_unit, signal_label=args.signal,
            units=_parse_units(args.unit))
    except ValueError as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 2

    written = write_outputs(
        result, args.out, figures=not args.no_figures, theme=args.theme,
        plate_scale=args.plate_scale, excel=not args.no_excel)

    if not args.quiet:
        print(result.plate_map.summary())
        print(result.summary())
        flagged = result.flagged_wells()
        if flagged:
            print(f"{len(flagged)} well(s) flagged for review: "
                  f"{', '.join(flagged[:12])}"
                  f"{' ...' if len(flagged) > 12 else ''}")
        print(f"\nWrote {len(written)} file(s) to {Path(args.out).resolve()}")
        print("  condition_means.csv   replicate means, SD, SEM, CV")
        print("  prism/                paste-ready XY tables")
        print("  figures/plate_curves.png   every curve with its fitted range")
        print("  analysis_report.txt   what was done, and what to look at")
    return 0


def _launch_app(port: int) -> int:
    app_path = Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    if not app_path.exists():  # installed as a package
        app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("The app needs Streamlit:  pip install streamlit", file=sys.stderr)
        return 2
    sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    return stcli.main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
