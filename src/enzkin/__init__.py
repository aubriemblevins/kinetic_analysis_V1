"""enzkin - enzyme kinetics from plate-reader time courses.

Typical use::

    from enzkin import read_kinetics, read_map, analyse, write_outputs

    data = read_kinetics("plate1.csv")
    layout = read_map("layout.yaml")
    result = analyse(data, layout)
    write_outputs(result, "results/")

``result.condition_table()`` is the replicate-averaged table; everything in
``results/prism/`` is laid out to paste straight into GraphPad Prism.
"""

from .analysis import AnalysisResult, ConditionResult, WellResult, analyse
from .export import prism_table, prism_tables, write_outputs
from .linearity import (DetectionSettings, LinearFit, detect_linear_range,
                        fit_line)
from .platemap import (Factor, PlateMap, WellInfo, map_from_layout, read_map)
from .plates import PlateFormat, expand_wells
from .readers import KineticData, read_kinetics
from .templates import blank_long_map, starter_layout
from .units import Quantity, parse_quantity

analyze = analyse  # both spellings work

__version__ = "1.0.0"

__all__ = [
    "AnalysisResult", "ConditionResult", "DetectionSettings", "Factor",
    "KineticData", "LinearFit", "PlateFormat", "PlateMap", "Quantity",
    "WellInfo", "WellResult", "analyse", "analyze", "blank_long_map",
    "detect_linear_range", "expand_wells", "fit_line", "map_from_layout",
    "parse_quantity", "prism_table", "prism_tables", "read_kinetics",
    "read_map", "starter_layout", "write_outputs", "__version__",
]
