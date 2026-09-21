"""Starter files: a layout to edit, or a blank table to fill in Excel."""

from __future__ import annotations

import pandas as pd

from .plates import PlateFormat, row_label

_LAYOUT = '''\
# ---------------------------------------------------------------------------
# Plate layout.  Blocks are painted onto the plate in order, so a later block
# overrides an earlier one wherever they overlap.
#
# Concentrations carry their own units - write "50 uM" here and "3 nM" there.
# Nothing needs converting by hand.
#
#   Well ranges:  A1:F12   a rectangle        A1,B4,C7   individual wells
#                 A        a whole row        3          a whole column
#                 A:F      rows A to F        5:8        columns 5 to 8
#
#   Roles:        sample             a normal reaction (the default)
#                 positive_control   no inhibitor - the 100% activity reference
#                 negative_control   no enzyme, or fully inhibited
#                 blank              background; can be subtracted
#                 empty              nothing in the well - ignored
# ---------------------------------------------------------------------------
name: My experiment
plate: {size}

# What each factor is called in tables and figures.
labels:
  protein: Enzyme
  compound: Test compound
  substrate: Substrate

# Units for displaying results.  Delete a line to let the tool choose.
units:
  protein: nM
  compound: uM
  substrate: uM

# Applied to every block unless the block overrides it.
defaults:
  protein:
    name: My enzyme
    conc: 5 nM

blocks:
  - name: Compound dose-response
    wells: {sample_range}
    compound:
      name: Compound-1
      series:
        across: columns      # or: rows
        replicates: 2        # adjacent columns sharing one concentration
        top: 10 uM           # highest concentration, at the low-numbered end
        dilution: 3          # 3-fold serial dilution
        # reverse: true      # uncomment if column 1 is the *lowest*
        # values: [10 uM, 1 uM, 0.1 uM]   # or list them outright
    substrate:
      name: My substrate
      series:
        across: rows
        top: 50 uM
        dilution: 2

  - name: No-compound controls
    wells: {control_range}
    role: positive_control
    compound:
      name: DMSO
      conc: 0
    substrate:
      name: My substrate
      conc: 50 uM

  - name: Background
    wells: {blank_range}
    role: blank
    substrate:
      name: My substrate
      conc: 0

  - name: Unused
    wells: {empty_range}
    role: empty
'''


def starter_layout(plate: int | str = 96) -> str:
    """A commented layout file, sized for the requested plate."""
    fmt = PlateFormat.of(plate)
    last_row = row_label(fmt.n_rows - 1)
    control_row = row_label(max(fmt.n_rows - 2, 0))
    sample_last = row_label(max(fmt.n_rows - 3, 0))
    return _LAYOUT.format(
        size=fmt.size,
        sample_range=f"A1:{sample_last}{fmt.n_cols}",
        control_range=f"{control_row}1:{control_row}{fmt.n_cols}",
        blank_range=f"{last_row}1:{last_row}2",
        empty_range=f"{last_row}3:{last_row}{fmt.n_cols}",
    )


def blank_long_map(plate: PlateFormat | int | str = 96) -> pd.DataFrame:
    """An empty one-row-per-well plate map, ready to fill in a spreadsheet."""
    fmt = PlateFormat.of(plate)
    return pd.DataFrame({
        "well": fmt.wells(),
        "protein": "",
        "protein_conc": "",
        "compound": "",
        "compound_conc": "",
        "substrate": "",
        "substrate_conc": "",
        "role": "",
        "exclude": "",
        "notes": "",
    })
