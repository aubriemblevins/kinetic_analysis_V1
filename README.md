# enzkin — enzyme kinetics without the spreadsheet

Turn a plate-reader time course into replicate-averaged initial rates, ready to
paste into GraphPad Prism. No dragging, no manual range selection, no
converting every concentration to the same molarity first.

```
raw RFU curves  →  linear range  →  slope per well  →  replicate means  →  Prism
```

Works with 96- and 384-well plates (and 6, 12, 24, 48, 1536), any plate layout
you like, and any mix of concentration units — substrate in µM next to
inhibitor in nM is fine.

---

## Install

```bash
pip install -e .          # core: numpy, pandas, matplotlib, PyYAML
pip install -e ".[app]"   # adds the point-and-click app
```

## Two ways to use it

### The app (start here)

```bash
enzkin app
```

Opens in your browser. Upload your CSV, describe the plate once, look at every
fit, download everything. Tick **Use the example plate** to try it with no file
of your own.

### The command line (for repeat experiments)

```bash
enzkin analyse plate1.csv --map layout.yaml --out results/
```

Same analysis, scriptable, and the layout file is a record of how the plate was
set up.

---

## Your data file

One `Time` column and one column per well:

```
Time,A1,A2,A3,...,H12
0:01:01,1380,1390,1390,...,4
0:02:01,1467,1443,1459,...,4
```

The reader is deliberately forgiving:

| It handles | Examples |
|---|---|
| Any time format | `0:01:01`, `05:30`, `90`, `1.5 min`, `Time (min)` header |
| Instrument preamble | metadata lines above the header are skipped |
| Missing readings | `OVRFLW`, `Sat.`, blanks become gaps, not errors |
| Either orientation | wells in columns, or wells in rows |
| Excel and TSV | `.xlsx`, `.xlsm`, `.tsv`, `.txt` |
| A byte-order mark | the invisible character Excel adds |

Check what it found before analysing:

```bash
enzkin check plate1.csv
```

---

## Describing the plate

Every well can carry three identities — **protein**, **test compound**,
**substrate** — each with its own concentration in its own unit. Extra factors
(a cofactor, a buffer, a timepoint) can be added freely; they behave the same
way.

Wells that share all three identities *and* all three concentrations are
technical replicates, and get averaged together. Nothing else needs saying.

### Layout rules (recommended)

A handful of blocks painted onto the plate, later blocks overriding earlier
ones. `enzkin new-layout -o layout.yaml` writes a commented starter.

```yaml
plate: 96
labels: {protein: Enzyme, compound: Inhibitor, substrate: Substrate}
units:  {protein: nM, compound: uM, substrate: uM}

defaults:
  protein: {name: MMP-9, conc: 5 nM}

blocks:
  - name: Dose-response
    wells: A1:F12
    compound:
      name: Compound-1
      series: {across: columns, replicates: 2, top: 10 uM, dilution: 3}
    substrate:
      name: FRET-substrate
      series: {across: rows, top: 50 uM, dilution: 2}

  - name: No-inhibitor controls
    wells: G1:G12
    role: positive_control
    compound: {name: DMSO, conc: 0}
    substrate:
      name: FRET-substrate
      series: {across: columns, replicates: 2, top: 50 uM, dilution: 2}

  - {name: Background, wells: H1:H2, role: blank, substrate: {conc: 0}}
  - {name: Unused,     wells: H3:H12, role: empty}
```

That is the example plate: inhibitor titrated across duplicate column pairs,
substrate titrated down the rows, a row of no-inhibitor controls with its own
substrate titration running the *other* way, two background wells and ten empty
ones. Rearrange the blocks however your plate is actually laid out.

**Well ranges**

| Spec | Means |
|---|---|
| `A1` | one well |
| `A1:F12` | the rectangle from A1 to F12 |
| `A1,B4,C7` | those three wells |
| `A` / `A:F` | a whole row / rows A to F |
| `3` / `5:8` | a whole column / columns 5 to 8 |
| `all` | every well |

**Concentration series**

```yaml
series:
  across: columns      # or rows
  replicates: 2        # adjacent columns sharing one concentration
  top: 10 uM           # highest, at the low-numbered end
  dilution: 3          # 3-fold serial dilution
  reverse: true        # if column 1 holds the *lowest* instead
# or, instead of top/dilution:
  values: [10 uM, 1 uM, 100 nM, 0]
```

**Roles**

| Role | Meaning |
|---|---|
| `sample` | a normal reaction (the default) |
| `positive_control` | no inhibitor — the 100 % activity reference |
| `negative_control` | no enzyme, or fully inhibited |
| `blank` | background; can be subtracted from samples |
| `empty` | nothing in the well — ignored entirely |

### A table instead

If you would rather fill in a spreadsheet, `enzkin new-map -o plate_map.csv`
writes one row per well:

```
well,protein,protein_conc,compound,compound_conc,substrate,substrate_conc,role
A1,MMP-9,5 nM,Compound-1,10 uM,FRET-substrate,50 uM,sample
```

Column names are matched loosely — `enzyme`, `inhibitor`, `Inhibitor conc`,
`substrate_conc_uM` all work. Put the unit with the number (`10 uM`), in the
header (`compound_conc_uM`), or in its own `compound_unit` column; either way
nothing needs converting.

### Or grids pasted from a plate template

If you keep your layout as plate-shaped blocks in Excel, paste them straight
in, one grid per field:

```
# substrate_conc_uM
,1,2,3
A,50,50,25
B,25,25,12.5

# compound
,1,2,3
A,Cmpd1,Cmpd1,Cmpd1
```

All three formats produce identical results — use whichever matches how you
already think about the plate.

---

## One window, shared by matched wells

A slope is only comparable with another slope if both came from the same
stretch of the reaction. Fitting well A over 0–30 min and well B over 0–60 min
biases the comparison in a known direction: on a curve that bends, the longer
window averages in more of the slow tail, so B reads slower than it is for
reasons that have nothing to do with what is in it.

So the window is chosen **once** and shared. It is chosen from the
**no-inhibitor controls**, because they are the fastest wells on the plate and
therefore the first to bend — a window over which the controls are straight is
a window over which every inhibited well sharing their protein and substrate is
also straight. Wells are matched on protein and substrate concentration, since
how long a reaction stays linear depends on how fast it consumes its substrate.

```
--window auto     plate-wide when that still leaves a usable range,
                  otherwise per matched group  (default)
--window plate    one window for every well on the plate
--window group    one window per protein + substrate combination
--window well     every well its own range — NOT comparable between wells
```

`--window-reference` picks what decides it (`controls`, falling back to the
`fastest` wells where a group has no control; or `all`). `--window-match` sets
which factors wells must share. `--window-consensus intersection` (default)
takes the range every reference well was straight over; `median` is more
forgiving of one odd control.

### Seeing why

`figures/window_choice.png` is the justification, and it is worth a look before
trusting any number:

- **Top** — the window each matched group ended up with, on one time axis, with
  grey ticks showing what each reference well was individually straight over.
  If the bars all line up, one window suited the whole plate.
- **Middle** — the reference curves, with the readings that were left out
  greyed and the window edge dashed.
- **Bottom** — the argument. Each reference well's rate measured over a short
  sliding window, as a percentage of the rate finally fitted. While that stays
  flat the reaction is still linear; where it rolls off is where the window has
  to end.

`linear_range_windows.csv` and the report carry the same thing as a table.

On the example plate every group lands on 1.0–117.0 min — one window for all 86
wells, set by control G10, which is the first to bend. Sharing it costs nothing
there: median replicate CV is 3.1 % either way.

## How the window itself is found

A progress curve usually has a short lag, a straight initial-velocity region,
and a tail that bends over as substrate runs out. The default method anchors
its search at the *start* of the reaction and extends the window for as long as
the leftover scatter stays consistent with that well's own noise.

Two choices in there are worth knowing about, because they change the numbers:

**Straightness is judged against the well's own noise, not a fixed R².**
On a dose-response plate the rates span a hundredfold. A well running at
1 RFU/min can be perfectly straight and still never reach R² = 0.98 — its R²
ceiling is set by the reader's noise, not by curvature. Comparing the residual
scatter against the noise asks the right question at every signal level. R² is
still reported for every fit, and `--r2-min` still imposes one if you want it.

**The search is anchored and the candidate windows are long.**
Taking the steepest of many short windows is an upward-biased estimate: with
enough candidates, some stretch of noise always drifts upwards. On the example
plate that inflated the fully-inhibited wells roughly twofold — precisely the
wells that set an IC₅₀'s bottom plateau.

Validated both ways: on the example plate the replicate CV matches what you get
from fitting whole curves that are genuinely linear (median 2.7 %), and on
synthetic curves with known initial rates it recovers them about 3.5× more
accurately than a whole-curve fit. Curves too fast for the sampling interval
are flagged rather than quietly under-reported.

### Other methods

```bash
--method auto        # the above (default)
--method fixed --fixed-start 0 --fixed-end 20      # a set window, in minutes
--method max_slope --window-points 10              # steepest N readings
--method initial --window-points 10                # first N readings
--method all                                       # the whole curve
```

These describe how the window is found from the reference wells; it is still
shared according to `--window`.

Useful adjustments: `--tolerance` (how much scatter still counts as straight;
higher keeps longer windows), `--from` / `--until` (ignore part of the run),
`--direction decreasing` (absorbance assays such as NADH consumption),
`--min-points`.

### Flags

Wells that need a look are marked in the figure and listed in the report.

| Flag | Meaning |
|---|---|
| `low_signal` | change is within noise — the slope means nothing |
| `no_linear_range` | the whole run barely moves; the whole curve was fitted |
| `fast_reaction` | levelled off by the end — read sooner or dilute the enzyme |
| `poor_fit` | low R² — inspect the curve |
| `saturated` | repeated identical maxima — the detector may be maxed out |
| `relaxed` | no window was straight enough; criteria had to be loosened |
| `lag_phase`, `curve_bends`, `uses_whole_curve`, `decreasing` | descriptive, not problems |

---

## What comes out

```
results/
├── condition_means.csv        replicate mean, SD, SEM, CV %, n, % inhibition
├── well_results.csv           every well: rate, R², window, SNR, flags
├── plate_map_resolved.csv     what the tool thinks is in each well
├── linear_range_windows.csv   the window each matched group was fitted over
├── analysis_report.txt        how each rate was obtained, and what to check
├── kinetics_results.xlsx      all of the above as one workbook
├── prism/
│   ├── prism_substrate_by_compound_mean.csv          Michaelis-Menten
│   ├── prism_substrate_by_compound_replicates.csv    replicate subcolumns
│   ├── prism_compound_by_substrate_mean.csv          dose-response
│   ├── prism_compound_by_substrate_replicates.csv
│   ├── prism_compound_by_substrate_percent_activity.csv
│   └── prism_compound_by_substrate_log_percent_activity.csv
└── figures/
    ├── plate_curves.png       every raw curve with its fitted range
    ├── window_choice.png      why the window is where it is
    ├── rate_heatmap.png       rate per well, in plate layout
    └── rate_vs_*.png
```

### Pasting into Prism

| File | Prism table |
|---|---|
| `*_mean.csv` | XY, **Y: single values**, one dataset per column |
| `*_replicates.csv` | XY, **Y: replicate values in side-by-side subcolumns** (set 2 per dataset) |
| `*_percent_activity.csv` | as above, for `log(inhibitor) vs. normalized response` |
| `*_log_percent_activity.csv` | X is already `log10`, for a plain sigmoidal fit |

Open the file, select all, paste at the first cell.

---

## Background subtraction

| Mode | What it does |
|---|---|
| `--blank slope` *(default)* | subtracts the mean blank **rate** — corrects drift and substrate autohydrolysis |
| `--blank trace` | subtracts blank readings point by point before fitting |
| `--blank none` | ignores blank wells |

If your blanks are themselves a substrate titration, each sample is corrected
by the blank at *its own* substrate concentration, automatically.

---

## From Python

```python
from enzkin import read_kinetics, read_map, analyse, write_outputs

data   = read_kinetics("plate1.csv")
layout = read_map("layout.yaml")
result = analyse(data, layout, rate_unit="min", blank_mode="slope")

print(result.summary())
print(result.window_plan.summary())
result.condition_table().to_csv("means.csv", index=False)
write_outputs(result, "results/")
```

`result.wells["A5"].fit` carries the chosen window, R², standard error, noise
estimate and flags for a single well; `result.window_plan` carries the shared
window, the reference wells behind it and each group's own range.

To change how widely the window is shared:

```python
from enzkin.windows import WindowPolicy

result = analyse(data, layout,
                 window_policy=WindowPolicy(scope="group", reference="controls"))
```

---

## Tests

```bash
pip install -e ".[dev]" && pytest
```

134 tests. Beyond the plumbing, they check what has to be true of the
chemistry: replicates agree, the dose-response and Michaelis-Menten series are
both monotonic, controls are the fastest wells, fully-inhibited wells read near
zero rather than being inflated by noise, and a control that bends early pulls
the slow wells sharing its conditions into the same shorter window.
