# Five minutes to your first result

## 1 · Install

```bash
pip install -e ".[app]"
```

## 2 · Try it on the example

```bash
enzkin app
```

Your browser opens. **Use the example plate** and **Use the example layout**
are already selected, so you will see:

- **1 · Plate map** — the plate coloured by role. Hover any well.
- **2 · Curves & fits** — all 96 raw curves, each with the range that was
  fitted picked out in blue. Amber wells want a look.
- **3 · Results** — replicate means, the dose-response plot, and a
  **Everything (zip)** button.

## 3 · Now your own plate

Upload your CSV in the sidebar. It needs a `Time` column and one column per
well (`A1`, `A2`, …). Times may be `0:01:30` or plain numbers.

Then on the **Plate map** tab choose **Build it here** and add one block per
region of the plate:

| Field | What to put |
|---|---|
| Wells | `A1:F12`, or `A`, or `3`, or `A1,B2,C3` |
| Role | `sample`, or `positive_control` for your no-inhibitor wells |
| Each factor | the identity, then either one concentration or a dilution series |

For a serial dilution: pick **Serial dilution**, say whether it runs across
columns or down rows, how many wells share each concentration (2 for duplicate
column pairs), the highest concentration (`10 uM`), and the fold (`3`).

Write concentrations with their units — `50 uM` here and `3 nM` there. Nothing
needs converting.

Press **Save this layout (YAML)** when it looks right. Next time:

```bash
enzkin analyse my_plate.csv --map layout.yaml --out results/
```

## 4 · Into Prism

From `results/prism/`:

- **IC₅₀** → `prism_compound_by_substrate_percent_activity.csv`
  → Prism *XY*, `log(inhibitor) vs. normalized response`
- **Km / Vmax** → `prism_substrate_by_compound_mean.csv`
  → Prism *XY*, `Michaelis-Menten`
- **Replicates rather than means** → the `*_replicates.csv` of either
  → Prism *XY* with **2 replicate values in side-by-side subcolumns**

Open, select all, paste at the first cell.

## Two things worth checking every time

1. **The plate figure.** If the blue range on a curve is not where you would
   have put it, that well is telling you something — usually a lag, a plateau,
   or a bubble.
2. **`analysis_report.txt`.** It lists the wells that were flagged and why, and
   any conditions whose replicates disagree by more than 20 %.

## If something does not work

| Problem | Try |
|---|---|
| "no header row with well labels" | `enzkin check my_plate.csv` to see what was found; make sure well columns are named `A1`, `A2`, … |
| Rates look too low | The curve may be bending — check the plate figure; `--method fixed --fixed-start 0 --fixed-end 10` |
| Everything is flagged `low_signal` | Genuinely flat wells, or the wrong `--direction` for a falling signal |
| "no wells to analyse" | The plate map and the data file name different wells |
| A window is wrong on one well | In the app, open that well and use **Override this well's window** |
