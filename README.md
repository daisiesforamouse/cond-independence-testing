# NNPT Numerical Experiments

This repo contains small scripts to reproduce the figures/experiments in the project.

## Setup

Create an environment with the required Python packages (at minimum: `numpy`, `matplotlib`, `seaborn`, `tqdm`).
Make sure the local modules (`lpt.py`, `utility.py`, etc.) are on your `PYTHONPATH` (running from the repo root is simplest).

## Reproducing the examples

### Step 1: Run simulations

`example1.py`, `example2.py`, and `example3.py` run Monte Carlo simulations and save p-value arrays to `data/`.

```bash
python example1.py
python example2.py
python example3.py
```

To run only a subset of simulations within a script, use `--sims`:

```bash
python example1.py --sims nnpt
python example2.py --sims sizing validity
python example3.py --sims 2d_sizing 3d_fixed
```

### Step 2: Make plots

Once the `.npy` files are in `data/`, generate figures with `make_plots.py`:

```bash
python make_plots.py
```

To make only specific plots:

```bash
python make_plots.py --plots example1_nnpt example2_sizing example3_bins
```

Available plot names:

| Plot name | Source |
|---|---|
| `example1_no_clt` | example1 |
| `example1_nnpt` | example1 |
| `example2_sizing` | example2 |
| `example2_fixed` | example2 |
| `example2_sizing` + `example2_fixed` | example2 (combined plot, fixed as dashed lines) |
| `example2_sizing_validity` | example2 |
| `example2_fixed_validity` | example2 |
| `example3_2d_sizing` | example3 |
| `example3_2d_fixed` | example3 |
| `example3_2d_sizing_validity` | example3 |
| `example3_2d_fixed_validity` | example3 |
| `example3_3d_sizing` | example3 |
| `example3_3d_fixed` | example3 |
| `example3_3d_sizing_validity` | example3 |
| `example3_3d_fixed_validity` | example3 |
| `example3_bins` | example3 (bin visualisations, no `.npy` needed) |

Figures are saved to `figures/`.

## Notes

- The `data/` and `figures/` directories are created automatically if needed.
- Simulation settings (sample sizes, bin exponents, number of MC repetitions) are defined near the top of `main()` in each script.
