# NNPT Numerical Experiments

This repo contains small scripts to reproduce the figures/experiments in the project.

## Setup

Create an environment with the required Python packages (at minimum: `numpy`, `matplotlib`, `seaborn`, `tqdm`).  
Make sure the local modules (`lpt.py`, `utility.py`, etc.) are on your `PYTHONPATH` (running from the repo root is simplest).

## Reproducing Example 1

`example1.py` runs Monte Carlo simulations to estimate Type I error under different binning schemes, then saves plots to `figures/` and cached results to `data/`.

### Recommended (recompute and save results)

```bash
python example1.py --recompute
```

This will:
- run the Monte Carlo simulations
- save p-value draws to:
  - `data/example_1_ps_fixed_bins.npy`
  - `data/example_1_ps_adaptive_bins.npy`
  - `data/example_1_ps_pareto.npy`
- save figures to:
  - `figures/example_1_fixed_bins.png`
  - `figures/example_1_adaptive_bins.png`
  - `figures/example_1_pareto.png`

### Use cached results (no recomputation)

```bash
python example1.py
```

If the `.npy` files are missing, the script will error and ask you to rerun with `--recompute`.

## Notes

- The `data/` and `figures/` directories are created automatically if needed.
- Simulation settings (sample sizes, bin exponents, number of MC repetitions) are defined near the top of `main()` in `example1.py`.
