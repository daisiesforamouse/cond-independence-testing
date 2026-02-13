# NNPT Numerical Experiments

This repo contains small scripts to reproduce the figures/experiments in the project.

## Setup

Create an environment with the required Python packages (at minimum: `numpy`, `matplotlib`, `seaborn`, `tqdm`).  
Make sure the local modules (`lpt.py`, `utility.py`, etc.) are on your `PYTHONPATH` (running from the repo root is simplest).

## Reproducing the examples

`example1.py` (resp. `example2.py`, `example3.py`) runs Monte Carlo simulations to estimate Type I error under different binning schemes, then saves plots to `figures/` and cached results to `data/`.

```bash
python example1.py --recompute
```

This will:
- run the Monte Carlo simulations
- save p-value draws to `data`
- save figures to `figures`

### Use cached results (no recomputation)

```bash
python example1.py
```

If the `.npy` files are missing, the script will error and ask you to rerun with `--recompute`.

## Notes

- The `data/` and `figures/` directories are created automatically if needed.
- Simulation settings (sample sizes, bin exponents, number of MC repetitions) are defined near the top of `main()` in each file.
