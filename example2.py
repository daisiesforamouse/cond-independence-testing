import lpt
import utility

import argparse
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def T(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        b = np.asarray(b, dtype=np.intp)
        m = b.size
        if m <= 1:
            continue

        xb = X[b]
        yb = Y[b]

        sxy = np.dot(xb, yb)
        sx  = xb.sum()
        sy  = yb.sum()

        pair_sum = m * sxy - sx * sy

        Tks[k] = pair_sum / (m * (m - 1))

    return np.sum(Tks)

def T_binary(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        Tks[k] = (X[b[0]] - X[b[1]]) * (Y[b[0]] - Y[b[1]]) >= 0

    return np.mean(Tks)

def sample_XYZ(n, theta, rho, *, rng):
    if rng is None:
        rng = np.random.default_rng()

    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)

    Z = rng.random(size = n)
    U = rng.normal(size = n)

    mean = (32 / 3) * np.pow(Z, 3) + 16 * np.pow(Z, 2) + (19 / 3) * Z

    X = theta * mean + rho * U + rng.normal(size = n)
    Y = theta * mean + rho * U + rng.normal(size = n)

    return X, Y, Z

def adaptive_bins(Z, bin_size):
    """
    Sorts Z and then just bin in order, with each bin of size bin_size
    """
    order = np.argsort(Z)
    bins = np.array_split(order, np.ceil(len(order) / bin_size).astype(int))

    return bins

def fixed_bins(Z, bin_width):
    """
    Partitions Z into bins [0, bin_width], [bin_width, 2 * bin_width], so on
    """
    bin_nums = np.floor(Z / bin_width).astype(int)
    bins = [[] for _ in range(np.max(bin_nums) + 1)]

    for i, num in enumerate(bin_nums):
        bins[num].append(i)

    return bins

def main(recompute, sims):
    ns = np.asarray([200, 400, 800, 1600, 3200, 6400])
    bin_sizes = [2, 6, 12, 0.15, 0.2, 0.35, 0.5]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    fig_dir  = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    mc_reps = 1000
    p_val_mc_reps = 5000

    bin_sizes_with_n = [size if size > 1 else int(np.floor(np.power(n, size)))
                        for size in bin_sizes for n in ns]
    bin_widths_with_n = np.asarray(bin_sizes_with_n) / np.asarray([n for _ in bin_sizes for n in ns])

    sim_configs = {
        "sizing": (
            data_dir / "example_2_ps_sizing.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.2, rng=rng),
            [lambda Z, size=size: adaptive_bins(Z, size) for size in bin_sizes_with_n],
            T,
            bin_sizes,
            lambda size: f"m = {size}" if size > 1 else f"m = n^{size}",
            "example_2_sizing.png",
            "Power",
            None,
        ),
        "fixed": (
            data_dir / "example_2_ps_fixed.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.2, rng=rng),
            [lambda Z, width=width: fixed_bins(Z, width) for width in bin_widths_with_n],
            T,
            bin_sizes,
            lambda size: f"width = {size} / n" if size > 1 else f"width = n^{size - 1}",
            "example_2_fixed.png",
            "Power",
            None,
        ),
        "validity": (
            data_dir / "example_2_validity.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.0, rng=rng),
            [lambda Z, size=size: adaptive_bins(Z, size) for size in bin_sizes_with_n],
            T,
            bin_sizes,
            lambda size: f"m = {size}" if size > 1 else f"m = n^{size}",
            "example_2_validity.png",
            "Type I error rate",
            None,
        ),
    }

    for sim in sims:
        f, batch_ns, sampling_fn, binning_fns, stat, plot_params, label_fn, fig_name, y_axis, y_lim = sim_configs[sim]
        if recompute:
            ps = utility.p_val_dist(
                batch_ns, sampling_fn, binning_fns, stat,
                mc_reps=mc_reps, p_val_mc_reps=p_val_mc_reps
            )
            np.save(f, ps)
        else:
            if not f.exists():
                raise FileNotFoundError(f"Missing saved results: {f}. Re-run with --recompute.")
            ps = np.load(f, allow_pickle=True)

        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            plot_params,
            label_fn,
            savepath=fig_dir / fig_name,
            x_axis="n",
            y_axis=y_axis,
            y_lim=y_lim,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute Monte Carlo simulations instead of loading saved .npy files."
    )
    parser.add_argument(
        "--sims",
        nargs="+",
        choices=["sizing", "fixed", "validity"],
        default=["sizing", "fixed", "validity"],
        help="Which simulations to run (default: all)."
    )
    args = parser.parse_args()
    main(args.recompute, args.sims)


