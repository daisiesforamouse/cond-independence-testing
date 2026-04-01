import lpt
import utility

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

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

def optimal_power(x, rho, alpha=0.05):
    return norm.cdf(x * rho / (1 - rho ** 2) - norm.ppf(1 - alpha))

def sample_XYZ(n, theta, beta, *, rng):
    if rng is None:
        rng = np.random.default_rng()

    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)

    Z = rng.random(size = n)
    U = rng.normal(size = n)

    mean = (32 / 3) * np.pow(Z, 3) + 16 * np.pow(Z, 2) + (19 / 3) * Z

    X = theta * mean + beta * U + rng.normal(size = n)
    Y = theta * mean + beta * U + rng.normal(size = n)

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

def main(sims):
    ns = np.asarray([200, 400, 800, 1600, 3200, 6400])
    bin_sizes = [2, 6, 12, 0.15, 0.2, 0.35, 0.5]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    mc_reps = 1000
    p_val_mc_reps = 5000

    bin_sizes_with_n = [size if size > 1 else int(np.floor(np.power(n, size)))
                        for size in bin_sizes for n in ns]
    bin_widths_with_n = np.asarray(bin_sizes_with_n) / np.asarray([n for _ in bin_sizes for n in ns])

    beta = 0.2
    sim_configs = {
        "sizing": (
            data_dir / "example_2_ps_sizing.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, beta, rng=rng),
            [lambda Z, size=size: adaptive_bins(Z, size) for size in bin_sizes_with_n],
            T,
        ),
        "fixed": (
            data_dir / "example_2_ps_fixed.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, beta, rng=rng),
            [lambda Z, width=width: fixed_bins(Z, width) for width in bin_widths_with_n],
            T,
        ),
        "sizing_validity": (
            data_dir / "example_2_sizing_validity.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.0, rng=rng),
            [lambda Z, size=size: adaptive_bins(Z, size) for size in bin_sizes_with_n],
            T,
        ),
        "fixed_validity": (
            data_dir / "example_2_fixed_validity.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.0, rng=rng),
            [lambda Z, width=width: fixed_bins(Z, width) for width in bin_widths_with_n],
            T,
        ),
    }

    for sim in sims:
        f, batch_ns, sampling_fn, binning_fns, stat = sim_configs[sim]
        ps = utility.p_val_dist(
            batch_ns, sampling_fn, binning_fns, stat,
            mc_reps=mc_reps, p_val_mc_reps=p_val_mc_reps
        )
        np.save(f, ps)


def plot(sims, fig_dir=Path("figures")):
    ns = np.asarray([200, 400, 800, 1600, 3200, 6400])
    bin_sizes = [2, 6, 0.2, 0.5]

    data_dir = Path("data")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    beta = 0.2
    rho = (beta ** 2) / ((1 + beta ** 2) ** 2)
    ns_fine = np.arange(ns[0], ns[-1] + 1)
    opt_power_line = {
        "x": ns_fine,
        "y": optimal_power(np.sqrt(ns_fine), rho),
        "color": "black",
        "linestyle": "dotted",
        "linewidth": 1.8,
        "label": "optimal power",
    }

    if "sizing" in sims and "fixed" in sims:
        ps_sizing = np.load(data_dir / "example_2_ps_sizing.npy", allow_pickle=True)
        ps_fixed = np.load(data_dir / "example_2_ps_fixed.npy", allow_pickle=True)
        rr_sizing = utility.rejection_rates(ps_sizing)
        rr_fixed = utility.rejection_rates(ps_fixed)
        y_lim_combined = (0, 1)
        _, ax = utility.plot_rejection(
            rr_sizing,
            ns,
            bin_sizes,
            lambda size: f"$m = {size}$" if size > 1 else f"$m = n^{{{size}}}$",
            x_axis="n",
            y_axis="Power",
            y_lim=y_lim_combined,
        )
        utility.plot_rejection(
            rr_fixed,
            ns,
            bin_sizes,
            lambda size: f"$w = {size}/n$" if size > 1 else f"$w = n^{{{size - 1}}}$",
            ax=ax,
            linestyle="dashed",
            x_axis="n",
            y_axis="Power",
            savepath=fig_dir / "example_2_sizing_and_fixed.png",
            y_lim=y_lim_combined,
            extra_lines=[opt_power_line],
        )
    elif "sizing" in sims:
        ps = np.load(data_dir / "example_2_ps_sizing.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$m = {size}$" if size > 1 else f"$m = n^{{{size}}}$",
            savepath=fig_dir / "example_2_sizing.png",
            x_axis="n",
            y_axis="Power",
            y_lim=None,
            extra_lines=[opt_power_line],
        )
    elif "fixed" in sims:
        ps = np.load(data_dir / "example_2_ps_fixed.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$w = {size}/n$" if size > 1 else f"$w = n^{{{size - 1}}}$",
            savepath=fig_dir / "example_2_fixed.png",
            x_axis="n",
            y_axis="Power",
            y_lim=None,
            extra_lines=[opt_power_line],
        )

    alpha_line = {
        "y": 0.05,
        "color": "0.2",
        "linestyle": "--",
        "linewidth": 1.5,
        "label": "$\\alpha = 0.05$",
    }

    if "sizing_validity" in sims:
        ps = np.load(data_dir / "example_2_sizing_validity.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$m = {size}$" if size > 1 else f"$m = n^{{{size}}}$",
            savepath=fig_dir / "example_2_sizing_validity.png",
            x_axis="n",
            y_axis="Type I error rate",
            y_lim=None,
            extra_lines=[alpha_line],
        )
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$m = {size}$" if size > 1 else f"$m = n^{{{size}}}$",
            savepath=fig_dir / "example_2_sizing_validity_zoom.png",
            x_axis="n",
            y_axis="Type I error rate",
            y_lim=(0, 0.1),
            extra_lines=[alpha_line],
        )

    if "fixed_validity" in sims:
        ps = np.load(data_dir / "example_2_fixed_validity.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$w = {size}/n$" if size > 1 else f"$w = n^{{{size - 1}}}$",
            savepath=fig_dir / "example_2_fixed_validity.png",
            x_axis="n",
            y_axis="Type I error rate",
            y_lim=None,
            extra_lines=[alpha_line],
        )
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            bin_sizes,
            lambda size: f"$w = {size}/n$" if size > 1 else f"$w = n^{{{size - 1}}}$",
            savepath=fig_dir / "example_2_fixed_validity_zoom.png",
            x_axis="n",
            y_axis="Type I error rate",
            y_lim=(0, 0.1),
            extra_lines=[alpha_line],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sims",
        nargs="+",
        choices=["sizing", "fixed", "sizing_validity", "fixed_validity"],
        default=["sizing", "fixed", "sizing_validity", "fixed_validity"],
        help="Which simulations to run (default: all)."
    )
    args = parser.parse_args()
    main(args.sims)
