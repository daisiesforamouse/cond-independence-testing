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

    X = theta * Z + rho * U + rng.normal(size = n)
    Y = theta * Z + rho * U + rng.normal(size = n)

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

def main(recompute):
    ns = np.asarray([50, 100, 200, 400, 800, 1600])
    bin_sizes = [2, 4, 8, 0.1, 0.25, 0.5]

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    fig_dir  = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    f_binary = data_dir / "example_2_ps_binary.npy"
    f_sizing = data_dir / "example_2_ps_sizing.npy"
    f_fixed = data_dir / "example_2_ps_fixed.npy"

    mc_reps = 400
    p_val_mc_reps = 400

    if recompute:
        test_fns = [T_binary, T]
        ps_binary = utility.p_val_dist(
            [n for _ in test_fns for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.2, rng=rng),
            lambda Z: adaptive_bins(Z, 2),
            [test_fn for test_fn in test_fns for n in ns],
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_binary, ps_binary)

        bin_sizes_with_n = [size if size > 1 else int(np.floor(2 * np.power(n, size)))
                            for size in bin_sizes for n in ns]
        ps_sizing = utility.p_val_dist(
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.2, rng=rng),
            [lambda Z: adaptive_bins(Z, size) for size in bin_sizes_with_n],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_sizing, ps_sizing)

        bin_widths_with_n = np.asarray(bin_sizes_with_n) / np.asarray([n for _ in bin_sizes for n in ns])
        ps_fixed = utility.p_val_dist(
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.2, rng=rng),
            [lambda Z: fixed_bins(Z, width) for width in bin_widths_with_n],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_fixed, ps_fixed)
    else:
        # load existing results
        missing = [p for p in (f_binary, f_sizing) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing saved results. Re-run with --recompute.\n"
                + "\n".join(str(p) for p in missing)
            )

        ps_binary = np.load(f_binary, allow_pickle=True)
        ps_sizing = np.load(f_sizing, allow_pickle=True)
        ps_fixed = np.load(f_fixed, allow_pickle=True)

    utility.plot_rejection(
        utility.rejection_rates(ps_binary),
        ns,
        ["binary", "nonbinary"],
        lambda name: name,
        savepath=fig_dir / "example_2_binary.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )

    utility.plot_rejection(
        utility.rejection_rates(ps_sizing),
        ns,
        bin_sizes,
        lambda size: f"m = {size}" if size > 1 else f"m = n^{size}",
        savepath=fig_dir / "example_2_sizing.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )

    utility.plot_rejection(
        utility.rejection_rates(ps_fixed),
        ns,
        bin_sizes,
        lambda size: f"width = {size} / n" if size > 1 else f"width = n^{size - 1}",
        savepath=fig_dir / "example_2_fixed.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute Monte Carlo simulations instead of loading saved .npy files."
    )
    args = parser.parse_args()
    main(args.recompute)
