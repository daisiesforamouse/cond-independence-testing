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
        i_idx, j_idx = np.triu_indices(len(b), k=1)
        for i, j in zip(i_idx, j_idx):
            Tks[k] += (X[b[i]] - X[b[j]]) * (Y[b[i]] - Y[b[j]])
        if len(b) > 1:
            Tks[k] /= len(b) * (len(b) - 1) / 2
    return np.sum(Tks)

def T_binary(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        Tks[k] = (X[b[i]] - X[b[j]]) * (Y[b[i]] - Y[b[j]]) >= 0
    
    return np.mean(Tks)

def sample_XYZ(n, theta, rho, *, rng=None):
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

def sample_XYZ_fat_tail(n, theta, rho, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)
    
    Z = rng.random(size=n)
    U = rng.normal(size=n)

    # X = theta * Z + rho * U + rng.choice([-1, 1], size = n) * rng.fat_tail(2, size = n)
    # Y = theta * Z + rho * U + rng.choice([-1, 1], size = n) * rng.fat_tail(2, size = n)

    X = theta * Z + rho * U + rng.standard_cauchy(size=n)
    Y = theta * Z + rho * U + rng.standard_cauchy(size=n)

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
    print(bin_width)
    bin_nums = np.floor(Z / bin_width).astype(int)
    bins = [[] for _ in range(np.max(bin_nums) + 1)]

    for i, num in enumerate(bin_nums):
        bins[num].append(i)

    return bins

def main(recompute):
    ns = np.asarray([50, 100, 200, 400])
    bin_width_exps = np.asarray([-1, -0.5, -0.4, -0.35, -0.25, -0.2])
    support_size_exps = np.asarray([0, 0.25, 0.5, 0.75, 1.0])

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    fig_dir  = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    f_adapt = data_dir / "example_1_ps_adaptive_bins.npy"
    f_fat_tail = data_dir / "example_1_ps_fat_tail.npy"
    f_nnpt = data_dir / "example_1_ps_nnpt.npy"

    mc_reps = 100
    p_val_mc_reps = 100

    if recompute:
        ps_adaptive_bins = utility.p_val_dist(
            [n for _ in bin_width_exps for n in ns],
            lambda n, rng: sample_XYZ(n, 1, 0.0, rng=rng),
            [lambda Z, n=n, exp=exp: adaptive_bins(Z, int(np.floor(2 * np.power(n, 1 + exp))))
             for exp in bin_width_exps for n in ns],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_adapt, ps_adaptive_bins)

        ps_fat_tail = utility.p_val_dist(
            [n for _ in support_size_exps for n in ns],
            [lambda n, rng, exp=exp: sample_XYZ_fat_tail(n, 10 * np.power(n, exp), 0.0, rng=rng)
             for exp in support_size_exps for n in ns],
            lambda Z: adaptive_bins(Z, 2),
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_fat_tail, ps_fat_tail)

        ps_nnpt = utility.p_val_dist(
            [n for _ in support_size_exps for n in ns],
            [lambda n, rng, exp=exp: sample_XYZ(n, np.power(n, exp), 0.0, rng=rng)
             for exp in support_size_exps for n in ns],
            lambda Z: adaptive_bins(Z, 2),
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_nnpt, ps_nnpt)

    else:
        # load existing results
        missing = [p for p in (f_adapt, f_fat_tail, f_nnpt) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing saved results. Re-run with --recompute.\n"
                + "\n".join(str(p) for p in missing)
            )

        ps_adaptive_bins = np.load(f_adapt, allow_pickle=True)
        ps_fat_tail = np.load(f_fat_tail, allow_pickle=True)
        ps_nnpt = np.load(f_nnpt, allow_pickle=True)

    utility.plot_rejection(
        utility.rejection_rates(ps_adaptive_bins),
        ns,
        bin_width_exps,
        lambda exp: f"m = n^{1 + exp}",
        savepath=fig_dir / "example_1_adaptive_bins.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )

    utility.plot_rejection(
        utility.rejection_rates(ps_fat_tail),
        ns,
        support_size_exps,
        lambda exp: f"theta = n^{exp}",
        savepath=fig_dir / "example_1_fat_tail.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )

    utility.plot_rejection(
        utility.rejection_rates(ps_nnpt),
        ns,
        support_size_exps,
        lambda exp: f"theta = n^{exp}",
        savepath=fig_dir / "example_1_nnpt.png",
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
