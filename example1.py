import lpt
import utility

import argparse
from pathlib import Path

import numpy as np

def T(X, Y, Z, bins):
    X = np.asarray(X)
    Y = np.asarray(Y)

    idx = np.asarray(bins, dtype=np.intp)
    i = idx[:, 0]
    j = idx[:, 1]

    Tks = (X[i] - X[j]) * (Y[i] - Y[j])
    return np.sum(Tks)

def T_no_clt(X, Y, Z, bins):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    idx = np.asarray(bins, dtype=np.intp)
    i = idx[:, 0]
    j = idx[:, 1]

    Tks = (X[i] - X[j]) * np.sign(Z[i] - Z[j]) * np.sign((Y[i] - Y[j]) * np.sign(Z[i] - Z[j]))
    return np.sum(Tks > 1)

def sample_XYZ(n, theta, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)

    Z = rng.random(size=n) * theta

    X = Z + rng.random(size=n)
    Y = Z + rng.random(size=n)

    return X, Y, Z

def adaptive_bins(Z, bin_size):
    """
    Sorts Z and then just bin in order, with each bin of size bin_size
    """
    order = np.argsort(Z)
    bins = np.array_split(order, np.ceil(len(order) / bin_size).astype(int))

    return bins

def main(sims):
    ns = np.asarray([50, 100, 200, 400, 800, 1600, 3200, 6400])
    support_size_exps = np.asarray([0, 0.5, 0.6, 0.75])

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    f_no_clt = data_dir / "example_1_ps_no_clt_merged.npy"
    f_nnpt = data_dir / "example_1_ps_nnpt_merged.npy"

    mc_reps = 3000
    p_val_mc_reps = 5000

    sampling_fns = [lambda n, rng, exp=exp: sample_XYZ(n, np.power(n, exp), rng=rng)
                    for exp in support_size_exps for _ in ns]
    batch_ns = [n for _ in support_size_exps for n in ns]

    sim_configs = {
        "no_clt": (f_no_clt, T_no_clt),
        "nnpt":   (f_nnpt,   T),
    }

    for sim in sims:
        f, stat = sim_configs[sim]
        ps = utility.p_val_dist(
            batch_ns,
            sampling_fns,
            lambda Z: adaptive_bins(Z, 2),
            stat,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f, ps)


def plot(sims, fig_dir=Path("figures")):
    ns = np.asarray([50, 100, 200, 400, 800, 1600, 3200, 6400])
    support_size_exps = np.asarray([0, 0.5, 0.6, 0.75])

    data_dir = Path("data")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "no_clt" in sims:
        ps = np.load(data_dir / "example_1_ps_no_clt_merged.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            support_size_exps,
            lambda exp: f"$\\theta = n^{{{exp}}}$",
            savepath=fig_dir / "example_1_no_clt_merged_zoomed.png",
            x_axis="n",
            y_axis="Type I error rate",
            x_geom=True,
            y_lim=(0, 1),
        )

    if "nnpt" in sims:
        ps = np.load(data_dir / "example_1_ps_nnpt_merged.npy", allow_pickle=True)
        utility.plot_rejection(
            utility.rejection_rates(ps),
            ns,
            support_size_exps,
            lambda exp: f"$\\theta = n^{{{exp}}}$",
            savepath=fig_dir / "example_1_nnpt_merged.png",
            x_axis="n",
            y_axis="Type I error rate",
            x_geom=True,
            y_lim=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sims",
        nargs="+",
        choices=["no_clt", "nnpt"],
        default=["no_clt", "nnpt"],
        help="Which simulations to run (default: both)."
    )
    args = parser.parse_args()
    main(args.sims)
