import lpt
import utility

import numpy as np
from scipy.spatial import cKDTree

import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def T(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        avg_X_bin = np.mean(X[b])
        avg_Y_bin = np.mean(Y[b])
        Tks[k] = np.sum(np.abs([X[i] * Y[i] - avg_X_bin * avg_Y_bin for i in b]))
    return np.sum(Tks)

def T_binary(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        avg_X_bin = np.mean(X[b])
        avg_Y_bin = np.mean(Y[b])
        sd_X_bin = np.std(X[b])
        sd_Y_bin = np.std(Y[b])
        Tks[k] = 2 * (np.sum(np.abs(X[b] + Y[b] - avg_X_bin - avg_Y_bin)) > len(b) * (sd_X_bin + sd_Y_bin)) - 1
    return np.sum(Tks)

def sample_XYZ_linear(n, theta, rho):
    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)
    
    Z = rng.random(size = n)
    U = rng.normal(size = n)

    X = theta * Z + rho * U + rng.normal(size = n)
    Y = theta * Z + rho * U + rng.normal(size = n)

    return X, Y, Z

def sample_XYZ_circular(n, rho = 0.0, theta = 0.0):
    """
    Z has polar representation (radius, angle), where radius is log-normal and angle is uniform on [0, 2pi].
    X, Y are maringally N(|Z|, 1) with correlation rho * (cos(theta * Z_1) + cos(theta * Z_2))

    n: number of samples
    rho: strength of X, Y correlation 
    theta: strength of the oscillation of the X, Y correlation
    """
    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)
    
    radii = rng.lognormal(mean = 0.0, sigma = 0.2, size = n)
    angles = rng.random(size = n) * 2 * np.pi

    Z = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle, radius in zip(angles, radii)])

    def cov(z_1, z_2):
        return np.array([[1.0, 0.5 * rho * (np.cos(theta * z_1) + np.cos(theta * z_2))],
                         [0.5 * rho * (np.cos(theta * z_1) + np.cos(theta * z_2)), 1.0]])

    for i in range(n):
        X[i], Y[i] = rng.multivariate_normal([np.linalg.norm(Z[i]), np.linalg.norm(Z[i])],
                                             cov(*Z[i]))

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

def fixed_bins_2d(Z, bin_width, zmin=None):
    """
    2D analog of fixed_bins: partitions Z into axis-aligned square bins.

    Z: (n, 2) array
        2D covariates.
    bin_width: float or (2,) array-like
        Bin width(s) in each dimension. If scalar, uses same width for x and y.
    zmin: None or (2,) array-like
        Optional origin for the grid. If None, uses per-dimension min(Z).

    Returns
    -------
    bins : list[list[int]]
        bins[g] contains indices i whose Z[i] falls in grid cell g.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Expected Z of shape (n, 2), got {Z.shape}")

    bw = np.asarray(bin_width, dtype=float)
    if bw.ndim == 0:
        bw = np.array([bw, bw], dtype=float)
    if bw.shape != (2,) or np.any(bw <= 0):
        raise ValueError("bin_width must be a positive scalar or length-2 array-like")

    if zmin is None:
        z0 = Z.min(axis=0)
    else:
        z0 = np.asarray(zmin, dtype=float)
        if z0.shape != (2,):
            raise ValueError("zmin must be None or length-2 array-like")

    # integer grid coordinate per point
    ij = np.floor((Z - z0) / bw).astype(int)  # (n, 2)

    # group indices by cell (i,j)
    bins_dict = {}
    for idx, cell in enumerate(map(tuple, ij)):
        bins_dict.setdefault(cell, []).append(idx)

    # return just the bins (order doesn't matter for most uses)
    return list(bins_dict.values())

def greedy_knn_bins(Z, k, knn=100, trials=100, seed=0):
    """
    Greedy partition into groups of size k, aiming to minimize max pairwise
    distance within each group (diameter).

    Z: (n, d)
    knn: how many nearest neighbors to consider per seed point
    trials: how many random candidate (k-1)-subsets to try from the knn list
    """

    def group_diameter(Z):
        diffs = Z[:, None, :] - Z[None, :, :]
        return np.sqrt((diffs * diffs).sum(axis=-1)).max()

    rng = np.random.default_rng(seed)
    n = len(Z)

    tree = cKDTree(Z)
    unused = np.ones(n, dtype=bool)

    order = np.arange(n)
    rng.shuffle(order)

    bins = []

    for i in order:
        if sum(unused) <= k:
            break

        if not unused[i]:
            continue

        # candidate neighbors (including i itself) from kNN
        nn = []
        num_to_check = knn
        while len(nn) < k - 1:
            d, nn = tree.query(Z[i], k=min(num_to_check, n))
            nn = [j for j in nn if unused[j] and j != i]
            num_to_check = min(num_to_check * 2, n)

        # pick best subset of size k-1 from candidates via random trials
        best = None
        best_diam = np.inf

        # always include i, choose k-1 from nn
        for _ in range(trials):
            sel = rng.choice(nn, size=k-1, replace=False)
            idx = np.concatenate([[i], sel])
            diam = group_diameter(Z[idx])
            if diam < best_diam:
                best_diam = diam
                best = idx

        bins.append(best)
        for j in best:
            unused[j] = False

    leftovers = np.flatnonzero(unused)
    bins.append(leftovers)

    return bins

def main(recompute: bool):
    ns = np.asarray([50, 100, 200, 400])
    bin_width_exps = np.asarray([-1, -0.5, -0.4, -0.35, -0.25, -0.2])
    support_size_exps = np.asanyarray([0, 0.25, 0.5, 0.75, 1])

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    fig_dir  = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    mc_reps = 250
    p_val_mc_reps = 250

    if recompute:
        ps_adaptive_bins = utility.p_val_dist(
            [n for _ in bin_width_exps for n in ns],
            lambda n: sample_XYZ(n, 1, 0.0),
            [lambda Z, n=n, exp=exp: adaptive_bins(Z, int(np.floor(2 * np.power(n, 1 + exp))))
             for exp in bin_width_exps for n in ns],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(data_dir / "example_1_ps_adaptive_bins.npy", ps_adaptive_bins)

        ps_pareto = utility.p_val_dist(
            [n for _ in bin_width_exps for n in ns],
            lambda n: sample_XYZ_pareto(n, 1, 0.0),
            [lambda Z, n=n, exp=exp: adaptive_bins(Z, int(np.floor(2 * np.power(n, 1 + exp))))
             for exp in bin_width_exps for n in ns],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(data_dir / "example_1_ps_pareto.npy", ps_pareto)

        ps_nnpt = utility.p_val_dist(
            [n for _ in support_size_exps for n in ns],
            [lambda n, exp=exp: sample_XYZ(n, 1, np.power(n, exp))
             for exp in support_size_exps for n in ns],
            lambda Z: adaptive_bins(Z, 2),
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(data_dir / "example_1_ps_nnpt.npy", ps_nnpt)

    else:
        # load existing results
        missing = [p for p in (f_fixed, f_adapt, f_pareto) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing saved results. Re-run with --recompute.\n"
                + "\n".join(str(p) for p in missing)
            )

        # ps_fixed_bins = np.load(f_fixed, allow_pickle=True)
        ps_adaptive_bins = np.load(f_adapt, allow_pickle=True)
        ps_pareto = np.load(f_pareto, allow_pickle=True)

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
        utility.rejection_rates(ps_pareto),
        ns,
        bin_width_exps,
        lambda exp: f"m = n^{1 + exp}",
        savepath=fig_dir / "example_1_pareto.png",
        x_axis = "n",
        y_axis = "Type I error rate"
    )

    utility.plot_rejection(
        utility.rejection_rates(ps_nnpt),
        ns,
        support_size_exps,
        lambda exp: f"theta = n^{exp}",
        savepath=fig_dir / "example_1_nnpt.png"
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
        
ps = utility.p_val_dist(500,
                        0.05,
                        lambda n: sample_XYZ_circular(n, 0.25, 1),
                        lambda Z: greedy_knn_bins(Z, 10),
                        T,
                        mc_reps=100,
                        p_val_mc_reps=100)


# ps_binary = utility.p_val_dist(500,
#                                0.05,
#                                lambda n: sample_XYZ(n, 0.25, 2),
#                                lambda Z: greedy_knn_bins(Z, 2),
#                                T_binary,
#                                mc_reps=100,
#                                p_val_mc_reps=100)
