import lpt
import utility

import argparse
from pathlib import Path

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

def greedy_knn_bins(Z, k, *, seed=0):
    """
    Greedy partition into groups of size k, aiming to minimize max pairwise
    distance within each group (diameter).

    Z: (n, d)
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

        # query more than k points if needed, because some neighbors may be already used
        num_to_check = min(max(2 * k, k + 1), n)
        while True:
            _, nn = tree.query(Z[i], k=num_to_check)
            nn = np.atleast_1d(nn)

            # keep only unused points (including i)
            nn = [j for j in nn if unused[j]]
            if len(nn) >= k:
                group = np.array(nn[:k], dtype=int)  # includes i since i is unused and distance 0
                break

            num_to_check = min(2 * num_to_check, n)

        if group is None:
            break

        bins.append(group)
        unused[group] = False

    leftovers = np.flatnonzero(unused)
    if len(leftovers):
        bins.append(leftovers)

    return bins

def sa_bins(
    Z,
    k,
    *,
    seed=0,
    steps=50_000,
    T0=1.0,
    Tend=1e-4,
    p_norm=2,
):
    """
    Simulated-annealing partition of points into groups of size k (last group may be smaller).

    This is a *different solver* than greedy_knn_bins: it formulates the problem as:
      minimize sum_b cost(bin_b)
    over partitions with fixed bin sizes, and uses SA with point-swap moves.

    Parameters
    ----------
    Z : (n, d) array
    k : int
    steps : int
        Number of SA iterations.
    T0, Tend : float
        Temperature schedule endpoints (geometric).
    init : str
        "random": random partition.
    objective : str
        "diameter": bin cost = max pairwise distance within bin.
        "sse":      bin cost = sum squared distances to bin mean (k-means-like).
    p_norm : int/float
        Norm for diameter computation when objective="diameter".

    Returns
    -------
    bins : list[np.ndarray]
        List of index arrays. All but last have size k; last has size n % k (or k if divisible).
    """
    Z = np.asarray(Z)
    n = Z.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return []
    if k >= n:
        return [np.arange(n)]

    rng = np.random.default_rng(seed)

    def bin_cost(idx):
        m = len(idx)
        if m <= 1:
            return 0.0
        X = Z[idx]
        diffs = X[:, None, :] - X[None, :, :]
        if p_norm == 2:
            D2 = np.sum(diffs * diffs, axis=-1)
            return float(np.sqrt(np.max(D2)))
        else:
            D = np.linalg.norm(diffs, ord=p_norm, axis=-1)
            return float(np.max(D))

    perm = np.arange(n)
    rng.shuffle(perm)

    bins = greedy_knn_bins(Z, k)

    B = len(bins)
    if B <= 1:
        return bins

    # bin lookup table
    p2b = -np.ones(n, dtype=int)
    for i, b in enumerate(bins):
        p2b[b] = i

    # per-bin costs
    costs = np.array([bin_cost(b) for b in bins], dtype=float)
    cur = float(costs.sum())

    best_bins = [b.copy() for b in bins]
    best = cur

    # temperature schedule (geometric)
    Ts = np.geomspace(T0, Tend, num=max(2, steps))

    # swap bins in two bins for simulated annealing
    for t in range(steps):
        T = Ts[t]

        # choose two different bins
        i, bj = rng.integers(0, B, size=2)
        if i == bj:
            continue

        # choose points within those bins
        ai = int(rng.choice(bins[i]))
        aj = int(rng.choice(bins[bj]))

        # potential swap
        new_bi = bins[i].copy()
        new_bj = bins[bj].copy()
        new_bi[new_bi == ai] = aj
        new_bj[new_bj == aj] = ai

        old_ci, old_cj = costs[i], costs[bj]
        new_ci, new_cj = bin_cost(new_bi), bin_cost(new_bj)

        new_cur = cur - old_ci - old_cj + new_ci + new_cj
        delta = new_cur - cur

        # accept/reject
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            bins[i] = new_bi
            bins[bj] = new_bj
            costs[i] = new_ci
            costs[bj] = new_cj
            cur = new_cur

            p2b[ai], p2b[aj] = bj, i

            if cur < best:
                best = cur
                best_bins = [b.copy() for b in bins]

    out = best_bins
    return out

def main(recompute):
    ns = np.asarray([50, 100, 200, 400])
    bin_sizes = np.asarray([2, 4, 8, 16])

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    fig_dir  = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    f_2d_sizing = data_dir / "example_3_2d_sizing.npy"

    mc_reps = 100
    p_val_mc_reps = 100

    if recompute:
        ps_2d_sizing = utility.p_val_dist(
            [n for _ in bin_sizes for n in ns],
            lambda n: sample_XYZ_circular(n, 0.1, 0.0),
            [lambda Z, n=n, size=size: sa_bins(Z, size)
             for size in bin_sizes for n in ns],
            T,
            mc_reps=mc_reps,
            p_val_mc_reps=p_val_mc_reps
        )
        np.save(f_2d_sizing, ps_2d_sizing)

    else:
        # load existing results
        missing = [p for p in (f_2d_sizing) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing saved results. Re-run with --recompute.\n"
                + "\n".join(str(p) for p in missing)
            )

        ps_2d_sizing = np.load(f_2d_sizing, allow_pickle=True)

    utility.plot_rejection(
        utility.rejection_rates(ps_2d_sizing),
        ns,
        bin_sizes,
        lambda exp: f"m = n^{exp}",
        savepath=fig_dir / "example_3_2d_sizing.png",
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
        
# ps = utility.p_val_dist(500,
#                         0.05,
#                         lambda n: sample_XYZ_circular(n, 0.25, 1),
#                         lambda Z: greedy_knn_bins(Z, 10),
#                         T,
#                         mc_reps=100,
#                         p_val_mc_reps=100)


# ps_binary = utility.p_val_dist(500,
#                                0.05,
#                                lambda n: sample_XYZ(n, 0.25, 2),
#                                lambda Z: greedy_knn_bins(Z, 2),
#                                T_binary,
#                                mc_reps=100,
#                                p_val_mc_reps=100)
