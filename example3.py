import lpt
import utility

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax

import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="paper")

def center(X):
    row_mean = X.mean(axis=1, keepdims=True)
    col_mean = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()
    return X - row_mean - col_mean + grand_mean

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

def T_universal(X, Y, Z, bins):
    Tks = np.zeros(len(bins))
    for k, b in enumerate(bins):
        m = len(b)
        if m > 1:
            xb = X[b] 
            yb = Y[b]

            dx = xb[:, None] - xb[None, :]
            dy = yb[:, None] - yb[None, :]

            gram_X = np.exp(-(dx * dx) / 2.0)
            gram_Y = np.exp(-(dy * dy) / 2.0)

            gram_X -= gram_X.mean(axis=0, keepdims=True)
            gram_Y -= gram_Y.mean(axis=0, keepdims=True)

            centering_matrix = np.eye(m) - np.ones((m, m)) / m

            Tks[k] = np.trace(center(gram_X) @ center(gram_Y)) / (m * m)
    return np.sum(Tks)

def sample_XYZ_circular(n, rho, theta, *, rng):
    """
    Z has polar representation (radius, angle), angle is uniform on [0, 2pi] and the radius (in [1, 2]) is selected
    such that Z is uniform on an annulus.
    X, Y are maringally N(radius, 1) with correlation rho * sin(theta * angle)

    n: number of samples
    rho: strength of X, Y correlation 
    theta: strength of the oscillation of the X, Y correlation
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.empty(n)
    Y = np.empty(n)
    Z = np.empty(n)

    radii = np.sqrt(1 + (5 / 4) * rng.random(size=n))
    angles = rng.random(size=n) * 2 * np.pi

    Z = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle, radius in zip(angles, radii)])

    for i in range(n):
        X[i], Y[i] = rng.multivariate_normal([radii[i], radii[i]],
                                             np.array([[1.0, rho * np.sin(theta * angles[i])],
                                                       [rho * np.sin(theta * angles[i]), 1.0]]))

    return X, Y, Z

def sample_XYZ_gaussian3d(n, rho_xy, rho_z, *, rng):
    """
    Z ~ N(0, Sigma_Z) in R^3, where Sigma_Z has equicorrelation rho_z.
    X = mean(Z) + rho_xy * U + sqrt(1 - rho_xy^2) * eps_X
    Y = mean(Z) + rho_xy * U + sqrt(1 - rho_xy^2) * eps_Y
    where U, eps_X, eps_Y ~ iid N(0, 1).

    When rho_xy = 0: X ⊥ Y | Z.
    When rho_xy > 0: X and Y share residual U, so X not ⊥ Y | Z.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = 3
    Sigma_Z = rho_z * np.ones((d, d)) + (1 - rho_z) * np.eye(d)
    Z = rng.multivariate_normal(np.zeros(d), Sigma_Z, size=n)

    z_proj = Z.mean(axis=1) / np.sqrt(d)

    U = rng.standard_normal(n)
    X = z_proj + rho_xy * U + np.sqrt(1 - rho_xy**2) * rng.standard_normal(n)
    Y = z_proj + rho_xy * U + np.sqrt(1 - rho_xy**2) * rng.standard_normal(n)

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

def fixed_bins_nd(Z, bin_width):
    """
    Axis-aligned grid binning for d-dimensional Z.
    Works for any dimension d.

    Z: (n, d) array
    bin_width: scalar — same grid width used along every axis.

    Returns
    -------
    bins : list[list[int]]
    """
    Z = np.asarray(Z)
    n, d = Z.shape
    bw = np.full(d, float(bin_width))
    z0 = Z.min(axis=0)
    ij = np.floor((Z - z0) / bw).astype(int)  # (n, d)
    bins_dict = {}
    for idx, cell in enumerate(map(tuple, ij)):
        bins_dict.setdefault(cell, []).append(idx)
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
    steps=50000,
    T0=1.0,
    Tend=1e-4,
    p_norm=2,
    focus_bins=True,
    partner_mode="near",
    refresh_every=200
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

    eps = 1e-12

    def make_cost_probs(costs):
        return softmax(np.asarray(costs, dtype=float) / 0.5)

    def compute_centroids(bins):
        C = np.zeros((len(bins), Z.shape[1]), dtype=float)
        for bi, b in enumerate(bins):
            C[bi] = Z[b].mean(axis=0)
        return C

    costs = np.array([bin_cost(b) for b in bins], dtype=float)
    cur = float(costs.sum())

    probs = make_cost_probs(costs)

    # swap bins in two bins for simulated annealing
    for t in range(steps):
        T = Ts[t]

        if t % refresh_every == 0:
            if focus_bins:
                probs = make_cost_probs(costs)
            if partner_mode == "near":
                centroids = compute_centroids(bins)

        # choose offender bin i
        if focus_bins:
            i = int(rng.choice(B, p=probs))
        else:
            i = int(rng.integers(0, B))

        # choose partner bin j
        if partner_mode == "near":
            d2 = np.sum((centroids - centroids[i])**2, axis=1)
            d2[i] = np.inf
            # closer => larger probability, using softmax on negative distances
            pj = softmax(-d2 / 0.05)
            bj = int(rng.choice(B, p=pj))
        elif partner_mode == "cost":
            bj = int(rng.choice(B, p=probs))
            if bj == i:
                continue
        else:  # "uniform"
            bj = int(rng.integers(0, B))
            if bj == i:
                continue

        # pick points and do swap as before
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

def plot_bins_3d(Z, bins, *, savepath=None, dpi=200, max_bins=40, title="",
                 elev=45, azim=45, lw=1.2, line_alpha=0.7, s=20, point_alpha=0.8):
    """
    3D scatter of points colored by bin membership, with MST edges within each bin.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    Z = np.asarray(Z)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    colors = sns.color_palette("tab20", n_colors=min(len(bins), max_bins))

    # tight axis limits around the data
    margin = 0.01
    lims = []
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo, hi = Z[:, dim].min(), Z[:, dim].max()
        pad = (hi - lo) * margin
        setter(lo - pad, hi + pad)
        lims.append(hi - lo + 2 * pad)
    ax.set_box_aspect(lims)

    for b, idx in enumerate(bins[:max_bins]):
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue
        pts = Z[idx]
        c = colors[b % len(colors)]

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=c, s=s, alpha=point_alpha, linewidths=0)

        if idx.size >= 2:
            D = squareform(pdist(pts))
            mst = minimum_spanning_tree(D).tocoo()
            for i, j in zip(mst.row, mst.col):
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    color=c, lw=lw, alpha=line_alpha,
                )

    ax.set_xlabel("Z₁")
    ax.set_ylabel("Z₂")
    ax.set_zlabel("Z₃")
    if title:
        ax.set_title(title)
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig, ax

def make_bin_plots(fig_dir):
    _, _, Z = sample_XYZ_circular(1000, 0, 0, rng=None)
    utility.plot_Z_bins_connected(Z, sa_bins(Z, 8, steps=100_000), savepath=fig_dir / "example_3_sa_bins.png")
    utility.plot_Z_bins_connected(Z, fixed_bins_nd(Z, 0.5), savepath=fig_dir / "example_3_fixed_bins.png")

    _, _, Z3d = sample_XYZ_gaussian3d(1000, 0, 0.5, rng=None)
    plot_bins_3d(Z3d, sa_bins(Z3d, 8), savepath=fig_dir / "example_3_3d_sa_bins.png", title="SA bins, 3D Gaussian Z")
    plot_bins_3d(Z3d, fixed_bins_nd(Z3d, 0.5), savepath=fig_dir / "example_3_3d_fixed_bins.png", title="Fixed bins (width=0.5), 3D Gaussian Z")

def main(recompute, sims, bin_plots_only):
    ns = np.asarray([100, 250, 500, 750, 1000, 1250])
    bin_sizes = np.asarray([4, 8, 12])
    grid_sizes = np.asarray([0.1, 0.5, 1, 1.5])
    grid_sizes_3d = np.asarray([0.25, 0.5, 1, 2])

    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if bin_plots_only:
        make_bin_plots(fig_dir)
        return

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    mc_reps = 250
    p_val_mc_reps = 250

    sim_configs = {
        "2d_sizing": (
            data_dir / "example_3_2d_sizing.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ_circular(n, 0.5, 5.0, rng=rng),
            [lambda Z, n=n, size=size: sa_bins(Z, size) for size in bin_sizes for n in ns],
            T_universal,
            bin_sizes,
            lambda size: f"m = {size}",
            "example_3_2d_sizing.png",
            "Power",
            None,
        ),
        "2d_fixed": (
            data_dir / "example_3_2d_fixed.npy",
            [n for _ in grid_sizes for n in ns],
            lambda n, rng: sample_XYZ_circular(n, 0.5, 5.0, rng=rng),
            [lambda Z, n=n, size=size: fixed_bins_nd(Z, size) for size in grid_sizes for n in ns],
            T_universal,
            grid_sizes,
            lambda size: f"grid size = {size}",
            "example_3_2d_fixed.png",
            "Power",
            None,
        ),
        "2d_sizing_validity": (
            data_dir / "example_3_2d_sizing_validity.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ_circular(n, 0.0, 0.0, rng=rng),
            [lambda Z, n=n, size=size: sa_bins(Z, size) for size in bin_sizes for n in ns],
            T_universal,
            bin_sizes,
            lambda size: f"m = {size}",
            "example_3_2d_sizing_validity.png",
            "Type I error rate",
            None,
        ),
        "2d_fixed_validity": (
            data_dir / "example_3_2d_fixed_validity.npy",
            [n for _ in grid_sizes for n in ns],
            lambda n, rng: sample_XYZ_circular(n, 0.0, 0.0, rng=rng),
            [lambda Z, n=n, size=size: fixed_bins_nd(Z, size) for size in grid_sizes for n in ns],
            T_universal,
            grid_sizes,
            lambda size: f"grid size = {size}",
            "example_3_2d_fixed_validity.png",
            "Type I error rate",
            None,
        ),
        "3d_sizing": (
            data_dir / "example_3_3d_sizing.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ_gaussian3d(n, 0.5, 0.5, rng=rng),
            [lambda Z, n=n, size=size: sa_bins(Z, size) for size in bin_sizes for n in ns],
            T_universal,
            bin_sizes,
            lambda size: f"m = {size}",
            "example_3_3d_sizing.png",
            "Power",
            None,
        ),
        "3d_fixed": (
            data_dir / "example_3_3d_fixed.npy",
            [n for _ in grid_sizes_3d for n in ns],
            lambda n, rng: sample_XYZ_gaussian3d(n, 0.5, 0.5, rng=rng),
            [lambda Z, n=n, size=size: fixed_bins_nd(Z, size) for size in grid_sizes_3d for n in ns],
            T_universal,
            grid_sizes_3d,
            lambda size: f"grid width = {size}",
            "example_3_3d_fixed.png",
            "Power",
            None,
        ),
        "3d_sizing_validity": (
            data_dir / "example_3_3d_sizing_validity.npy",
            [n for _ in bin_sizes for n in ns],
            lambda n, rng: sample_XYZ_gaussian3d(n, 0.0, 0.5, rng=rng),
            [lambda Z, n=n, size=size: sa_bins(Z, size) for size in bin_sizes for n in ns],
            T_universal,
            bin_sizes,
            lambda size: f"m = {size}",
            "example_3_3d_sizing_validity.png",
            "Type I error rate",
            None,
        ),
        "3d_fixed_validity": (
            data_dir / "example_3_3d_fixed_validity.npy",
            [n for _ in grid_sizes_3d for n in ns],
            lambda n, rng: sample_XYZ_gaussian3d(n, 0.0, 0.5, rng=rng),
            [lambda Z, n=n, size=size: fixed_bins_nd(Z, size) for size in grid_sizes_3d for n in ns],
            T_universal,
            grid_sizes_3d,
            lambda size: f"grid width = {size}",
            "example_3_3d_fixed_validity.png",
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

    make_bin_plots(fig_dir)


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
        choices=[
            "2d_sizing", "2d_fixed", "2d_sizing_validity", "2d_fixed_validity",
            "3d_sizing", "3d_fixed", "3d_sizing_validity", "3d_fixed_validity",
        ],
        default=[
            "2d_sizing", "2d_fixed", "2d_sizing_validity", "2d_fixed_validity",
            "3d_sizing", "3d_fixed", "3d_sizing_validity", "3d_fixed_validity",
        ],
        help="Which simulations to run (default: all)."
    )
    parser.add_argument(
        "--bin-plots-only",
        action="store_true",
        help="Only generate binning visualisation plots; skip all simulations."
    )
    args = parser.parse_args()
    main(args.recompute, args.sims, args.bin_plots_only)
