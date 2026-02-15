from joblib import Parallel, delayed, wrap_non_picklable_objects
import lpt

from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def p_val_dist(n,
               sampling_fn,
               binning_fn,
               T,
               mc_reps=1000,
               p_val_mc_reps=1000,
               progress=True,
               n_jobs=-1,
               base_seed=0):
    """
    Computes a matrix of p-values where each entry corresponds to a Monte Carlo simulation. Each row p[i]
    corresponds to the settings enforced by the i-th values in the arguments.

    Each argument may be:
      - a scalar / single callable (treated as constant across batches), or
      - a list/array of length B providing per-batch values.

    For each batch b, runs mc_reps simulations using:
      n[b], sampling_fn[b], binning_fn[b], T[b], p_val_mc_reps[b], mc_reps[b]

    Returns
    -------
    ps_list : list of 1D numpy arrays
        ps_list[b] is the p-value vector for batch b of length mc_reps[b].
    """
    def is_seq(x):
        return isinstance(x, (list, tuple, np.ndarray))

    # determine number of batches B
    candidates = [n, sampling_fn, binning_fn, T, mc_reps, p_val_mc_reps]
    lengths = [len(x) for x in candidates if is_seq(x)]
    B = max(lengths) if lengths else 1

    def get(x, b):
        if is_seq(x):
            if len(x) != B:
                raise ValueError(f"Length mismatch: expected {B}, got {len(x)} for {type(x)}")
            return x[b]
        return x

    root_ss = np.random.SeedSequence(base_seed)

    def one_rep(rep_ss, nb, sampling_fnb, binning_fnb, Tb, p_val_mc_repsb):
        # Derive two independent seeds from the replicate SeedSequence:
        # one for data generation, one for permutation test.
        data_ss, test_ss = rep_ss.spawn(2)
        data_rng = np.random.default_rng(data_ss)
        test_seed = int(test_ss.generate_state(1, dtype=np.uint32)[0])

        X, Y, Z = sampling_fnb(nb, data_rng)
        bins = binning_fnb(Z)

        return lpt.test_ci(
            X, Y, Z, bins, Tb,
            mc_reps=int(p_val_mc_repsb),
            seed=test_seed,
            n_jobs=1,
        )

    ps_list = []
    outer = range(B)
    if progress and B > 1:
        outer = tqdm(outer, desc="Batches", ncols=80)

    for b in outer:
        nb = int(get(n, b))
        mc_repsb = int(get(mc_reps, b))
        p_val_mc_repsb = int(get(p_val_mc_reps, b))

        sampling_fnb = wrap_non_picklable_objects(get(sampling_fn, b))
        binning_fnb = wrap_non_picklable_objects(get(binning_fn, b))
        Tb = wrap_non_picklable_objects(get(T, b)) if is_seq(T) else T

        # Per-batch seed sequence, then per-rep seed sequences
        batch_ss = root_ss.spawn(1)[0] if B == 1 else root_ss.spawn(B)[b]
        rep_ss_list = batch_ss.spawn(mc_repsb)

        it = range(mc_repsb)
        if progress:
            it = tqdm(it, desc="Simulating p-value distribution", leave=False, ncols=80)

        ps = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(one_rep)(rep_ss_list[i], nb, sampling_fnb, binning_fnb, Tb, p_val_mc_repsb)
            for i in it
        )
        ps_list.append(np.asarray(ps, dtype=float))

    return ps_list

def rejection_rates(ps):
    """
    Computes type I error for p-value matrices output from p_val_dist
    """
    return np.apply_along_axis(lambda x: np.mean(x < 0.05), 1, ps)

def plot_rejection(rejection_rates,
                   x_params,
                   params,
                   get_param_label,
                   alpha=0.05,
                   ax=None,
                   title="",
                   x_axis="",
                   y_axis="",
                   savepath=None,
                   dpi=200):
    """
    type_I_adaptive: array-like of length len(x_params)*len(params)
    Assumes the batches were ordered like:
        for param in params:
            for x_param in x_params:
                    ...
    """
    x_params = np.asarray(x_params)
    params = np.asarray(params)
    rejection_rates = np.asarray(rejection_rates)

    E, N = len(params), len(x_params)
    if rejection_rates.size != E * N:
        raise ValueError(f"Expected {E*N} entries, got {rejection_rates.size}")

    Y = rejection_rates.reshape(E, N)  # rows=exp (bin setting), cols=n

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    for i, param in enumerate(params):
        ax.plot(x_params, Y[i], marker="o", linewidth=2, label=get_param_label(param))

    ax.axhline(alpha, color="k", linestyle="--", linewidth=1, label=f"alpha={alpha}")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_xticks(x_params)
    ax.set_ylim(0, max(0.1, float(np.max(rejection_rates)) * 1.1))
    ax.legend(title=title, fontsize=9)
    ax.grid(True, alpha=0.3)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_Z_bins_connected(Z, bins, *, ax=None, s=35, alpha=0.9, lw=1.5,
                          palette="tab20", show_points=True):
    """
    Scatter plot of 2D points Z, with points belonging to the same bin connected.

    Parameters
    ----------
    Z : (n, 2) array-like
    bins : list of list[int] (or arrays)
        bins[b] contains indices of points in that bin.
    ax : matplotlib Axes or None
    s, alpha : scatter styling
    lw : line width for connections
    palette : seaborn palette name or list of colors
    show_points : bool
        If True, draw points as well as connecting lines.

    Returns
    -------
    fig, ax
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"Expected Z of shape (n, 2), got {Z.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    colors = sns.color_palette(palette, n_colors=max(1, len(bins)))

    if show_points:
        sns.scatterplot(x=Z[:, 0], y=Z[:, 1], ax=ax, s=s, alpha=alpha, color="k", linewidth=0)

    for b, idx in enumerate(bins):
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue

        pts = Z[idx]

        # draw colored points for this bin (on top)
        if show_points:
            sns.scatterplot(x=pts[:, 0], y=pts[:, 1], ax=ax, s=s, alpha=alpha,
                            color=colors[b], linewidth=0)

        # connect points within the bin
        if idx.size >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color=colors[b], lw=lw, alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    return fig, ax
