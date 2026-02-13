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
               progress=True):
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

    ps_list = []
    it = range(B)
    if progress and B > 1:
        it = tqdm(it, desc="Batches", position=0, leave=True, ncols=80)
        tqdm_pos = 1

    for b in it:
        nb = int(get(n, b))
        sampling_fnb = get(sampling_fn, b)
        binning_fnb = get(binning_fn, b)
        Tb = get(T, b)
        mc_repsb = int(get(mc_reps, b))
        p_val_mc_repsb = int(get(p_val_mc_reps, b))

        ps = np.empty(mc_repsb)
        inner = range(mc_repsb)
        if progress:
            inner = tqdm(inner, desc="Simulating p-value distribution", position=tqdm_pos, leave=False, ncols=80)

        for i in inner:
            X, Y, Z = sampling_fnb(nb)
            bins = binning_fnb(Z)
            ps[i] = lpt.test_ci(X, Y, Z, bins, Tb, mc_reps=p_val_mc_repsb)

        ps_list.append(ps)

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
        # your adaptive bin-size choice in example1.py:
        # bin_size = np.floor(2 * np.power(ns, 1.0 + param)).astype(int)
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
