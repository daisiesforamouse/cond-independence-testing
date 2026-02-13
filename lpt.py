from joblib import Parallel, delayed

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import numpy as np

def swap(X, bins, perms = None):
    if perms == None:
        perms = [np.random.permutation(b) for b in bins]

    X_swapped = np.copy(X)
    for bin, perm in zip(bins, perms):
        for i, j in zip(bin, perm):
            X_swapped[j] = X[i]

    return X_swapped

def test_ci(X, Y, Z,
            bins,
            T,
            mc_reps=1000,
            n_jobs=-2,
            seed = None):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=mc_reps, dtype=np.uint32)

    def one_mc_rep(s):
        np.random.seed(int(s))
        return T(swap(X, bins), Y, Z, bins)

    perm_stats = np.array(Parallel(n_jobs=n_jobs)(delayed(one_mc_rep)(s) for s in seeds))

    p = np.mean(perm_stats >= T(X, Y, Z, bins))
    return p
