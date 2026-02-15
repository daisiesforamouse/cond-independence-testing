from joblib import Parallel, delayed
import joblib

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


import numpy as np

def swap(X, bins, *, perms=None, rng=None):
    n = len(X)

    if perms is None:
        if rng is None:
            rng = np.random.default_rng()
        perms = [rng.permutation(b) for b in bins]

    idx = np.arange(n)
    for b, p in zip(bins, perms):
        b = np.asarray(b, dtype=int)
        p = np.asarray(p, dtype=int)
        idx[p] = b

    return X[idx]

def test_ci(X, Y, Z,
            bins,
            T,
            mc_reps=1000,
            batch_size=20,
            n_jobs=-2,
            seed = None):
    if isinstance(seed, np.random.SeedSequence):
        root_ss = seed
    else:
        root_ss = np.random.SeedSequence(seed)

    rep_ss = root_ss.spawn(int(mc_reps))

    rep_ss_batches = [rep_ss[i:i + batch_size] for i in range(0, mc_reps, batch_size)]

    def one_batch(ss_batch):
        out = np.empty(len(ss_batch), dtype=float)
        for j, ss in enumerate(ss_batch):
            r = np.random.default_rng(ss)
            out[j] = T(swap(X, bins, rng=r), Y, Z, bins)
        return out

    perm_stats = np.concatenate(
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(one_batch)(sb) for sb in rep_ss_batches
        )
    )

    T_original = T(X, Y, Z, bins)
    return float(np.mean(perm_stats >= T_original))
