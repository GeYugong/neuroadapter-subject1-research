"""Deployable ranking has no target-image argument or ground-truth access."""
import numpy as np


def select_candidates(predictions, candidates):
    """Accept [N,D] predictions and [N,K,D] candidates in index order."""
    p = np.asarray(predictions, dtype=np.float64)
    g = np.asarray(candidates, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 3 or g.shape[0] != p.shape[0] or g.shape[2] != p.shape[1]:
        raise ValueError('expected aligned [N,D] and [N,K,D]')
    if g.shape[1] == 0 or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('empty or non-finite features')
    pn = np.linalg.norm(p, axis=-1, keepdims=True)
    gn = np.linalg.norm(g, axis=-1, keepdims=True)
    if np.any(pn == 0) or np.any(gn == 0):
        raise ValueError('zero feature norm')
    scores = np.einsum('nd,nkd->nk', p / pn, g / gn)
    return np.argmax(scores, axis=1), scores


def derangement(count, seed):
    if count < 2:
        raise ValueError('derangement requires at least two entries')
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(count)
        if np.all(permutation != np.arange(count)):
            return permutation
