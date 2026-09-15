"""Train-only preprocessing and explicitly oriented retrieval diagnostics."""
import hashlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def partition(ids, namespace, fit_count):
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate IDs')
    ordered = sorted(ids, key=lambda i: (hashlib.sha256(f'{namespace}:{i}'.encode()).hexdigest(), i))
    return ordered[:fit_count], ordered[fit_count:]


def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(a).all() or np.any(norms == 0):
        raise ValueError('non-finite or zero feature')
    return a / norms


def retrieval(prediction, target):
    if prediction.shape != target.shape or len(target) < 2:
        raise ValueError('aligned prediction and target required')
    similarity = normalize(prediction) @ normalize(target).T
    truth = np.diag(similarity)
    below = (similarity < truth[:, None]).sum(1)
    equal = (similarity == truth[:, None]).sum(1)
    above = (similarity > truth[:, None]).sum(1)
    return {'cosine': truth, 'forward_two_way': (below + .5*(equal-1))/(len(target)-1),
            'top1': np.clip((1-above)/equal, 0, 1), 'top5': np.clip((5-above)/equal, 0, 1)}


def fit_preprocessing(x, dimension, spec):
    scaler = StandardScaler().fit(x)
    keep = scaler.var_ > 0
    if keep.sum() < dimension:
        raise ValueError('too few nonconstant input features')
    z = scaler.transform(x)[:, keep]
    pca = PCA(n_components=dimension, whiten=False, svd_solver='randomized',
              random_state=spec['pca_seed'], iterated_power=spec['pca_iterated_power'],
              n_oversamples=spec['pca_n_oversamples'])
    pca.fit(z)
    return scaler, keep, pca, pca.transform(z)


def transform(x, scaler, keep, pca):
    return pca.transform(scaler.transform(x)[:, keep])


def choose(rows):
    return sorted(rows, key=lambda r: (-r['forward_two_way'], r['dimension'], -r['alpha']))[0]
