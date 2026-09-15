import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('semantic_core', Path(__file__).parents[1]/'scripts/semantic_probe_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_split_stable_unique_and_disjoint():
    a, b = core.partition(list(range(20)), 'test', 15)
    assert (a,b) == core.partition(list(reversed(range(20))), 'test', 15)
    assert not set(a)&set(b) and len(a)==15 and len(b)==5
    with pytest.raises(ValueError): core.partition([1,1], 'test', 1)


def test_retrieval_perfect_and_tied_mean():
    gt = np.eye(6)
    perfect = core.retrieval(gt, gt)
    assert all(np.allclose(v,1) for v in perfect.values())
    mean = core.retrieval(np.ones((6,6)), gt)
    assert np.allclose(mean['forward_two_way'], .5)
    assert np.allclose(mean['top1'], 1/6) and np.allclose(mean['top5'], 5/6)


def test_retrieval_matches_explicit_forward_formula():
    rng=np.random.default_rng(12)
    p=rng.normal(size=(9,7)); g=rng.normal(size=(9,7))
    score=core.normalize(p)@core.normalize(g).T
    expected=[sum(float(score[i,i]>score[i,j])+.5*float(score[i,i]==score[i,j]) for j in range(9) if i!=j)/8 for i in range(9)]
    assert np.allclose(core.retrieval(p,g)['forward_two_way'],expected)


def test_no_holdout_preprocessing_fit():
    x=np.arange(120,dtype=float).reshape(20,6); x[:,0]=3
    settings={'pca_seed':1,'pca_iterated_power':4,'pca_n_oversamples':2}
    scaler,keep,pca,z=core.fit_preprocessing(x,2,settings)
    mean=scaler.mean_.copy(); components=pca.components_.copy()
    core.transform(np.full((4,6),1e6),scaler,keep,pca)
    assert np.array_equal(mean,scaler.mean_) and np.array_equal(components,pca.components_)
    assert not keep[0] and np.allclose(z,core.transform(x,scaler,keep,pca))


def test_ties_prefer_smaller_pca_then_larger_alpha():
    rows=[{'forward_two_way':.8,'dimension':d,'alpha':a} for d in [1024,256] for a in [1,1000]]
    assert core.choose(rows)=={'forward_two_way':.8,'dimension':256,'alpha':1000}
