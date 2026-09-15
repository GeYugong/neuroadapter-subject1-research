import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from run_residual_rerank import residual_select


def test_raw_prediction_is_subtracted_before_normalization():
    p=np.array([[2.,1.]])
    mu=np.array([1.,0.])
    g=np.array([[[1.,0.],[0.,5.]]])
    chosen,common,residual=residual_select(p,mu,g)
    assert chosen.tolist()==[0]
    assert np.array_equal(common,[[1.,0.]])
    assert np.array_equal(residual,[[1.,1.]])


def test_zero_residual_ties_first_without_error():
    chosen,_,scores=residual_select(np.array([[.3,.4]]),np.array([.3,.4]),np.ones((1,8,2)))
    assert chosen.tolist()==[0] and not scores.any()


def test_decomposition_matches_raw_dot_and_invalid_rejected():
    rng=np.random.default_rng(2);p=rng.normal(size=(4,3));mu=np.array([.2,.3,.4]);g=rng.normal(size=(4,8,3))
    _,a,b=residual_select(p,mu,g)
    assert np.allclose(a+b,np.einsum('nd,nkd->nk',p,g/np.linalg.norm(g,axis=-1,keepdims=True)))
    with pytest.raises(ValueError):residual_select(p,mu,np.zeros_like(g))
