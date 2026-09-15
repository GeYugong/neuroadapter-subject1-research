import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('rerank_core', Path(__file__).parents[1]/'scripts/semantic_rerank_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_selector_interface_has_no_gt_and_ties_choose_first():
    assert list(inspect.signature(core.select_candidates).parameters) == ['predictions', 'candidates']
    p = np.array([[1., 0.], [0., 1.]])
    g = np.array([[[1.,0.],[2.,0.]], [[1.,0.],[0.,1.]]])
    indices, scores = core.select_candidates(p,g)
    assert indices.tolist() == [0,1]
    assert np.allclose(scores, [[1,1],[0,1]])


def test_selector_rejects_invalid_inputs():
    for bad in [np.zeros((2,3)), np.full((2,3),np.nan)]:
        with pytest.raises(ValueError):
            core.select_candidates(bad,np.ones((2,8,3)))
    with pytest.raises(ValueError):
        core.select_candidates(np.ones((2,3)),np.ones((3,8,3)))


def test_mismatch_is_reproducible_permutation_without_fixed_points():
    for seed in [20260931,20260932,20260933,20260934,20260935]:
        p = core.derangement(500,seed)
        assert sorted(p) == list(range(500))
        assert np.all(p != np.arange(500))
        assert np.array_equal(p,core.derangement(500,seed))
