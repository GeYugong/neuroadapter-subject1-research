import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def module(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def test_checkpoint_whitelist_rejects_unlisted_before_io():
    m = module("diagnose_generalization_v2")
    assert set(m.WEIGHTS) == {106250,159375,239063}
    with pytest.raises(ValueError, match="whitelist"):
        m.snapshot(SimpleNamespace(paths={}), 500)


def test_segment_correlation_matches_independent_numpy():
    m = module("audit_signal_generalization_v2")
    rng = np.random.default_rng(1)
    x,y = rng.normal(size=(2,29))
    starts, lengths = np.array([0,9,20]), np.array([9,11,9])
    result = m.spatial_corr(x,y,starts,lengths)
    expected = [np.corrcoef(x[a:a+n],y[a:a+n])[0,1] for a,n in zip(starts,lengths)]
    np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-12)


def test_constant_segment_is_explicitly_undefined():
    m = module("audit_signal_generalization_v2")
    assert np.isnan(m.spatial_corr(np.ones(4),np.arange(4),np.array([0]),np.array([4]))[0])


def test_leave_image_out_baseline_excludes_all_three_repeats():
    reps = np.arange(5*3*7,dtype=float).reshape(5,3,7)
    means = reps.mean(1)
    total = means.sum(0)
    for i in range(5):
        np.testing.assert_allclose((total-means[i])/4, np.delete(reps,i,axis=0).mean((0,1)))


def test_paired_change_uses_image_as_unit():
    m = module("summarize_generalization_v2")
    rows = []
    for split in ("train","validation"):
        for k,step in enumerate((106250,159375,239063)):
            for i in range(32):
                rows.append({"split":split,"checkpoint":step,"image_id":i,
                    **{name: i/100+k*.01 for name in
                       ("correct_clip","wrong_clip","advantage_clip","correct_pixcorr","correct_ssim")}})
    result = m.paired_changes(rows)
    for comparison in result.values():
        assert comparison["correct_clip"]["n_images"] == 32
        assert comparison["correct_clip"]["mean"] == pytest.approx(.01)
