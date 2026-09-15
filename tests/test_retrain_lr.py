import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from retrain_lr_core import U, NODES, lr_at, lr_state


def test_constant_and_reference_epochs():
    assert [n*16/8500 for n in NODES] == [100,200,300]
    assert all(lr_at('T1',i)==3e-5 for i in range(U))


def test_cosine_endpoints_monotonic_and_no_restart():
    values=[lr_at('T2',i) for i in range(U)]
    assert values[0]==1e-4 and values[-1]==1e-5
    assert all(a>=b for a,b in zip(values,values[1:]))
    assert lr_state('T2',U)['next_lr'] is None


def test_microsteps_do_not_advance_and_resume_keeps_real_period():
    cursor=10
    assert [lr_at('T2',cursor) for _ in range(2)] == [lr_at('T2',10)]*2
    assert [lr_at('T2',i) for i in range(20)] == [lr_at('T2',i) for i in range(10)]+[lr_at('T2',i) for i in range(lr_state('T2',10)['completed_updates'],20)]
    assert lr_at('T2',19)>0.000099
    with pytest.raises(ValueError): lr_at('T2',U)
