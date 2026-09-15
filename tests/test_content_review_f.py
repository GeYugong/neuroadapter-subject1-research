import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from content_review_f import anonymous_choices, sample_ids, validate_ratings, analyze, DIMENSIONS


def test_sampling_fixed_excludes_old_and_order_independent():
    ids = list(range(500))
    got = sample_ids(ids, ids[:32])
    assert len(got) == len(set(got)) == 160
    assert not set(got) & set(ids[:32])
    assert got == sample_ids(ids[::-1], ids[:32])


def test_duplicate_candidates_have_one_label():
    selected, methods, unique = anonymous_choices(123, 4, 4, 'frozen')
    assert methods['C'] == methods['E']
    assert len(unique) == len(set(selected.values()))
    assert anonymous_choices(123, 4, 4, 'frozen') == (selected, methods, unique)


def fixture():
    cases = [{'case':'F001','labels':{'A':2,'B':4},'methods':{'C':'A','E':'A','FixedRandom':'B'}}]
    rows = [{'case':'F001','output':label, **dict.fromkeys(DIMENSIONS,'correct'), 'best':'A'} for label in 'AB']
    return cases, rows


def test_no_incomplete_or_conflicting_reviews():
    cases, rows = fixture()
    with pytest.raises(ValueError): validate_ratings(rows[:1], cases)
    with pytest.raises(ValueError): validate_ratings(rows+rows[:1], cases)
    rows[0]['scene']=''
    with pytest.raises(ValueError): validate_ratings(rows, cases)
    rows[0]['scene']='uncertain'; rows[1]['best']='B'
    with pytest.raises(ValueError): validate_ratings(rows, cases)


def test_shared_choices_receive_identical_labels_and_uncertainty_reported():
    cases, rows = fixture()
    rows[0]['scene']='uncertain'
    result = analyze(cases, validate_ratings(rows, cases))
    assert result['absolute_counts']['E'] == result['absolute_counts']['C']
    assert result['E_minus']['C']['scene']['both_judgable_difference']['n']==0
    assert result['E_minus']['C']['preference']['both_best']==1
