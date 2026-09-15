import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from report_content_review_ai import agreement
from content_review_f import DIMENSIONS


def test_agreement_counts_unique_outputs_not_methods_and_ties_as_sets():
    cases = [{'case': 'F001', 'labels': {'A': 1, 'B': 3}, 'methods': {'E': 'A', 'C': 'A', 'FixedRandom': 'B'}}]
    first = {('F001', a): {**dict.fromkeys(DIMENSIONS, 'correct'), 'best': 'A;B'} for a in 'AB'}
    second = {('F001', a): {**dict.fromkeys(DIMENSIONS, 'wrong'), 'best': 'B;A'} for a in 'AB'}
    result = agreement(cases, first, second)
    assert result['object']['n_unique_outputs'] == 2
    assert result['object']['exact_agreement'] == 0
    assert result['object']['unweighted_kappa'] == 0
    assert result['best']['exact_set_agreement'] == 1


def test_constant_agreement_kappa_undefined_not_invented():
    cases = [{'case': 'F001', 'labels': {'A': 1}}]
    labels = {('F001', 'A'): {**dict.fromkeys(DIMENSIONS, 'correct'), 'best': 'all_wrong'}}
    result = agreement(cases, labels, labels)
    assert result['object']['exact_agreement'] == 1
    assert result['object']['unweighted_kappa'] is None
