"""Report AI labels separately from the still-pending human assessment."""
import argparse
import json
import shutil
from pathlib import Path

from content_review_f import DIMENSIONS, VALUES, analyze, validate_ratings
from prepare_semantic_rerank import sha, write
from run_residual_rerank import read


def agreement(cases, first, second):
    result = {}
    # Count each displayed candidate once, not once per selector that picked it.
    for dimension in DIMENSIONS:
        confusion = {a: {b: 0 for b in VALUES} for a in VALUES}
        for case in cases:
            for label in case['labels']:
                key = case['case'], label
                confusion[first[key][dimension]][second[key][dimension]] += 1
        n = sum(sum(r.values()) for r in confusion.values())
        observed = sum(confusion[v][v] for v in VALUES)/n
        expected = sum(sum(confusion[v].values())*sum(confusion[a][v] for a in VALUES) for v in VALUES)/(n*n)
        result[dimension] = {'n_unique_outputs': n, 'confusion': confusion,
                             'exact_agreement': observed,
                             'unweighted_kappa': (observed-expected)/(1-expected) if expected < 1 else None}
    same = 0
    for case in cases:
        key = case['case'], next(iter(case['labels']))
        a, b = first[key]['best'], second[key]['best']
        same += set(a.split(';')) == set(b.split(';'))
    result['best'] = {'n_cases': len(cases), 'exact_set_agreement': same/len(cases)}
    return result


def run(source, out, manifest):
    lock = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if lock['reviewer_type'] != 'AI' or len(lock['raters']) != 2:
        raise ValueError('expected two AI raters')
    paths = []
    for r in lock['raters']:
        p = manifest.parent/r['ratings_file']
        if sha(p) != r['sha256']:
            raise ValueError('ratings changed after lock')
        paths.append(p)
    protocol = json.loads((source/'protocol.json').read_text())
    if sha(source/'sealed_mapping.json') != protocol['mapping_sha256']:
        raise ValueError('frozen mapping changed')
    cases = json.loads((source/'sealed_mapping.json').read_text())['cases']
    if len(cases) != 160:
        raise ValueError('expected fixed160')
    labels = [validate_ratings(read(p), cases) for p in paths]
    out.mkdir(parents=True, exist_ok=False)
    for j, p in enumerate(paths, 1):
        shutil.copyfile(p, out/f'rater-{j}.csv')
    shutil.copyfile(manifest, out/'rating_commitments.json')
    result = {'reviewer_type': 'AI', 'human_review_complete': False,
              'scope': 'two context-separated agents inheriting the same model; correlated judgments, not independent humans',
              'raters': {f'rater-{j}': analyze(cases, label) for j, label in enumerate(labels, 1)},
              'agreement': agreement(cases, *labels),
              'no_adjudication_or_pooling': True, 'automatic_acceptance': False}
    write(out/'summary.json', result)
    write(out/'revealed_cases.json', {'cases': cases})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    run(args.source, args.out, args.manifest)
