"""Prepare a finite human review; reveal only after complete labels are supplied."""
import argparse
import csv
import hashlib
import html
import json
import secrets
import shutil
from pathlib import Path

import numpy as np

from prepare_semantic_rerank import sha, write
from run_residual_rerank import read, save_csv

NAMESPACE = 'content-review-f-fixed160-v1'
DIMENSIONS = ('object', 'action_relation', 'scene', 'count_layout')
VALUES = ('correct', 'partial', 'wrong', 'uncertain')


def digest(domain, value):
    return hashlib.sha256(f'{NAMESPACE}|{domain}|{value}'.encode()).hexdigest()


def sample_ids(ids, excluded, count=160):
    if len(ids) != len(set(ids)) or not set(excluded) <= set(ids):
        raise ValueError('invalid population')
    pool = set(ids) - set(excluded)
    if len(pool) < count:
        raise ValueError('insufficient unreviewed images')
    return sorted(pool, key=lambda i: (digest('sample', i), i))[:count]


def anonymous_choices(image_id, c, e, salt):
    random_index = int(digest('random-candidate', image_id), 16) % 8
    selected = {'C': c, 'E': e, 'FixedRandom': random_index}
    if any(k not in range(8) for k in selected.values()):
        raise ValueError('invalid candidate index')
    unique = sorted(set(selected.values()), key=lambda k: digest(salt, f'{image_id}:{k}'))
    labels = {k: chr(65+j) for j, k in enumerate(unique)}
    return selected, {method: labels[k] for method, k in selected.items()}, unique


def prepare(root):
    import h5py
    from PIL import Image

    base = root/'runs/diagnostics'
    d, e = base/'semantic-rerank-d-v1', base/'semantic-rerank-e-residual-v1'
    out = base/NAMESPACE
    if out.exists():
        raise RuntimeError('frozen package exists; do not overwrite or resample')
    plan = json.loads((d/'plan.json').read_text())
    eplan = json.loads((e/'plan.json').read_text())
    for name, expected in eplan['source_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError(f'changed frozen input: {name}')
    ids = sample_ids(plan['image_ids'], plan['visual_ids'])
    assert len(plan['image_ids']) == 500 and len(plan['visual_ids']) == 32
    csel = {int(r['image_id']): int(r['candidate_index']) for r in read(d/'selector_indices.csv') if r['method']=='C-selected'}
    esel = {int(r['image_id']): int(r['candidate_index']) for r in read(e/'selector_indices.csv') if r['method']=='E-selected'}
    if set(csel) != set(plan['image_ids']) or set(esel) != set(csel):
        raise ValueError('selector ID mismatch')
    inventory = json.loads((d/'candidate_manifest.json').read_text())
    decode = root/inventory['source']
    if sha(decode) != inventory['source_sha256']:
        raise ValueError('decode manifest changed')
    records = {(r['image_id'], r['candidate_index']): r for r in inventory['records']}
    out.mkdir()
    pack = out/'reviewer_pack'
    (pack/'images').mkdir(parents=True)
    salt = secrets.token_hex(32)
    mapping, blank, sections = [], [], []
    stimuli = root/'data/raw/nsd/stimuli/nsd_stimuli.hdf5'
    with h5py.File(stimuli, 'r') as h5:
        for number, i in enumerate(ids, 1):
            case = f'F{number:03d}'
            selected, methods, unique = anonymous_choices(i, csel[i], esel[i], salt)
            images = [(label, k) for label, k in zip('ABC', unique)]
            Image.fromarray(h5['imgBrick'][i]).convert('RGB').save(pack/'images'/f'{case}-GT.png')
            mapping.append({'case': case, 'image_id': i, 'methods': methods, 'indices': selected,
                            'labels': {label: k for label, k in images}})
            tiles = [f'<figure><figcaption>GT</figcaption><img src="images/{case}-GT.png" alt="{case} GT"></figure>']
            for label, k in images:
                record = records[i, k]
                source = decode.parent/record['path']
                if sha(source) != record['sha256']:
                    raise ValueError(f'candidate changed: {i}/{k}')
                shutil.copyfile(source, pack/'images'/f'{case}-{label}.png')
                tiles.append(f'<figure><figcaption>{label}</figcaption><img src="images/{case}-{label}.png" alt="{case} {label}"></figure>')
                blank.append({'case': case, 'output': label, **{key: '' for key in DIMENSIONS}, 'best': '', 'notes': ''})
            sections.append(f'<section id="{case}"><h2>{case}</h2><div class="tiles">'+''.join(tiles)+'</div></section>')
    # The reviewer package deliberately contains neither mapping nor scores.
    mapping_path = out/'sealed_mapping.json'
    write(mapping_path, {'salt': salt, 'cases': mapping})
    save_csv(pack/'ratings.csv', blank)
    instructions = root/'repo/docs/CONTENT_REVIEW_F_RATER_GUIDE.md'
    shutil.copyfile(instructions, pack/'README.md')
    style = 'body{font:16px system-ui;margin:24px;color:#202124;background:#fff}a{color:#176953}nav{display:flex;flex-wrap:wrap;gap:12px}section{border-top:1px solid #ddd;padding:20px 0}h2{font-size:20px}.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{padding:8px 0}img{width:100%;height:auto;display:block}*{letter-spacing:0}@media(max-width:800px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    def document(title, body):
        return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{style}</style><body><h1>{html.escape(title)}</h1>{body}</body></html>'
    links = []
    for start in range(0, 160, 10):
        name = f'page-{start//10+1:02d}.html'
        links.append(f'<a href="{name}">{start+1}–{start+10}</a>')
        previous = f'<a href="page-{start//10:02d}.html">上一页</a>' if start else ''
        following = f'<a href="page-{start//10+2:02d}.html">下一页</a>' if start < 150 else ''
        body = '<nav><a href="index.html">目录</a>'+previous+following+'</nav>'+''.join(sections[start:start+10])
        (pack/name).write_text(document(f'内容评阅 {start+1}–{start+10}', body), encoding='utf-8')
    (pack/'index.html').write_text(document('内容评阅 · 160图', '<nav>'+''.join(links)+'</nav>'), encoding='utf-8')
    sources = [d/'plan.json', d/'candidate_manifest.json', d/'selector_indices.csv', e/'plan.json', e/'selector_indices.csv']
    write(out/'protocol.json', {'namespace': NAMESPACE, 'status': 'awaiting_human_ratings',
        'sample_rule': 'SHA256(namespace|sample|decimal_image_id), ascending first160 after excluding D32',
        'random_rule': 'int(SHA256(namespace|random-candidate|decimal_image_id),16) modulo8',
        'sample_count': 160, 'excluded_count': 32, 'population_remaining': 468,
        'sample_commitment': digest('sample-ids', ','.join(map(str, ids))),
        'mapping_sha256': sha(mapping_path), 'source_sha256': {str(p.relative_to(root)): sha(p) for p in sources},
        'code_sha256': sha(Path(__file__)), 'guide_sha256': sha(instructions),
        'dimensions': DIMENSIONS, 'values': VALUES, 'bootstrap_seed': 20261002, 'draws': 10000,
        'interval': .975, 'scope': 'developmental human content review; not independent test',
        'decision': 'report absolute rates and paired differences; no automatic pass; no resampling or method changes',
        'unique_outputs': len(blank), 'new_training': 0, 'new_generation': 0,
        'new_feature_extraction': 0, 'standard_test_access': False})
    write(out/'pack_manifest.json', {str(p.relative_to(pack)): sha(p) for p in sorted(pack.rglob('*')) if p.is_file()})
    print(json.dumps({'status': 'awaiting_human_ratings', 'cases': 160, 'unique_outputs': len(blank)}))


def validate_ratings(rows, cases):
    expected = {(c['case'], label) for c in cases for label in c['labels']}
    keys = [(r['case'], r['output']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing, extra or duplicate ratings')
    lookup = dict(zip(keys, rows))
    for case in cases:
        best = set()
        for label in case['labels']:
            row = lookup[case['case'], label]
            if any(row[d] not in VALUES for d in DIMENSIONS):
                raise ValueError('all dimensions need a valid rating, including uncertain')
            best.add(row['best'])
        if len(best) != 1:
            raise ValueError('best must be identical on all rows of a case')
        value = best.pop()
        if value not in ('all_wrong', 'uncertain'):
            chosen = value.split(';')
            if not value or len(chosen) != len(set(chosen)) or not set(chosen) <= set(case['labels']):
                raise ValueError('invalid best labels')
    return lookup


def analyze(cases, lookup):
    summary, comparisons = {}, {}
    for method in ('E', 'C', 'FixedRandom'):
        summary[method] = {}
        for d in DIMENSIONS:
            values = [lookup[c['case'], c['methods'][method]][d] for c in cases]
            summary[method][d] = {v: values.count(v) for v in VALUES}
            summary[method][d]['correct_fraction_all_cases'] = values.count('correct')/len(cases)
    rng = np.random.default_rng(20261002)
    def interval(a):
        if not len(a):
            return {'n': 0, 'mean': None, 'ci97_5': None}
        draw = rng.integers(len(a), size=(10000, len(a)))
        return {'n': len(a), 'mean': float(a.mean()), 'ci97_5': np.quantile(a[draw].mean(1), [.0125, .9875]).tolist()}
    for other in ('C', 'FixedRandom'):
        comparisons[other] = {}
        for d in DIMENSIONS:
            pairs = [(lookup[c['case'], c['methods']['E']][d], lookup[c['case'], c['methods'][other]][d]) for c in cases]
            delta = np.array([float(a=='correct')-float(b=='correct') for a, b in pairs])
            complete = np.array([float(a=='correct')-float(b=='correct') for a, b in pairs if 'uncertain' not in (a, b)])
            comparisons[other][d] = {'all_cases_confirmed_correct_difference': interval(delta),
                'both_judgable_difference': interval(complete)}
        preference = dict.fromkeys(('E_wins', 'other_wins', 'both_best', 'neither_best', 'all_wrong', 'uncertain'), 0)
        for c in cases:
            best = lookup[c['case'], next(iter(c['labels']))]['best']
            if best in ('all_wrong', 'uncertain'):
                preference[best] += 1
            else:
                win_e, win_o = c['methods']['E'] in best.split(';'), c['methods'][other] in best.split(';')
                preference['both_best' if win_e and win_o else 'E_wins' if win_e else 'other_wins' if win_o else 'neither_best'] += 1
        comparisons[other]['preference'] = preference
    return {'absolute_counts': summary, 'E_minus': comparisons,
            'interpretation': 'descriptive development evidence; intervals not corrected across dimensions; no automatic acceptance'}


def report(out, rating_paths):
    protocol = json.loads((out/'protocol.json').read_text())
    if sha(out/'sealed_mapping.json') != protocol['mapping_sha256']:
        raise ValueError('mapping changed')
    cases = json.loads((out/'sealed_mapping.json').read_text())['cases']
    if len(rating_paths) != len(set(p.resolve() for p in rating_paths)):
        raise ValueError('duplicate reviewer files')
    # Validate every reviewer before creating any revealed output.
    ratings = [validate_ratings(read(p), cases) for p in rating_paths]
    target = out/'human_results'
    target.mkdir(exist_ok=False)
    write(target/'rating_commitments.json', {f'rater-{j+1}': sha(p) for j, p in enumerate(rating_paths)})
    summaries = {}
    for j, (path, lookup) in enumerate(zip(rating_paths, ratings), 1):
        shutil.copyfile(path, target/f'rater-{j}.csv')
        summaries[f'rater-{j}'] = analyze(cases, lookup)
    write(target/'summary.json', {'rater_count': len(ratings), 'raters_separate_not_pooled': summaries})
    shutil.copyfile(out/'sealed_mapping.json', target/'revealed_mapping.json')
    print(f'Completed {len(ratings)} separately reported raters; no automatic acceptance decision.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('report'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ratings', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    prepare(args.root) if args.phase == 'prepare' else report(args.out, args.ratings)
