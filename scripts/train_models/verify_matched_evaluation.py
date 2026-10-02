"""Audit saved matched-run boundaries and predictions without fitting or human outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation_features import digest_file, feature_identity
from matched_inputs import digest, training_payload
from nested_evaluation import Settings, candidates, make_splits, metrics


def read(path):
    return json.loads(Path(path).read_text())


def verify_checkpoint(root, checkpoint):
    root, checkpoint = Path(root), Path(checkpoint)
    provenance = read(root / 'provenance.json')
    for path, expected in provenance['input_sha256'].items():
        assert digest_file(path) == expected, f'Changed input: {path}'
    for name, expected in provenance['source_sha256'].items():
        assert digest_file(root / 'source' / name) == expected, f'Changed source snapshot: {name}'
    plan = read(root / 'plan.json')
    assert plan == read(Path(provenance['prepared']) / 'plan.json')
    result = read(checkpoint / 'predictions.json')
    a, f = result['chemist'], result['outer_fold']
    arm = read(Path(provenance['prepared']) / f'arm-{a}.json')
    target = {int(k): v for k, v in arm['target'].items()}
    approved, fold = plan['arms'][a - 1], plan['outer_folds'][f - 1]
    assert digest(target) == approved['target_sha256']
    assert digest(arm['pairs']) == approved['pairs_sha256']
    for name in ('train', 'test', 'secondary_query'):
        assert result[name] == fold[name], name
    train = set(fold['train'])
    primary = set(plan['primary_records'])
    assert train | set(fold['test']) == primary
    assert not train & set(fold['test'])
    groups = np.array(plan['group_hashes'])
    held = set(groups[fold['test'] + fold['secondary_query']])
    assert not held & set(groups[list(train)])
    assert result['settings'] == Settings().__dict__
    names = [c.name for c in candidates()]
    assert set(result['selection_scores']) == set(names) and len(names) == 186
    ranking = sorted(names, key=lambda name: (-result['selection_scores'][name], name))
    assert result['selected'] == ranking[:10]
    assert any(np.isclose(result['ridge_alpha'], x) for x in np.logspace(-3, 3, 50))
    expected_inner = [dict(train=b['train'], test=b['validation']) for b in approved['inner_splits']
                      if b['outer_fold'] == f]
    assert result['inner_splits'] == expected_inner
    t = np.array(fold['train'])
    regenerated = make_splits([target[int(i)] for i in t], 5, 42, groups[t])
    assert [dict(train=t[x].tolist(), test=t[y].tolist()) for x, y in regenerated] == expected_inner
    expected_boundaries = [b for b in approved['boundaries'] if b['outer_fold'] == f]
    expected_train = {tuple(sorted(b['train'])) for b in expected_boundaries}
    assert len(expected_train) == 12
    for record in result['audit']:
        fit, query = set(record['train']), set(record['query'])
        assert record['outer_fold'] == f and fit <= train and not fit & query
        assert not set(groups[list(fit)]) & (set(groups[list(query)]) | held)
        assert tuple(sorted(fit)) in expected_train
    assert sum(r['stage'] == 'base_tuning' for r in result['audit']) == 940
    assert sum(r['stage'] == 'base_inner' for r in result['audit']) == 930
    assert sum(r['stage'] == 'base_outer' for r in result['audit']) == 10
    for record in result['transforms']:
        assert tuple(sorted(record['train'])) in expected_train
    assert len(result['timings']) == 940
    assert all(t['outer_fold'] == f and t['trials'] == 10 for t in result['timings'])
    from collections import Counter
    assert Counter(t['candidate'] for t in result['timings'] if t['role'] == 'inner') == Counter({n: 5 for n in names})
    assert Counter(t['candidate'] for t in result['timings'] if t['role'] == 'outer') == Counter(result['selected'])
    records = result['feature_fits']
    assert {tuple(sorted(r['train'])) for r in records} == expected_train and len(records) == 12
    pairs = arm['pairs']
    left, right, labels = ([p[1] for p in pairs], [p[2] for p in pairs],
                           [float(p[3] == 'A') for p in pairs])
    for record in records:
        assert not record['cached'], 'Fresh arm-specific FSscore fitting required'
        cache = root / f'cache-arm-{a}' / record['key']
        manifest = read(cache / 'manifest.json')
        identity = manifest['identity']
        spec = identity['specification']
        assert spec['epochs'] == 20 and spec['seed'] == 42
        assert spec['checkpoint_sha256'] == provenance['pretrained_checkpoint_sha256']
        assert spec['provider_sha256'] == provenance['source_sha256']['evaluation_features.py']
        key, pair_rows, expected_identity = feature_identity(arm['ids'], record['train'],
                                                          np.array(left), np.array(right), np.array(labels), spec)
        assert key == record['key'] and identity == expected_identity
        assert record['pair_rows'] == pair_rows.tolist() and record['pair_count'] == len(pair_rows) > 0
        boundary = next(b for b in expected_boundaries if b['train'] == record['train'])
        assert [pairs[i][0] for i in pair_rows] == boundary['pair_rows']
        assert digest(training_payload(target, pairs, record['train'])) == boundary['payload_sha256']
        assert digest_file(cache / 'scores.npy') == manifest['scores_sha256']
        scores = np.load(cache / 'scores.npy', allow_pickle=False)
        assert scores.shape == (500,) and np.isfinite(scores).all()
    assert result['predictions'].keys() == result['secondary_predictions'].keys()
    assert 'stack' in result['predictions']
    scores = {}
    for name, prediction in result['predictions'].items():
        assert len(prediction) == len(fold['test']) and np.isfinite(prediction).all()
        secondary = result['secondary_predictions'][name]
        assert len(secondary) == len(fold['secondary_query']) and np.isfinite(secondary).all()
        scores[name] = metrics([target[i] for i in fold['test']], prediction)
    return dict(status='passed', chemist=a, outer_fold=f,
                primary_n=len(fold['test']), secondary_n=len(fold['secondary_query']),
                base_procedures=940, base_estimator_fits=10340, fresh_fs_fits=len(records),
                audited_boundaries=len(result['audit']), metrics=scores,
                predictions_sha256=digest_file(checkpoint / 'predictions.json'))


def verify_run(root, require_complete=False):
    root = Path(root)
    reports, folds = [], []
    for checkpoint in sorted(root.glob('arm-*/fold-*')):
        if not (checkpoint / 'predictions.json').exists():
            continue
        report = verify_checkpoint(root, checkpoint)
        saved = read(checkpoint / 'audit.json')
        assert saved == report
        reports.append(report)
        folds.append(read(checkpoint / 'predictions.json'))
    keys = [(f['chemist'], f['outer_fold']) for f in folds]
    assert len(set(keys)) == len(keys)
    complete = set(keys) == {(a, f) for a in range(1, 4) for f in range(1, 6)}
    plan = read(root / 'plan.json')
    for a in range(1, 4):
        for key, population in [('test', 'primary_records'), ('secondary_query', 'secondary_records')]:
            indices = [i for f in folds if f['chemist'] == a for i in f[key]]
            assert len(indices) == len(set(indices))
            if complete:
                assert sorted(indices) == plan[population]
    if require_complete:
        assert complete, 'All 15 arm/fold checkpoints are required'
    return dict(status='passed', complete=complete, completed_checkpoints=len(reports),
                primary_predictions=sum(r['primary_n'] for r in reports),
                secondary_predictions=sum(r['secondary_n'] for r in reports),
                base_procedures=sum(r['base_procedures'] for r in reports),
                base_estimator_fits=sum(r['base_estimator_fits'] for r in reports),
                fresh_fs_fits=sum(r['fresh_fs_fits'] for r in reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_run(args.root, args.require_complete), indent=2))
