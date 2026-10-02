"""Run the approved 15 nested folds with an aggregate hard fitting deadline."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation_features import (FSScoreFeatures, REPO, atomic_json, canonical_ids,
                                 digest_file, outside_checkout)
from matched_inputs import digest
from nested_evaluation import FeatureSpace, NestedEvaluator, Settings, candidates, make_splits
from verify_matched_evaluation import verify_checkpoint, verify_run


def read(path):
    return json.loads(Path(path).read_text())


def software_versions():
    versions = {p: importlib.metadata.version(p) for p in
                ('numpy', 'pandas', 'scikit-learn', 'torch', 'optuna', 'pytorch-lightning')}
    # The pinned environment provides XGBoost through its source package.
    import xgboost
    versions['xgboost'] = xgboost.__version__
    return versions


def supervised_process(command, seconds, status_path):
    """One process group and one wall-clock budget, including checkpoint reviews."""
    start = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=seconds)
        status = 'completed' if code == 0 else 'failed'
    except subprocess.TimeoutExpired:
        status, code = 'budget_exhausted', 124
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        raise
    atomic_json(status_path, dict(status=status, exit_code=code, cap_seconds=seconds,
                                 aggregate_wall_seconds=time.monotonic() - start,
                                 note='Conservative fitting budget includes worker loading and checkpoint review; shutdown grace <=10 s.'))
    return code


def worker(prepared, feature_path, root):
    plan = read(prepared / 'plan.json')
    audit = read(prepared / 'preparation-audit.json')
    assert audit['status'] == 'passed'
    # Human outcomes are intentionally not opened, even to recompute their hash.
    inputs = [prepared / 'preparation-audit.json']
    for name in ('plan.json', 'arm-1.json', 'arm-2.json', 'arm-3.json'):
        path = prepared / name
        assert digest_file(path) == audit['output_sha256'][name]
        inputs.append(path)
    frozen = pd.read_csv(feature_path)
    manifest_path = feature_path.with_suffix('.manifest.json')
    manifest = read(manifest_path)
    assert manifest['label_free'] and digest_file(feature_path) == manifest['features_sha256']
    arm_inputs = [read(prepared / f'arm-{a}.json') for a in range(1, 4)]
    ids = arm_inputs[0]['ids']
    assert all(arm['ids'] == ids for arm in arm_inputs)
    groups = [hashlib.sha256(s.encode()).hexdigest() for s in canonical_ids(ids)]
    assert groups == plan['group_hashes']
    assert len(frozen) == len(set(frozen.smiles)) == len(ids)
    assert set(frozen.smiles) == set(ids)
    frozen = frozen.set_index('smiles').loc[ids]
    scores = {k: frozen[k].to_numpy() for k in ('sa', 'sc', 'syba', 'ra', 'fs_base')}
    columns = sorted([c for c in frozen if c.startswith('fp_')], key=lambda c: int(c[3:]))
    assert len(columns) >= 8
    embeddings = frozen[columns].to_numpy()
    checkpoint = REPO / 'fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt'
    inputs += [feature_path, manifest_path, checkpoint]
    atomic_json(root / 'plan.json', plan)
    source_dir = root / 'source'
    source_dir.mkdir()
    source_hashes = {}
    for name in ('nested_evaluation.py', 'evaluation_features.py', 'matched_inputs.py',
                 'prepare_matched_evaluation.py', 'run_matched_evaluation.py', 'verify_matched_evaluation.py'):
        src = Path(__file__).with_name(name)
        (source_dir / name).write_bytes(src.read_bytes())
        source_hashes[name] = digest_file(src)
    provenance = dict(prepared=str(prepared), features=str(feature_path), settings=asdict(Settings()),
                      input_sha256={str(p): digest_file(p) for p in inputs}, source_sha256=source_hashes,
                      pretrained_checkpoint_sha256=digest_file(checkpoint),
                      versions=software_versions(),
                      started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      cap_seconds=43200, numerical_threads=2)
    atomic_json(root / 'provenance.json', provenance)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ['WANDB_DISABLED'] = 'true'
    started = time.monotonic()
    completed = 0
    with threadpool_limits(limits=2):
        for a, arm in enumerate(arm_inputs, 1):
            assert arm['chemist'] == a
            target_map = {int(k): v for k, v in arm['target'].items()}
            assert digest(target_map) == plan['arms'][a - 1]['target_sha256']
            assert digest(arm['pairs']) == plan['arms'][a - 1]['pairs_sha256']
            target = np.full(len(ids), np.nan)
            for i, value in target_map.items():
                target[i] = value
            pairs = arm['pairs']
            source = FSScoreFeatures(ids, [p[1] for p in pairs], [p[2] for p in pairs],
                                     [float(p[3] == 'A') for p in pairs], root / f'cache-arm-{a}',
                                     checkpoint, max_fits=60)
            features = FeatureSpace(ids, scores, embeddings, learned=source, groups=groups)
            evaluator = NestedEvaluator(features, target, candidates(), Settings(), plan['primary_records'])
            for fold in plan['outer_folds']:
                f = fold['fold']
                t = np.array(fold['train'])
                regenerated = make_splits(target[t], 5, 42, np.array(groups)[t])
                assert [dict(train=t[x].tolist(), validation=t[y].tolist(), outer_fold=f, inner_fold=j)
                        for j, (x, y) in enumerate(regenerated, 1)] == [b for b in plan['arms'][a - 1]['inner_splits']
                                                                     if b['outer_fold'] == f]
                print(f'BEGIN chemist={a} fold={f} completed={completed}/15', flush=True)
                before_fit, before_time = len(source.records), len(evaluator.timings)
                before_transform = len(features.fit_records)
                result = evaluator.evaluate_outer(fold['train'], fold['test'], f,
                                                   secondary_query=fold['secondary_query'], score=False)
                result.update(chemist=a, settings=asdict(Settings()),
                              feature_fits=source.records[before_fit:], timings=evaluator.timings[before_time:],
                              transforms=features.fit_records[before_transform:])
                directory = root / f'arm-{a}' / f'fold-{f}'
                directory.mkdir(parents=True)
                atomic_json(directory / 'predictions.json', result)
                # Only after immutable predictions exist do we score outer targets.
                verification = verify_checkpoint(root, directory)
                atomic_json(directory / 'audit.json', verification)
                completed += 1
                elapsed = time.monotonic() - started
                progress = dict(status='running', completed_checkpoints=completed,
                                chemist=a, outer_fold=f, elapsed_seconds=elapsed,
                                projected_total_seconds=elapsed * 15 / completed,
                                checkpoint=str(directory), audit='passed')
                atomic_json(root / 'progress.json', progress)
                print('CHECKPOINT ' + json.dumps(progress), flush=True)
                if completed == 1:
                    progress['status'] = 'awaiting_first_fold_review'
                    atomic_json(root / 'progress.json', progress)
                    decision = root / 'continue-after-first-fold.json'
                    while not decision.exists():
                        time.sleep(2)
                    review = read(decision)
                    assert review['continue'] is True and review['predictions_sha256'] == verification['predictions_sha256']
                    assert review['cap_seconds'] == 43200
                # Detect a code change before another real fit starts.
                assert all(digest_file(Path(__file__).with_name(n)) == h for n, h in source_hashes.items())
    final = verify_run(root, require_complete=True)
    atomic_json(root / 'final-audit.json', final)
    atomic_json(root / 'progress.json', dict(final, status='completed',
                                            elapsed_seconds=time.monotonic() - started))
    print('COMPLETE ' + json.dumps(final), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'features', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--prior-budget-status', action='append', default=[],
                        help='Completed failed attempts to charge against the same aggregate cap')
    args = parser.parse_args()
    prepared, features, root = map(outside_checkout, (args.prepared, args.features, args.output))
    if args.worker:
        worker(prepared, features, root)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh run root; preserve previous attempts and charge their elapsed time')
    prior = [outside_checkout(p) for p in args.prior_budget_status]
    if len(set(prior)) != len(prior):
        raise ValueError('Duplicate prior budget entries')
    histories = [read(p) for p in prior]
    if any(h['status'] != 'failed' for h in histories):
        raise ValueError('Budget-exhausted or successful runs cannot be restarted')
    used = sum(h['aggregate_wall_seconds'] for h in histories)
    if not 0 <= used < 43200:
        raise ValueError('Aggregate fitting budget is exhausted or invalid')
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'budget-start.json', dict(cap_seconds=43200, numerical_threads=2,
                                               prior_attempt_seconds=used, remaining_seconds=43200 - used,
                                               prior_status_sha256={str(p): digest_file(p) for p in prior},
                                               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    code = supervised_process([sys.executable, str(Path(__file__).resolve()), '--worker',
                               '--prepared', str(prepared), '--features', str(features), '--output', str(root)],
                              43200 - used, root / 'budget-status.json')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
