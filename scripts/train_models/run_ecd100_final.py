"""Final all-data fit of the nested procedure and label-free scoring of external molecules.

--check-equivalence (pilot profile): on outer fold 1, `evaluate_outer` with the external rows as
secondary query must give the same predictions as `fit_final` trained on that fold's training
records only. This shows `fit_final` is the outer-fold procedure without an outer test set.

No external label is read by this script; labels are joined afterwards by score_ecd100.py.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation_features import (FSScoreFeatures, REPO, atomic_json, canonical_ids, digest_file,
                                 outside_checkout)
from final_fit import FinalFitEvaluator
from nested_evaluation import FeatureSpace, NestedEvaluator, Settings, candidates, make_splits


def load(regression_path, pairs_path, features_path):
    regression = pd.read_csv(regression_path, usecols=['smiles', 'consensus_score'])
    pairs = pd.read_csv(pairs_path, usecols=['smiles_a', 'smiles_b', 'consensus_choice'])
    frozen = pd.read_csv(features_path)
    n = len(regression)
    if frozen.smiles.iloc[:n].tolist() != regression.smiles.tolist() or frozen.smiles.duplicated().any():
        raise ValueError('Feature rows must start with the regression population, in order, without duplicates')
    ids = frozen.smiles.tolist()
    groups = np.array(canonical_ids(ids))
    target = np.full(len(ids), np.nan)
    target[:n] = regression.consensus_score.to_numpy()
    lookup = {s: i for i, s in enumerate(ids[:n])}
    group_lookup = {g: i for i, g in enumerate(groups[:n])}

    def endpoints(values):
        return np.array([lookup[s] if s in lookup else group_lookup[g]
                         for s, g in zip(values, canonical_ids(values))])
    left, right = endpoints(pairs.smiles_a.tolist()), endpoints(pairs.smiles_b.tolist())
    labels = (pairs.consensus_choice == 'A').astype(float).to_numpy()
    scores = {k: frozen[k].to_numpy() for k in ('sa', 'sc', 'syba', 'ra', 'fs_base') if k in frozen}
    fp = sorted((c for c in frozen if c.startswith('fp_')), key=lambda c: int(c[3:]))
    primary = np.arange(n)
    external = np.arange(n, len(ids))
    training_groups = set(groups[:n])
    query = np.array([i for i in external if groups[i] not in training_groups], dtype=int)
    overlaps = [ids[i] for i in external if groups[i] in training_groups]
    return ids, groups, target, scores, frozen[fp].to_numpy(), left, right, labels, primary, query, overlaps


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('regression', 'pairs', 'features', 'output', 'cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--profile', choices=['pilot', 'full'], required=True)
    parser.add_argument('--check-equivalence', action='store_true')
    parser.add_argument('--selection', help='JSON file holding a "selected" list from a completed fit '
                                           '(and optionally "ridge_alpha" to assert). Reuses that '
                                           'selection instead of re-ranking every candidate.')
    parser.add_argument('--include-training-overlaps', action='store_true',
                        help='Also score external molecules that duplicate a training structure. '
                             'Those rows are then IN-SAMPLE and the run must be reported as '
                             'contaminated for the fitted model; the baselines are unaffected.')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    selection, expected_alpha = None, None
    if args.selection:
        payload = json.loads(Path(args.selection).read_text())
        selection, expected_alpha = list(payload['selected']), payload.get('ridge_alpha')
    output = outside_checkout(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    pilot = args.profile == 'pilot'
    settings = Settings(inner_folds=3 if pilot else 5, trials=2 if pilot else 10,
                        top_k=4 if pilot else 10, threads=args.threads)
    (ids, groups, target, scores, embedded, left, right, labels,
     primary, query, overlaps) = load(args.regression, args.pairs, args.features)
    if args.include_training_overlaps:
        query = np.arange(len(primary), len(ids), dtype=int)
    configurations = candidates(pilot=pilot)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ['WANDB_DISABLED'] = 'true'
    checkpoint = REPO / 'fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt'

    def evaluator(cls, primary_indices, cache):
        source = FSScoreFeatures(ids, left, right, labels, cache, checkpoint)
        features = FeatureSpace(ids, scores, embedded, learned=source, groups=groups)
        return cls(features, target_for(primary_indices), configurations, settings, primary_indices), source

    def target_for(primary_indices):
        masked = np.full(len(ids), np.nan)
        masked[primary_indices] = target[primary_indices]
        return masked

    provenance = dict(
        profile=args.profile, settings=asdict(settings), n_ids=len(ids), n_primary=len(primary),
        n_query=len(query), include_training_overlaps=args.include_training_overlaps,
        excluded_training_overlaps=[] if args.include_training_overlaps else overlaps,
        scored_training_overlaps=overlaps if args.include_training_overlaps else [],
        precomputed_selection=selection,
        input_sha256={k: digest_file(getattr(args, k)) for k in ('regression', 'pairs', 'features')},
        source_sha256={p.name: digest_file(p) for p in Path(__file__).parent.glob('*.py')
                       if p.name in ('nested_evaluation.py', 'evaluation_features.py', 'final_fit.py',
                                     'run_ecd100_final.py')},
        commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        versions={p: importlib.metadata.version(p) for p in
                  ('numpy', 'pandas', 'scikit-learn', 'torch', 'optuna', 'pytorch-lightning', 'rdkit-pypi')})
    import xgboost
    provenance['versions']['xgboost'] = xgboost.__version__
    atomic_json(output / 'provenance.json', provenance)

    with threadpool_limits(limits=args.threads):
        if args.check_equivalence:
            train, test = make_splits(target[primary], settings.outer_folds, settings.seed, groups[primary])[0]
            train, test = primary[train], primary[test]
            a, _ = evaluator(NestedEvaluator, primary, Path(args.cache) / 'outer')
            outer = a.evaluate_outer(train, test, 1, secondary_query=query, score=False)
            b, _ = evaluator(FinalFitEvaluator, train, Path(args.cache) / 'final')
            final = b.fit_final(train, np.concatenate([test, query]))
            report = dict(selected_equal=outer['selected'] == final['selected'],
                          alpha_equal=outer['ridge_alpha'] == final['ridge_alpha'])
            for name, values in final['predictions'].items():
                expected = np.array(outer['predictions'][name] + outer['secondary_predictions'][name])
                report[name + '_max_abs_diff'] = float(np.max(np.abs(expected - np.array(values))))
            report['passed'] = (report['selected_equal'] and report['alpha_equal'] and
                                max(v for k, v in report.items() if k.endswith('_max_abs_diff')) <= 1e-9)
            atomic_json(output / 'equivalence.json', report)
            print(json.dumps(report, indent=2))
            return
        final_evaluator, source = evaluator(FinalFitEvaluator, primary, args.cache)
        final_evaluator.allow_query_group_overlap = args.include_training_overlaps
        result = final_evaluator.fit_final(primary, query, selection=selection)
        if expected_alpha is not None and result['ridge_alpha'] != expected_alpha:
            raise ValueError(f"Ridge alpha {result['ridge_alpha']!r} does not reproduce the recorded "
                             f"{expected_alpha!r}; the reused selection is not equivalent")
        result.update(ids=[ids[i] for i in result['query']], settings=asdict(settings),
                      feature_fits=source.records, timings=final_evaluator.timings)
        atomic_json(output / 'final-fit.json', result)
        rows = pd.DataFrame({'smiles': result['ids'], **{k: v for k, v in result['predictions'].items()},
                             **{'raw_' + k: v for k, v in result['raw_scores'].items()}})
        rows.to_csv(output / 'predictions.csv', index=False)
        atomic_json(output / 'predictions.sha256.json',
                    {name: digest_file(output / name) for name in ('predictions.csv', 'final-fit.json')})
        print(json.dumps(dict(n_query=len(query), selected=result['selected'],
                              ridge_alpha=result['ridge_alpha'], seconds=result['seconds']), indent=2))


if __name__ == '__main__':
    main()
