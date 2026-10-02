"""Run a bounded pilot or explicitly requested full nested evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation_features import (FSScoreFeatures, REPO, atomic_json, canonical_ids,
                                 digest_file, outside_checkout)
from nested_evaluation import FeatureSpace, NestedEvaluator, Settings, candidates, make_splits
from verify_nested_evaluation import verify


def load_data(regression_path, pairs_path, features_path):
    regression = pd.read_csv(outside_checkout(regression_path), usecols=['smiles', 'consensus_score'])
    pairs = pd.read_csv(outside_checkout(pairs_path), usecols=['smiles_a', 'smiles_b', 'consensus_choice'])
    frozen = pd.read_csv(outside_checkout(features_path))
    sidecar = Path(features_path).with_suffix('.manifest.json')
    metadata = json.loads(sidecar.read_text())
    if metadata['features_sha256'] != digest_file(features_path) or not metadata['label_free']:
        raise ValueError('Frozen feature provenance failed')
    ids = regression.smiles.tolist()
    groups = canonical_ids(ids)
    frozen_ids = frozen.smiles.tolist()
    if len(set(ids)) != len(ids) or len(set(frozen_ids)) != len(frozen_ids):
        raise ValueError('Duplicate row identifiers; resolve before aligning inputs')
    if set(ids) != set(frozen_ids):
        raise ValueError('Feature and target populations differ')
    frozen.index = frozen_ids
    frozen = frozen.loc[ids]
    lookup = {identity: row for row, identity in enumerate(ids)}
    group_lookup = {identity: row for row, identity in enumerate(groups)}
    if not pairs.consensus_choice.isin(['A', 'B']).all():
        raise ValueError('Pairwise labels must be A or B')
    try:
        def endpoints(values):
            canonical = canonical_ids(values)
            return np.array([lookup[s] if s in lookup else group_lookup[g] for s, g in zip(values, canonical)])
        left = endpoints(pairs.smiles_a.tolist())
        right = endpoints(pairs.smiles_b.tolist())
    except KeyError as error:
        raise ValueError('Pair endpoint absent from the regression population') from error
    scores = {name: frozen[name].to_numpy() for name in ('sa', 'sc', 'syba', 'ra', 'fs_base')
              if name in frozen}
    fp_columns = sorted((name for name in frozen if name.startswith('fp_')),
                        key=lambda name: int(name.split('_')[1]))
    if not fp_columns:
        raise ValueError('No frozen embeddings supplied')
    return (ids, groups, regression.consensus_score.to_numpy(), scores, frozen[fp_columns].to_numpy(),
            left, right, (pairs.consensus_choice == 'A').astype(float).to_numpy(), metadata)


def save_results(output, result, target, plan, provenance, source, start, fitting_start):
    output.mkdir(parents=True, exist_ok=True)
    result = dict(result, **provenance)
    result['publication_result'] = result['profile'] == 'full' and result['complete']
    result['feature_fits'] = list(source.records)
    result['wall_seconds'] = perf_counter() - start
    result['fitting_seconds'] = perf_counter() - fitting_start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['process_resources'] = dict(cpu_seconds=usage.ru_utime + usage.ru_stime,
                                       peak_rss_mib=usage.ru_maxrss / 1024)
    rows = []
    for fold in result['folds']:
        for position, index in enumerate(fold['test']):
            rows.append(dict(row=index, outer_fold=fold['outer_fold'], target=float(target[index]),
                             **{name: values[position] for name, values in fold['predictions'].items()}))
    atomic_json(output / 'plan.json', plan)
    pd.DataFrame(rows).to_csv(output / 'outer_predictions.csv', index=False)
    atomic_json(output / 'results.json', result)
    verification = verify(output)
    atomic_json(output / 'verification.json', verification)
    return result, verification


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--outer-limit', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-fit-seconds', type=float, default=21600)
    args = parser.parse_args()
    if args.max_fit_seconds <= 0:
        raise ValueError('Fitting budget must be positive')
    output = outside_checkout(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing run artifacts are immutable')
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    pilot = args.profile == 'pilot'
    settings = Settings(inner_folds=3 if pilot else 5, trials=2 if pilot else 10,
                        top_k=4 if pilot else 10, threads=args.threads)
    ids, groups, target, scores, embedded, left, right, labels, feature_manifest = load_data(
        args.regression, args.pairs, args.features)
    configurations = candidates(pilot=pilot)
    required = {name for c in configurations for name in c.tools if name != 'ft'}
    if not required <= scores.keys():
        raise ValueError(f'Missing required frozen tools: {sorted(required - scores.keys())}')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ['WANDB_DISABLED'] = 'true'
    source = FSScoreFeatures(ids, left, right, labels, args.cache,
                            REPO / 'fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt',
                            max_fits=8 if pilot else None)
    features = FeatureSpace(ids, scores, embedded, learned=source, groups=groups)
    evaluator = NestedEvaluator(features, target, configurations, settings)
    limit = args.outer_limit if args.outer_limit is not None else (1 if pilot else None)
    splits = make_splits(target, settings.outer_folds, settings.seed, groups)
    plan = dict(profile=args.profile, arguments=vars(args),
                outer_splits=[dict(train=a.tolist(), test=b.tolist()) for a, b in splits],
                canonical_groups=[hashlib.sha256(g.encode()).hexdigest() for g in groups],
                n_records=len(ids), n_canonical_groups=len(set(groups)))
    atomic_json(output / 'plan.json', plan)
    provenance = dict(profile=args.profile, frozen_features=feature_manifest)
    provenance['input_sha256'] = {name: digest_file(path) for name, path in
                              [('regression', args.regression), ('pairs', args.pairs), ('features', args.features)]}
    provenance['source_sha256'] = {p.name: digest_file(p) for p in
                              Path(__file__).parent.glob('*evaluation*.py')}
    provenance['commit'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    provenance['working_tree_modified'] = bool(subprocess.check_output(
        ['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    provenance['versions'] = {name: importlib.metadata.version(name) for name in
                          ['numpy', 'pandas', 'scikit-learn', 'torch', 'optuna', 'pytorch-lightning']}
    import xgboost
    provenance['versions']['xgboost'] = xgboost.__version__
    fitting_start = perf_counter()

    def checkpoint(partial):
        directory = output / f"checkpoint-{partial['evaluated_folds']:02d}"
        saved, verification = save_results(directory, partial, target, plan, provenance,
                                           source, start, fitting_start)
        projected = saved['fitting_seconds'] * len(splits) / saved['evaluated_folds']
        progress = dict(evaluated_folds=saved['evaluated_folds'], evaluated_n=saved['evaluated_n'],
                        fitting_seconds=saved['fitting_seconds'], projected_total_seconds=projected,
                        verification=verification, checkpoint=str(directory))
        atomic_json(output / 'progress.json', progress)
        print('CHECKPOINT ' + json.dumps(progress), flush=True)
        if saved['fitting_seconds'] >= args.max_fit_seconds and not saved['complete']:
            raise TimeoutError('Fitting budget exhausted; verified checkpoints preserved')

    with threadpool_limits(limits=args.threads):
        result = evaluator.run(outer_limit=limit, splits=splits, on_fold=checkpoint)
    result, _ = save_results(output, result, target, plan, provenance, source, start, fitting_start)
    print(json.dumps(dict(profile=args.profile, complete=result['complete'],
                          publication_result=result['publication_result'],
                          evaluated_n=result['evaluated_n'], wall_seconds=result['wall_seconds'],
                          pooled_evaluated_metrics=result['pooled_evaluated_metrics']), indent=2))


if __name__ == '__main__':
    main()
