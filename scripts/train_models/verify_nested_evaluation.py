"""Independently check saved splits, fit provenance, predictions, and input hashes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_features import digest_file
from nested_evaluation import metrics


def verify(directory):
    directory = Path(directory)
    result = json.loads((directory / 'results.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    predictions = pd.read_csv(directory / 'outer_predictions.csv')
    groups = np.asarray(plan['canonical_groups'])
    population = set(range(len(groups)))
    tested = []
    test_groups = set()
    for split in plan['outer_splits']:
        train, test = set(split['train']), set(split['test'])
        assert train | test == population and not train & test
        assert not set(groups[list(train)]) & set(groups[list(test)])
        assert not test_groups & set(groups[list(test)])
        test_groups.update(groups[list(test)])
        tested.extend(test)
    assert sorted(tested) == sorted(population)
    for name, expected in result['input_sha256'].items():
        assert digest_file(plan['arguments'][name]) == expected, name
    for name, expected in result['source_sha256'].items():
        assert digest_file(Path(__file__).parent / name) == expected, name
    raw_target = pd.read_csv(plan['arguments']['regression'], usecols=['consensus_score'])
    assert not predictions.row.duplicated().any()
    np.testing.assert_allclose(predictions.target, raw_target.consensus_score.iloc[predictions.row],
                               rtol=0, atol=1e-12)
    for fold in result['folds']:
        train, test = set(fold['train']), set(fold['test'])
        assert fold['train'] == plan['outer_splits'][fold['outer_fold'] - 1]['train']
        assert fold['test'] == plan['outer_splits'][fold['outer_fold'] - 1]['test']
        for step in fold['audit']:
            fit, query = set(step['train']), set(step['query'])
            assert fit <= train and not fit & query
            assert not set(groups[list(fit)]) & set(groups[list(query)])
        for split in fold['inner_splits']:
            fit, query = set(split['train']), set(split['test'])
            assert fit | query == train and not fit & query
            assert not set(groups[list(fit)]) & set(groups[list(query)])
        rows = predictions[predictions.outer_fold == fold['outer_fold']]
        assert rows.row.tolist() == fold['test']
        for name, values in fold['predictions'].items():
            np.testing.assert_allclose(rows[name], values, rtol=0, atol=1e-12)
            for metric, value in metrics(rows.target, rows[name]).items():
                np.testing.assert_allclose(value, fold['metrics'][name][metric], rtol=0, atol=1e-12)
    for fit in result['feature_fits']:
        directory = Path(plan['arguments']['cache']) / fit['key']
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert digest_file(directory / 'scores.npy') == manifest['scores_sha256']
        assert fit['train'] == manifest['identity']['train']
        assert fit['pair_count'] == len(manifest['identity']['pairs'])
        train = set(fit['train'])
        assert any(train <= set(fold['train']) for fold in result['folds'])
        for left, right, label in manifest['identity']['pairs']:
            assert left in train and right in train and label in (0, 1)
    for name in result['pooled_evaluated_metrics']:
        for metric, value in metrics(predictions.target, predictions[name]).items():
            np.testing.assert_allclose(value, result['pooled_evaluated_metrics'][name][metric],
                                       rtol=0, atol=1e-12)
    assert len(predictions) == result['evaluated_n']
    assert result['complete'] == (len(result['folds']) == len(plan['outer_splits'])
                                  and set(predictions.row) == population)
    assert result['publication_result'] == (result['profile'] == 'full' and result['complete'])
    if len(result['folds']) == 1:
        assert result['fold_mean_and_sd']['stack']['r2']['sd'] is None
    return dict(status='passed', evaluated_rows=len(predictions),
                checked_feature_fits=len(result['feature_fits']),
                checked_fit_boundaries=sum(len(fold['audit']) for fold in result['folds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
