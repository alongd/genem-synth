"""Behavioral checks of the validation boundary, using synthetic data only."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/train_models'))
from evaluation_features import FSScoreFeatures, feature_identity, training_pair_rows
from final_fit import FinalFitEvaluator
from nested_evaluation import (Candidate, FeatureSpace, MidCDF, NestedEvaluator,
                               Settings, make_splits)


class LearnedSignal:
    def __init__(self, X, left, right, labels):
        self.X, self.left, self.right, self.labels = X, left, right, labels
        self.fits = []

    def __call__(self, train):
        rows = training_pair_rows(train, self.left, self.right)
        self.fits.append((set(train), set(rows)))
        return self.X[:, 0] * (1 + self.labels[rows].mean())


def fixture(seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 12))
    y = 3 + X[:, 0] * 0.4 + rng.normal(0, .1, 60)
    left, right = np.arange(60), np.roll(np.arange(60), -1)
    labels = (y[left] < y[right]).astype(float)
    return X, y, left, right, labels


def engine(X, y, left, right, labels, groups=None):
    learned = LearnedSignal(X, left, right, labels)
    space = FeatureSpace([f'row{i}' for i in range(len(y))], {'sa': X[:, 0], 'sc': X[:, 1]},
                         X, learned=learned, groups=groups)
    configurations = [Candidate('linear', ('sa',)), Candidate('linear', ('sa', 'ft'), 'pca')]
    return NestedEvaluator(space, y, configurations,
                           Settings(inner_folds=3, trials=1, top_k=2)), learned


def test_held_out_targets_and_pair_labels_cannot_change_predictions():
    X, y, left, right, labels = fixture()
    train, test = np.arange(48), np.arange(48, 60)
    original, original_source = engine(X, y, left, right, labels)
    a = original.evaluate_outer(train, test, 1)
    changed_y, changed_labels = y.copy(), labels.copy()
    changed_y[test] += np.arange(12) * 100
    incident = np.isin(left, test) | np.isin(right, test)
    changed_labels[incident] = 1 - changed_labels[incident]
    changed, _ = engine(X, changed_y, left, right, changed_labels)
    b = changed.evaluate_outer(train, test, 1)
    assert a['selected'] == b['selected']
    assert a['ridge_alpha'] == b['ridge_alpha']
    assert a['selection_scores'] == b['selection_scores']
    for name in a['predictions']:
        np.testing.assert_allclose(a['predictions'][name], b['predictions'][name], rtol=0, atol=1e-12)
    assert a['metrics']['stack'] != b['metrics']['stack']
    for fit_rows, pair_rows in original_source.fits:
        assert fit_rows <= set(train)
        assert set(left[list(pair_rows)]) <= fit_rows
        assert set(right[list(pair_rows)]) <= fit_rows
    for step in a['audit']:
        assert set(step['train']) <= set(train)
        assert not set(step['train']) & set(step['query'])


def test_training_target_change_is_detectable():
    X, y, left, right, labels = fixture()
    train, test = np.arange(48), np.arange(48, 60)
    a, _ = engine(X, y, left, right, labels)
    before = a.evaluate_outer(train, test, 1)
    y[train] += 4
    b, _ = engine(X, y, left, right, labels)
    after = b.evaluate_outer(train, test, 1)
    np.testing.assert_allclose(np.array(after['predictions']['stack']) - before['predictions']['stack'],
                               4, atol=1e-9)


def test_scaling_and_pca_do_not_fit_query_rows():
    X, y, left, right, labels = fixture()
    train, test = np.arange(48), np.arange(48, 60)
    candidate = Candidate('linear', ('sa',), 'pca')
    a = FeatureSpace(np.arange(60), {'sa': X[:, 0]}, X)
    a_train, a_test = a.arrays(candidate, train, test)
    changed = X.copy()
    changed[test] += 1e6
    b = FeatureSpace(np.arange(60), {'sa': changed[:, 0]}, changed)
    b_train, b_test = b.arrays(candidate, train, test)
    np.testing.assert_allclose(a_train, b_train, rtol=0, atol=1e-12)
    assert not np.allclose(a_test, b_test)


def test_canonical_groups_are_kept_together_at_every_split():
    X, y, left, right, labels = fixture()
    groups = np.array([f'g{i // 2}' for i in range(60)])
    splits = make_splits(y, 5, 42, groups)
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    evaluator, _ = engine(X, y, left, right, labels, groups)
    result = evaluator.evaluate_outer(*splits[0], 1)
    for step in result['audit']:
        assert not set(groups[step['train']]) & set(groups[step['query']])


def test_outer_alias_overlap_is_rejected():
    X, y, left, right, labels = fixture()
    groups = np.arange(60)
    groups[59] = groups[0]
    evaluator, _ = engine(X, y, left, right, labels, groups)
    with pytest.raises(AssertionError, match='representation overlap'):
        evaluator.evaluate_outer(np.arange(48), np.arange(48, 60), 1)


def test_repeated_outer_test_group_is_rejected_before_fitting():
    X, y, left, right, labels = fixture()
    evaluator, _ = engine(X, y, left, right, labels)
    split = (np.arange(48), np.arange(48, 60))
    with pytest.raises(ValueError, match='multiple outer test folds'):
        evaluator.run(splits=[split, split])
    assert not evaluator.audit


def test_transform_eviction_preserves_features():
    X, _, _, _, _ = fixture()
    space = FeatureSpace(np.arange(60), {'sa': X[:, 0]}, X)
    candidate = Candidate('linear', ('sa',), 'pca')
    before = space.arrays(candidate, np.arange(48), np.arange(48, 60))
    space.clear_transforms()
    assert not space.cache and not space.embedding_cache
    after = space.arrays(candidate, np.arange(48), np.arange(48, 60))
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_feature_cache_key_tracks_only_the_labels_used_for_fitting():
    _, _, left, right, labels = fixture()
    ids, train, spec = list(map(str, range(60))), np.arange(48), {'seed': 42}
    key, rows, _ = feature_identity(ids, train, left, right, labels, spec)
    changed = labels.copy()
    changed[59] = 1 - changed[59]
    assert feature_identity(ids, train, left, right, changed, spec)[0] == key
    changed[rows[0]] = 1 - changed[rows[0]]
    assert feature_identity(ids, train, left, right, changed, spec)[0] != key
    assert feature_identity(ids, train[:-1], left, right, labels, spec)[0] != key
    assert feature_identity(ids, train, left, right, labels, {'seed': 3})[0] != key


def test_feature_cache_reuse_and_corruption_detection(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'base.ckpt'
    checkpoint.write_bytes(b'not-a-model; fit is replaced by deterministic test provider')
    args = ([str(i) for i in range(10)], np.arange(9), np.arange(1, 10),
            np.ones(9), tmp_path / 'cache', checkpoint)
    calls = []
    def fake_fit(self, directory, rows):
        calls.append(rows.tolist())
        return np.arange(10, dtype=float)
    monkeypatch.setattr(FSScoreFeatures, 'fit_scores', fake_fit)
    a = FSScoreFeatures(*args)
    expected = a(np.arange(8))
    b = FSScoreFeatures(*args)
    np.testing.assert_array_equal(b(np.arange(8)), expected)
    assert len(calls) == 1
    cached = next((tmp_path / 'cache').glob('*/scores.npy'))
    np.save(cached, expected + 2)
    with pytest.raises(ValueError, match='corruption'):
        FSScoreFeatures(*args)(np.arange(8))


def test_partial_run_does_not_report_final_score_or_zero_fold_sd():
    X, y, left, right, labels = fixture()
    evaluator, _ = engine(X, y, left, right, labels)
    result = evaluator.run(outer_limit=1)
    assert result['complete'] is False
    assert result['evaluated_n'] < 60
    assert result['fold_mean_and_sd']['stack']['r2']['sd'] is None
    json.dumps(result, allow_nan=False)


def test_rank_baseline_uses_training_empirical_cdf():
    transformer = MidCDF().fit([[1], [2], [2], [4]])
    np.testing.assert_allclose(transformer.transform([[0], [1], [2], [3], [5]]).ravel(),
                               [0, .125, .5, .75, 1])


def test_per_fold_callback_preserves_predictions_and_partial_status():
    X, y, left, right, labels = fixture()
    evaluator, _ = engine(X, y, left, right, labels)
    checkpoints = []
    result = evaluator.run(outer_limit=2, on_fold=checkpoints.append)
    assert [c['evaluated_folds'] for c in checkpoints] == [1, 2]
    assert len(checkpoints[0]['folds']) == 1
    assert len(checkpoints[0]['timings']) < len(checkpoints[1]['timings'])
    assert not any(c['complete'] for c in checkpoints)
    plain, _ = engine(X, y, left, right, labels)
    expected = plain.run(outer_limit=2)
    for a, b in zip(result['folds'], expected['folds']):
        assert a['predictions'] == b['predictions']
    assert result['pooled_evaluated_metrics'] == checkpoints[-1]['pooled_evaluated_metrics']


def test_failed_checkpoint_stops_before_the_next_fold():
    X, y, left, right, labels = fixture()
    evaluator, _ = engine(X, y, left, right, labels)
    def fail_check(partial):
        raise RuntimeError('failed provenance check')
    with pytest.raises(RuntimeError, match='failed provenance'):
        evaluator.run(on_fold=fail_check)
    assert evaluator.outer_fold == 1


@pytest.mark.parametrize('kind', ['duplicate', 'global_ft', 'nan'])
def test_invalid_feature_inputs_fail_closed(kind):
    ids, scores = np.arange(10), {'sa': np.ones(10)}
    if kind == 'duplicate':
        ids[1] = ids[0]
    elif kind == 'global_ft':
        scores['ft'] = np.ones(10)
    else:
        scores['sa'][0] = np.nan
    with pytest.raises(ValueError):
        FeatureSpace(ids, scores, np.ones((10, 8)))


@pytest.mark.parametrize('model', ['xgb', 'mlp'])
def test_actual_base_learner_smoke(model):
    X, y, left, right, labels = fixture()
    evaluator, _ = engine(X, y, left, right, labels)
    evaluator.settings = Settings(inner_folds=3, trials=1, top_k=1, mlp_epochs=3)
    evaluator.allowed, evaluator.outer_fold = set(range(48)), 1
    prediction = evaluator.fit_base(Candidate(model, ('sa', 'sc'), 'pca'),
                                     np.arange(48), np.arange(48, 60), 'outer')
    assert prediction.shape == (12,) and np.isfinite(prediction).all()


def final_engine(X, y, left, right, labels, groups=None, primary=None):
    learned = LearnedSignal(X, left, right, labels)
    space = FeatureSpace([f'row{i}' for i in range(len(y))], {'sa': X[:, 0], 'sc': X[:, 1]},
                         X, learned=learned, groups=groups)
    primary = np.arange(48) if primary is None else np.asarray(primary)
    target = np.full(len(y), np.nan)
    target[primary] = y[primary]
    configurations = [Candidate('linear', ('sa',)), Candidate('linear', ('sc',)),
                      Candidate('linear', ('sa', 'sc')), Candidate('linear', ('sa', 'ft'), 'pca')]
    return FinalFitEvaluator(space, target, configurations,
                             Settings(inner_folds=3, trials=1, top_k=2), primary), learned


def test_precomputed_selection_reproduces_the_full_fit_exactly():
    X, y, left, right, labels = fixture()
    query = np.arange(48, 60)
    full, _ = final_engine(X, y, left, right, labels)
    a = full.fit_final(np.arange(48), query)
    reused, _ = final_engine(X, y, left, right, labels)
    b = reused.fit_final(np.arange(48), query, selection=a['selected'])
    assert b['selection_source'] == 'precomputed' and a['selection_source'] == 'inner_oof'
    assert b['selected'] == a['selected'] and b['ridge_alpha'] == a['ridge_alpha']
    assert b['ridge_coefficients'] == a['ridge_coefficients']
    for name, values in a['predictions'].items():
        assert np.array_equal(values, b['predictions'][name]), name
    for name in b['selection_scores']:
        assert b['selection_scores'][name] == a['selection_scores'][name]


@pytest.mark.parametrize('selection', [['linear:sa:none'], ['nope:x:none', 'linear:sa:none'],
                                       ['linear:sa:none', 'linear:sa:none']])
def test_an_unusable_precomputed_selection_fails_closed(selection):
    X, y, left, right, labels = fixture()
    engine_, _ = final_engine(X, y, left, right, labels)
    with pytest.raises(ValueError):
        engine_.fit_final(np.arange(48), np.arange(48, 60), selection=selection)


def test_query_rows_that_duplicate_training_structures_need_an_explicit_opt_in():
    X, y, left, right, labels = fixture()
    groups = np.array([f'g{i}' for i in range(48)] + [f'g{i}' for i in range(12)])
    engine_, _ = final_engine(X, y, left, right, labels, groups=groups)
    with pytest.raises(AssertionError, match='Molecular representation overlap'):
        engine_.fit_final(np.arange(48), np.arange(48, 60))
    allowed, _ = final_engine(X, y, left, right, labels, groups=groups)
    allowed.allow_query_group_overlap = True
    result = allowed.fit_final(np.arange(48), np.arange(48, 60))
    assert len(result['predictions']['stack']) == 12
