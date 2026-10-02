"""Synthetic negative controls for matched targets, query-only records and hard budget."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/train_models'))
from matched_inputs import arm_inputs, mutate, ratings
from nested_evaluation import Candidate, FeatureSpace, NestedEvaluator, Settings
from run_matched_evaluation import software_versions, supervised_process
from test_nested_evaluation import LearnedSignal, fixture


def annotations():
    ids = [f'm{i}' for i in range(60)]
    people = ['person1', 'person2', 'person3']
    rows = []
    for i in range(60):
        j = (i + 1) % 60
        row = {'Molecule A SMILES': ids[i], 'Molecule B SMILES': ids[j]}
        for a, p in enumerate(people):
            row.update({f'{p} - Score A': str(1 + ((i + a) % 5)),
                        f'{p} - Score B': str(1 + ((j + a) % 5)), f'{p} - Choice': 'A'})
        rows.append(row)
    return ids, people, rows


def matched_engine(ids, target_map, pairs, model='linear', groups=None):
    X = np.random.default_rng(42).normal(size=(len(ids), 12))
    y = np.full(len(ids), np.nan)
    for i, v in target_map.items():
        y[i] = v
    left, right = np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])
    labels = np.array([float(p[3] == 'A') for p in pairs])
    source = LearnedSignal(X, left, right, labels)
    features = FeatureSpace(ids, {'sa': X[:, 0], 'sc': X[:, 1]}, X, learned=source, groups=groups)
    return NestedEvaluator(features, y, [Candidate(model, ('sa',)), Candidate(model, ('sa', 'ft'), 'pca')],
                           Settings(inner_folds=3, trials=1, top_k=2, mlp_epochs=3), sorted(target_map))


def test_omitted_annotations_and_outer_labels_leave_both_prediction_populations_unchanged():
    ids, people, rows = annotations()
    primary, train, test, query = set(range(54)), np.arange(42), np.arange(42, 54), np.arange(54, 60)
    target, pairs = arm_inputs(rows, people[1:], primary, ids)
    before = matched_engine(ids, target, pairs).evaluate_outer(train, test, 1, query, score=False)
    omitted = arm_inputs(mutate(rows, [people[0]], ids), people[1:], primary, ids)
    assert omitted == (target, pairs)
    other = arm_inputs(mutate(rows, people[1:], ids, set(test) | set(query)), people[1:], primary, ids)
    for inputs in (omitted, other):
        engine = matched_engine(ids, *inputs)
        after = engine.evaluate_outer(train, test, 1, query, score=False)
        for key in ('selected', 'selection_scores', 'ridge_alpha', 'predictions', 'secondary_predictions'):
            assert after[key] == before[key]
        assert 'metrics' not in after  # Outer targets are scored only after saving.
        assert all(set(r['train']) <= set(train) for r in after['audit'])
        assert all(fit <= set(train) for fit, _ in engine.features.learned.fits)
    assert len(before['predictions']['stack']) == 12
    assert len(before['secondary_predictions']['stack']) == 6


@pytest.mark.parametrize('model', ['xgb', 'mlp'])
def test_actual_estimators_predict_secondary_without_extra_fits(model):
    ids, people, rows = annotations()
    inputs = arm_inputs(rows, people[1:], set(range(54)), ids)
    engine = matched_engine(ids, *inputs, model=model)
    # One real base procedure: the query combines primary and secondary records.
    engine.allowed, engine.outer_fold = set(range(42)), 1
    prediction = engine.fit_base(engine.configurations[1], np.arange(42), np.arange(42, 60), 'outer')
    assert prediction.shape == (18,) and np.isfinite(prediction).all()
    assert len(engine.timings) == 1 and engine.timings[0]['query_n'] == 18
    assert all(r['train'] <= set(range(42)) for r in
               [dict(train=set(x['train'])) for x in engine.audit])


def test_secondary_training_and_alias_overlap_are_rejected_before_fitting():
    ids, people, rows = annotations()
    inputs = arm_inputs(rows, people[1:], set(range(54)), ids)
    engine = matched_engine(ids, *inputs)
    with pytest.raises(ValueError, match='separate'):
        engine.evaluate_outer(np.arange(43), np.arange(43, 55), 1)
    with pytest.raises(ValueError, match='separate'):
        engine.evaluate_outer(np.arange(42), np.arange(42, 54), 1, [1, 54])
    groups = np.arange(60)
    groups[54] = groups[0]
    engine = matched_engine(ids, *inputs, groups=groups)
    with pytest.raises(AssertionError, match='overlap'):
        engine.evaluate_outer(np.arange(42), np.arange(42, 54), 1, np.arange(54, 60))
    assert not engine.timings


def test_query_targets_cannot_be_filled_and_primary_coverage_is_explicit():
    ids, people, rows = annotations()
    inputs = arm_inputs(rows, people[1:], set(range(54)), ids)
    engine = matched_engine(ids, *inputs)
    filled = engine.target.copy()
    filled[54:] = 0
    with pytest.raises(ValueError, match='not imputed'):
        NestedEvaluator(engine.features, filled, engine.configurations, engine.settings, np.arange(54))
    # Complete primary OOF coverage is 54, even though six additional feature rows exist.
    result = engine.run()
    assert result['complete'] and result['n_records'] == result['evaluated_n'] == 54


def test_pair_filter_uses_both_retained_votes_and_population_sd_only():
    ids, people, rows = annotations()
    row = rows[0]
    for p in people[1:]:
        row[f'{p} - Score A'], row[f'{p} - Score B'] = '1', '4'
    row[f'{people[0]} - Choice'] = 'B'
    assert len(arm_inputs([row], people[1:], {0, 1}, ids)[1]) == 1
    row[f'{people[2]} - Choice'] = 'UNSURE'
    assert not arm_inputs([row], people[1:], {0, 1}, ids)[1]
    # UNSURE removes the pair, but does not erase numerical ratings.
    assert ratings([row], people[1:], ids)[0, people[2]] == 1


def test_budget_terminates_an_unfinished_process_and_preserves_partial_file(tmp_path):
    partial = tmp_path / 'partial'
    command = [sys.executable, '-c',
               'import pathlib,time; pathlib.Path(' + repr(str(partial)) + ').write_text("checkpoint"); time.sleep(30)']
    status_path = tmp_path / 'budget.json'
    start = time.monotonic()
    assert supervised_process(command, .4, status_path) == 124
    assert time.monotonic() - start < 5
    assert partial.read_text() == 'checkpoint'
    assert json.loads(status_path.read_text())['status'] == 'budget_exhausted'


def test_versions_include_source_installed_xgboost():
    import xgboost
    versions = software_versions()
    assert versions['xgboost'] == xgboost.__version__
    assert all(versions.values())
