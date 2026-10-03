"""Nested evaluation of an entire supervised feature/selection/stacking procedure."""

from dataclasses import asdict, dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, StratifiedGroupKFold, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Candidate:
    model: str
    tools: tuple
    representation: str = "none"

    @property
    def name(self):
        return f"{self.model}:{'+'.join(self.tools)}:{self.representation}"


@dataclass(frozen=True)
class Settings:
    outer_folds: int = 5
    inner_folds: int = 5
    trials: int = 10
    top_k: int = 10
    mlp_epochs: int = 100
    seed: int = 42
    threads: int = 2

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("At least two outer and inner folds are required")
        if min(self.trials, self.top_k, self.mlp_epochs, self.threads) < 1:
            raise ValueError("Trial, model, epoch, and thread counts must be positive")


def candidates(pilot=False):
    if pilot:
        feature_sets = [(('sa',), 'none')]
        feature_sets += [(('sa', 'sc', 'ft'), rep) for rep in ('none', 'pca', 'full')]
    else:
        tools = ('sa', 'sc', 'syba', 'ra', 'ft')
        feature_sets = [
            (subset, rep) for k in range(1, len(tools) + 1)
            for subset in combinations(tools, k) for rep in ('none', 'pca', 'full')
        ]
    return [Candidate(model, subset, rep)
            for model in ('xgb', 'mlp') for subset, rep in feature_sets]


def make_splits(y, folds, seed, groups=None):
    """Bin training targets only; use KFold if stratification cannot populate folds."""
    y = np.asarray(y)
    if folds > len(y):
        raise ValueError("More folds than observations")
    edges = np.unique(np.quantile(y, np.linspace(0, 1, 6)))
    bins = np.searchsorted(edges[1:-1], y, side='right')
    counts = np.bincount(bins)
    if groups is not None:
        if len(set(groups)) < folds:
            raise ValueError("Fewer molecular groups than folds")
        if len(counts[counts > 0]) > 1:
            return list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(y, bins, groups))
        return list(GroupKFold(folds).split(y, groups=groups))
    if len(counts[counts > 0]) > 1 and counts[counts > 0].min() >= folds:
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(y, bins))
    return list(KFold(folds, shuffle=True, random_state=seed).split(y))


def metrics(y, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction; evaluation failed")
    mse = float(mean_squared_error(y, prediction))
    return dict(r2=float(r2_score(y, prediction)), mae=float(mean_absolute_error(y, prediction)),
                mse=mse, rmse=float(np.sqrt(mse)))


class MidCDF(TransformerMixin, BaseEstimator):
    """Mean of empirical strict and inclusive CDFs, fitted on training scores only."""

    def fit(self, X, y=None):
        self.sorted_ = np.sort(np.asarray(X).reshape(-1))
        return self

    def transform(self, X):
        values = np.asarray(X).reshape(-1)
        lo = np.searchsorted(self.sorted_, values, side='left')
        hi = np.searchsorted(self.sorted_, values, side='right')
        return ((lo + hi) / (2 * len(self.sorted_))).reshape(-1, 1)


class TorchMLP(RegressorMixin, BaseEstimator):
    """The reproduction script's network and optimizer, with an estimator interface."""

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.0, lr=1e-3,
                 epochs=100, seed=42):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        layers = [nn.Linear(X.shape[1], self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(self.num_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        self.model_ = nn.Sequential(*layers, nn.Linear(self.hidden_dim, 1))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        features = torch.as_tensor(X, dtype=torch.float32)
        target = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.model_(features).reshape(-1), target)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            return self.model_(torch.as_tensor(X, dtype=torch.float32)).reshape(-1).numpy()


def build_model(candidate, params, settings):
    if candidate.model == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=settings.seed, n_jobs=settings.threads, verbosity=0, **params)
    if candidate.model == 'mlp':
        return TorchMLP(epochs=settings.mlp_epochs, seed=settings.seed, **params)
    if candidate.model == 'linear':  # Lightweight deterministic integration tests.
        return Ridge(alpha=params.get('alpha', 1.0))
    raise ValueError(f"Unknown model: {candidate.model}")


def trial_parameters(trial, model):
    if model == 'xgb':
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0))
    if model == 'mlp':
        return dict(hidden_dim=trial.suggest_int('hidden_dim', 32, 128),
                    num_layers=trial.suggest_int('num_layers', 1, 3),
                    dropout=trial.suggest_float('dropout', 0.0, 0.5),
                    lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))
    return dict(alpha=trial.suggest_float('alpha', 0.1, 10.0, log=True))


class FeatureSpace:
    """Frozen raw features plus an optional training-scope-specific learned feature."""

    def __init__(self, ids, scores, embeddings, learned=None, groups=None):
        self.ids = np.asarray(ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Molecule identifiers must be unique")
        self.groups = np.asarray(groups if groups is not None else ids)
        if self.groups.shape != self.ids.shape:
            raise ValueError("Molecular groups must align with row identifiers")
        self.scores = {name: np.asarray(values, dtype=float) for name, values in scores.items()}
        if 'ft' in self.scores:
            raise ValueError("A global fine-tuned score is not an admissible frozen feature")
        self.embeddings = np.asarray(embeddings, dtype=float)
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.ids):
            raise ValueError("Embeddings must have one row per molecule")
        for values in [*self.scores.values(), self.embeddings]:
            if len(values) != len(self.ids) or not np.isfinite(values).all():
                raise ValueError("Features have missing, non-finite, or misaligned values")
        self.learned = learned
        self.cache = {}
        self.embedding_cache = {}
        self.fit_records = []

    def clear_transforms(self):
        """Release fold-local matrices; the learned provider retains its small score cache."""
        self.cache.clear()
        self.embedding_cache.clear()

    def raw(self, tool, train):
        if tool == 'ft':
            if self.learned is None:
                raise ValueError("A training-scoped feature provider is required")
            values = np.asarray(self.learned(np.asarray(train)), dtype=float)
            if values.shape != (len(self.ids),) or not np.isfinite(values).all():
                raise ValueError("Learned features are incomplete or non-finite")
            return values
        return self.scores[tool]

    def arrays(self, candidate, train, query):
        train = np.asarray(train, dtype=int)
        key = (tuple(train), candidate.tools, candidate.representation)
        if key not in self.cache:
            raw = np.column_stack([self.raw(tool, train) for tool in candidate.tools])
            scaler = StandardScaler().fit(raw[train])
            X = scaler.transform(raw)
            self.fit_records.append(dict(train=train.tolist(), tools=list(candidate.tools),
                                         representation=candidate.representation))
            if candidate.representation != 'none':
                ekey = (tuple(train), candidate.representation)
                if ekey not in self.embedding_cache:
                    if not self.embeddings.shape[1]:
                        raise ValueError("Embedding configuration requested without embeddings")
                    escaler = StandardScaler().fit(self.embeddings[train])
                    embedded = escaler.transform(self.embeddings)
                    if candidate.representation == 'pca':
                        if min(embedded[train].shape) < 8:
                            raise ValueError("PCA-8 requires at least eight training rows and dimensions")
                        pca = PCA(n_components=8, random_state=42).fit(embedded[train])
                        embedded = pca.transform(embedded)
                    elif candidate.representation != 'full':
                        raise ValueError("Unknown embedding representation")
                    self.embedding_cache[ekey] = embedded
                X = np.column_stack([X, self.embedding_cache[ekey]])
            self.cache[key] = X
        X = self.cache[key]
        return X[train], X[query]


class NestedEvaluator:
    # Permit query rows that are the same molecular structures as training rows. Off by default:
    # the group check in `record` is what keeps a held-out evaluation honest, and nothing in the
    # ordinary evaluation path turns it off. Only a run whose purpose is to quantify that
    # contamination, and that reports the affected query rows as in-sample, may set it.
    allow_query_group_overlap = False

    def __init__(self, features, target, configurations, settings, primary_indices=None):
        self.features = features
        self.target = np.asarray(target, dtype=float)
        self.primary = (np.arange(len(features.ids)) if primary_indices is None
                        else np.asarray(primary_indices, dtype=int))
        if (self.primary.ndim != 1 or len(self.primary) < 2
                or len(set(self.primary)) != len(self.primary)
                or self.primary.min() < 0 or self.primary.max() >= len(features.ids)):
            raise ValueError("Invalid primary population")
        if (self.target.shape != (len(features.ids),)
                or not np.isfinite(self.target[self.primary]).all()):
            raise ValueError("Target must be finite for each primary molecule")
        secondary = np.setdiff1d(np.arange(len(features.ids)), self.primary)
        if len(secondary) and not np.isnan(self.target[secondary]).all():
            raise ValueError("Query-only targets must be absent, not imputed")
        self.configurations = list(configurations)
        if not self.configurations or len({c.name for c in self.configurations}) != len(self.configurations):
            raise ValueError("Configurations must be nonempty and unique")
        if settings.top_k > len(self.configurations):
            raise ValueError("top_k exceeds the configuration count")
        self.settings = settings
        self.audit = []
        self.timings = []

    def record(self, stage, train, query=()):
        train, query = set(map(int, train)), set(map(int, query))
        if not train <= self.allowed or train & query:
            raise AssertionError(f"Held-out leakage in {stage}")
        if (not self.allow_query_group_overlap
                and set(self.features.groups[list(train)]) & set(self.features.groups[list(query)])):
            raise AssertionError(f"Molecular representation overlap in {stage}")
        self.audit.append(dict(outer_fold=self.outer_fold, stage=stage,
                               train=sorted(train), query=sorted(query)))

    def fit_base(self, candidate, train, query, role):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        start = perf_counter()
        train, query = np.asarray(train), np.asarray(query)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.settings.seed).split(
            train, groups=self.features.groups[train]))
        tuning_train, tuning_val = train[a], train[b]
        self.record('base_tuning', tuning_train, tuning_val)
        feature_start = perf_counter()
        X_tune, X_val = self.features.arrays(candidate, tuning_train, tuning_val)
        feature_seconds = perf_counter() - feature_start
        # Target scaling belongs to the same fitting subset as feature scaling.
        def objective(trial):
            model = TransformedTargetRegressor(
                regressor=build_model(candidate, trial_parameters(trial, candidate.model), self.settings),
                transformer=StandardScaler())
            model.fit(X_tune, self.target[tuning_train])
            prediction = model.predict(X_val)
            return metrics(self.target[tuning_val], prediction)['mse']

        study = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=self.settings.seed))
        study.optimize(objective, n_trials=self.settings.trials)
        self.record('base_' + role, train, query)
        feature_start = perf_counter()
        X_train, X_query = self.features.arrays(candidate, train, query)
        feature_seconds += perf_counter() - feature_start
        model = TransformedTargetRegressor(
            regressor=build_model(candidate, study.best_params, self.settings),
            transformer=StandardScaler())
        model.fit(X_train, self.target[train])
        prediction = model.predict(X_query)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite predictions from {candidate.name}")
        elapsed = perf_counter() - start
        self.timings.append(dict(outer_fold=self.outer_fold, role=role, candidate=candidate.name,
                                 model=candidate.model, representation=candidate.representation,
                                 train_n=len(train), query_n=len(query), trials=self.settings.trials,
                                 seconds=elapsed, feature_seconds=feature_seconds,
                                 model_seconds=elapsed - feature_seconds, best_params=study.best_params))
        return prediction

    def evaluate_outer(self, train, test, fold_number, secondary_query=(), score=True):
        start = perf_counter()
        self.features.clear_transforms()
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        secondary_query = np.asarray(secondary_query, dtype=int)
        if (not set(train) <= set(self.primary) or not set(test) <= set(self.primary)
                or len(set(secondary_query)) != len(secondary_query)
                or set(secondary_query) & set(self.primary)
                or any(i < 0 or i >= len(self.target) for i in secondary_query)):
            raise ValueError("Primary fitting and secondary query populations must be separate")
        query = np.concatenate([test, secondary_query])
        if (len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test)
                or np.intersect1d(train, test).size or min(len(train), len(test)) < 2
                or min(train.min(), test.min()) < 0
                or max(train.max(), test.max()) >= len(self.target)):
            raise ValueError("Outer split has overlap, duplicates, or invalid indices")
        self.allowed = set(map(int, train))
        self.outer_fold = int(fold_number)
        audit_start = len(self.audit)
        self.record('outer_boundary', train, query)
        inner_splits = make_splits(self.target[train], self.settings.inner_folds, self.settings.seed,
                                  self.features.groups[train])
        oof = np.full((len(train), len(self.configurations)), np.nan)
        for inner_fold, (fit_positions, val_positions) in enumerate(inner_splits):
            print(f"outer {fold_number}: inner {inner_fold + 1}/{len(inner_splits)}", flush=True)
            for column, candidate in enumerate(self.configurations):
                oof[val_positions, column] = self.fit_base(
                    candidate, train[fit_positions], train[val_positions], 'inner')
                print(f"  {candidate.name}", flush=True)
            self.features.clear_transforms()
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete inner OOF matrix")
        self.record('configuration_selection', train)
        ranking = sorted(range(oof.shape[1]), key=lambda j: (
            -r2_score(self.target[train], oof[:, j]), self.configurations[j].name))
        selected = ranking[:self.settings.top_k]
        self.record('meta_fitting_and_alpha_selection', train)
        # This CV tunes alpha only. All its inputs are inside the outer training set.
        # Only the untouched outer predictions estimate the complete procedure's error.
        ridge = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge()), transformer=StandardScaler())
        meta = GridSearchCV(ridge, {'regressor__ridge__alpha': np.logspace(-3, 3, 50)},
                            scoring='neg_mean_squared_error',
                            cv=make_splits(self.target[train], min(5, len(set(self.features.groups[train]))),
                                           self.settings.seed, self.features.groups[train]),
                            n_jobs=1, error_score='raise')
        meta.fit(oof[:, selected], self.target[train])
        outer_features = np.column_stack([
            self.fit_base(self.configurations[column], train, query, 'outer') for column in selected
        ])
        predictions = {'stack': meta.predict(outer_features), 'selected_single': outer_features[:, 0]}
        baseline_tools = sorted(set(self.features.scores) | (
            {'ft'} if any('ft' in c.tools for c in self.configurations) else set()))
        for tool in baseline_tools:
            self.record('baseline_' + tool, train, query)
            raw_score = self.features.raw(tool, train).reshape(-1, 1)
            baseline = make_pipeline(MidCDF(), LinearRegression()).fit(raw_score[train], self.target[train])
            predictions['baseline_' + tool] = baseline.predict(raw_score[query])
        result = dict(
            outer_fold=int(fold_number), train=train.tolist(), test=test.tolist(),
            selected=[self.configurations[j].name for j in selected],
            selection_scores={self.configurations[j].name: float(r2_score(self.target[train], oof[:, j]))
                              for j in ranking},
            ridge_alpha=float(meta.best_params_['regressor__ridge__alpha']),
            predictions={name: pred[:len(test)].tolist() for name, pred in predictions.items()},
            secondary_query=secondary_query.tolist(),
            secondary_predictions={name: pred[len(test):].tolist() for name, pred in predictions.items()},
            inner_splits=[dict(train=train[a].tolist(), test=train[b].tolist()) for a, b in inner_splits],
            seconds=perf_counter() - start,
            audit=self.audit[audit_start:])
        if score:
            result['metrics'] = {name: metrics(self.target[test], pred[:len(test)])
                                 for name, pred in predictions.items()}
        return result

    def run(self, outer_limit=None, splits=None, on_fold=None):
        if splits is None:
            splits = [(self.primary[a], self.primary[b]) for a, b in make_splits(
                self.target[self.primary], self.settings.outer_folds, self.settings.seed,
                self.features.groups[self.primary])]
        if outer_limit is not None and not 1 <= outer_limit <= len(splits):
            raise ValueError("outer_limit is outside the available folds")
        chosen = splits if outer_limit is None else splits[:outer_limit]
        seen_groups = set()
        for _, test in chosen:
            test_groups = set(self.features.groups[test])
            if test_groups & seen_groups:
                raise ValueError("A molecular group appears in multiple outer test folds")
            seen_groups.update(test_groups)
        results = []
        for i, (train, test) in enumerate(chosen):
            results.append(self.evaluate_outer(train, test, i + 1))
            if on_fold is not None:
                on_fold(self.summarize(results, len(splits)))
        return self.summarize(results, len(splits))

    def summarize(self, results, planned_folds):
        indices = np.concatenate([r['test'] for r in results])
        if len(set(indices)) != len(indices):
            raise ValueError("Outer test folds overlap")
        pooled = {}
        for name in results[0]['predictions']:
            prediction = np.concatenate([r['predictions'][name] for r in results])
            pooled[name] = metrics(self.target[indices], prediction)
        fold_summary = {}
        for name in pooled:
            fold_summary[name] = {}
            for metric in pooled[name]:
                values = [r['metrics'][name][metric] for r in results]
                fold_summary[name][metric] = dict(mean=float(np.mean(values)),
                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        return dict(complete=len(results) == planned_folds and set(indices) == set(self.primary),
                    n_records=len(self.primary), n_molecular_groups=len(set(self.features.groups[self.primary])),
                    evaluated_folds=len(results), planned_folds=planned_folds, evaluated_n=len(indices),
                    settings=asdict(self.settings), configurations=[c.name for c in self.configurations],
                    pooled_evaluated_metrics=pooled, fold_mean_and_sd=fold_summary,
                    folds=list(results), timings=list(self.timings),
                    note='Only outer predictions evaluate the complete procedure; a partial run is not a final score.')
