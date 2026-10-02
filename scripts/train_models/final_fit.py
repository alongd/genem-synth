"""Fit the nested-evaluation procedure once on the whole primary population.

`FinalFitEvaluator.fit_final` is `NestedEvaluator.evaluate_outer` with no outer test set:
inner-fold OOF selection, top-k choice, ridge alpha search and base refits all use only
`train`; `query` rows receive predictions and nothing else. Query targets must be NaN.
"""
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nested_evaluation import MidCDF, NestedEvaluator, make_splits


class FinalFitEvaluator(NestedEvaluator):
    def fit_final(self, train, query, fold_number=0):
        start = perf_counter()
        self.features.clear_transforms()
        train, query = np.asarray(train, dtype=int), np.asarray(query, dtype=int)
        if (set(train) != set(self.primary) or len(np.unique(train)) != len(train)
                or len(np.unique(query)) != len(query) or set(query) & set(self.primary)
                or not np.isnan(self.target[query]).all()):
            raise ValueError("Final fit trains on the whole primary population; query rows are label-free")
        self.allowed = set(map(int, train))
        self.outer_fold = int(fold_number)
        audit_start = len(self.audit)
        self.record('final_boundary', train, query)
        inner_splits = make_splits(self.target[train], self.settings.inner_folds, self.settings.seed,
                                   self.features.groups[train])
        oof = np.full((len(train), len(self.configurations)), np.nan)
        for inner_fold, (fit_positions, val_positions) in enumerate(inner_splits):
            print(f"final: inner {inner_fold + 1}/{len(inner_splits)}", flush=True)
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
        ridge = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge()), transformer=StandardScaler())
        meta = GridSearchCV(ridge, {'regressor__ridge__alpha': np.logspace(-3, 3, 50)},
                            scoring='neg_mean_squared_error',
                            cv=make_splits(self.target[train], min(5, len(set(self.features.groups[train]))),
                                           self.settings.seed, self.features.groups[train]),
                            n_jobs=1, error_score='raise')
        meta.fit(oof[:, selected], self.target[train])
        base = np.column_stack([
            self.fit_base(self.configurations[column], train, query, 'final') for column in selected
        ])
        predictions = {'stack': meta.predict(base), 'selected_single': base[:, 0]}
        baseline_tools = sorted(set(self.features.scores) | (
            {'ft'} if any('ft' in c.tools for c in self.configurations) else set()))
        raw_scores, raw_training_scores = {}, {}
        for tool in baseline_tools:
            self.record('baseline_' + tool, train, query)
            raw_score = self.features.raw(tool, train).reshape(-1, 1)
            raw_scores[tool] = raw_score[query, 0].tolist()
            raw_training_scores[tool] = raw_score[train, 0].tolist()
            baseline = make_pipeline(MidCDF(), LinearRegression()).fit(raw_score[train], self.target[train])
            predictions['baseline_' + tool] = baseline.predict(raw_score[query])
        return dict(
            train=train.tolist(), query=query.tolist(),
            selected=[self.configurations[j].name for j in selected],
            selection_scores={self.configurations[j].name: float(r2_score(self.target[train], oof[:, j]))
                              for j in ranking},
            ridge_alpha=float(meta.best_params_['regressor__ridge__alpha']),
            ridge_coefficients=meta.best_estimator_.regressor_[-1].coef_.tolist(),
            predictions={name: pred.tolist() for name, pred in predictions.items()},
            raw_scores=raw_scores, raw_training_scores=raw_training_scores,
            inner_splits=[dict(train=train[a].tolist(), test=train[b].tolist()) for a, b in inner_splits],
            seconds=perf_counter() - start,
            audit=self.audit[audit_start:])
