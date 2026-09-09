"""
interpretability_analysis.py

Reconstructs the Figure 7 interpretability analysis, which was never committed to
either repository. Follows the Methods description:

  "permutation importance was computed within the same five-fold stratified CV
   protocol used for model selection, including per-fold Bayesian hyperparameter
   optimization (Optuna, 10 trials). Within each test fold, all features belonging
   to a given group were permuted simultaneously and the drop in R^2 recorded"

Fold construction, scaling, PCA and the XGBoost search space are copied from
run_cv_pipeline.py so the numbers are directly comparable to the sweep.

Target config (paper): XGBoost on SA + SC + SYBA + FT-FSscore + CheMeleon PCA,
reported at OOF R2 = 0.592 +/- 0.012 (rank 25 of 187).

Outputs: data/output/interpretability_permutation.csv
         data/output/interpretability_ablation.csv
"""
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
N_OPTUNA = 10
N_PCA = 8
N_PERM_REPEATS = 5          # permutation is stochastic; average over repeats

ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "analysis_output" / "regression_cv" / "no_leakage"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

COL_SA = "sascore  (1–10, lower = easier to synthesize)"
COL_SC = "scscore (1–5, lower = easier to synthesize)"
COL_SYBA = "syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)"
COL_FT = "fsscore_ft"
SCORE_COLS = [COL_SA, COL_SC, COL_SYBA, COL_FT]
GROUPS = {"SAscore": [0], "SCScore": [1], "SYBA": [2], "FT-FSscore": [3],
          "CheMeleon PCA": list(range(4, 4 + N_PCA))}

df = pd.read_csv(CV_DIR / "scored_500_molecules.csv")
fp_cols = [c for c in df.columns if c.startswith("fp_")]
df = df.dropna(subset=SCORE_COLS + ["consensus_score"]).reset_index(drop=True)
y = df["consensus_score"].values.astype(float)
X_scores = df[SCORE_COLS].values.astype(float)
X_fp = df[fp_cols].values.astype(float)
print(f"{len(df)} molecules | {len(SCORE_COLS)} scores | {len(fp_cols)} fp -> PCA {N_PCA}")

bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=SEED).split(X_scores, bins))


def build(train_idx, test_idx):
    """Fit scalers and PCA on the training fold only, exactly as the pipeline does."""
    ss = StandardScaler()
    s_tr = ss.fit_transform(X_scores[train_idx])
    s_te = ss.transform(X_scores[test_idx])
    fs = StandardScaler()
    f_tr = fs.fit_transform(X_fp[train_idx])
    f_te = fs.transform(X_fp[test_idx])
    pca = PCA(n_components=N_PCA, random_state=SEED)
    f_tr = pca.fit_transform(f_tr)
    f_te = pca.transform(f_te)
    return np.hstack([s_tr, f_tr]), np.hstack([s_te, f_te])


rows, base_r2 = [], []
for k, (tr, te) in enumerate(folds):
    X_tr, X_te = build(tr, te)
    ys = StandardScaler()
    y_tr = ys.fit_transform(y[tr].reshape(-1, 1)).ravel()

    inn_tr, inn_val = train_test_split(np.arange(len(tr)), test_size=0.2,
                                       random_state=SEED)

    def objective(trial):
        p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
             "max_depth": trial.suggest_int("max_depth", 2, 10),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
             "random_state": SEED, "verbosity": 0}
        m = XGBRegressor(**p).fit(X_tr[inn_tr], y_tr[inn_tr])
        return mean_squared_error(y_tr[inn_val], m.predict(X_tr[inn_val]))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA)
    model = XGBRegressor(random_state=SEED, verbosity=0, **study.best_params)
    model.fit(X_tr, y_tr)

    inv = lambda p: ys.inverse_transform(p.reshape(-1, 1)).ravel()
    r2_full = r2_score(y[te], inv(model.predict(X_te)))
    base_r2.append(r2_full)
    print(f"  fold {k}: R2={r2_full:+.4f}")

    rng = np.random.default_rng(SEED)
    for gname, idxs in GROUPS.items():
        drops = []
        for _ in range(N_PERM_REPEATS):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idxs] = Xp[perm][:, idxs]      # permute the group jointly
            drops.append(r2_full - r2_score(y[te], inv(model.predict(Xp))))
        rows.append({"fold": k, "group": gname, "delta_r2": float(np.mean(drops))})

perm = (pd.DataFrame(rows).groupby("group")["delta_r2"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "delta_r2_mean", "std": "delta_r2_std"})
        .sort_values("delta_r2_mean", ascending=False))
perm.to_csv(OUT / "interpretability_permutation.csv", index=False)

print(f"\nfull model R2 = {np.mean(base_r2):.4f} +/- {np.std(base_r2, ddof=1):.4f}"
      f"   (paper 0.592 +/- 0.012)")
pub = {"CheMeleon PCA": 0.245, "SAscore": 0.229, "FT-FSscore": 0.072,
       "SYBA": 0.069, "SCScore": 0.068}
print(f"\n{'group':<16}{'ours':>10}{'+/-':>9}{'paper':>9}")
for _, r in perm.iterrows():
    print(f"{r['group']:<16}{r.delta_r2_mean:>10.3f}{r.delta_r2_std:>9.3f}"
          f"{pub.get(r['group'], float('nan')):>9.3f}")

# Ablation: Methods says these were "extracted directly from the full configuration sweep"
cv = pd.read_csv(CV_DIR / "cv_summary.csv")
get = lambda c: cv[(cv.model == "xgb") & (cv.config == c)]
full_cfg = get("sa_sc_syba_fss_ft_pca").iloc[0]
drops = {"CheMeleon PCA": "sa_sc_syba_fss_ft", "SAscore": "sc_syba_fss_ft_pca",
         "SCScore": "sa_syba_fss_ft_pca", "SYBA": "sa_sc_fss_ft_pca",
         "FT-FSscore": "sa_sc_syba_pca"}
ab = []
for g, c in drops.items():
    r = get(c)
    if r.empty:
        continue
    ab.append({"group": g, "config_without": c, "r2": r.iloc[0].r2_mean,
               "delta_r2": r.iloc[0].r2_mean - full_cfg.r2_mean})
pd.DataFrame(ab).to_csv(OUT / "interpretability_ablation.csv", index=False)
print(f"\nSaved -> {OUT/'interpretability_permutation.csv'}")
print(f"Saved -> {OUT/'interpretability_ablation.csv'}")
