"""
shap_analysis.py

Reconstructs SI Figures S3a-d, which were never committed to either repository.
Follows the Methods description:

  "SHAP values ... were computed via TreeSHAP on a separate XGBoost model trained
   on all 500 molecules. Mean SHAP values were aggregated by feature group and
   compared between the 76 shared-failure molecules and the remaining 424"

The published E[f(X)] of 3.618 equals the mean consensus score, so the SHAP model
was fitted to the raw target rather than a standardised one; we do the same.

Verification targets from the SI:
  S3b hard molecule    consensus 4.67, predicted 4.661, SA +0.40 largest contributor
  S3c easy molecule    consensus 2.00, predicted 1.864, SA -1.11
  S3d failure case     consensus 4.33, predicted 2.30, E[f(X)] = 3.616
                       SMILES [H]C12C3([H])C4([H])C1([H])C1([H])C2([H])C3([H])C41[H]

Outputs: data/output/shap_values.csv, data/output/shap_summary.txt
"""
import numpy as np
import optuna
import pandas as pd
import shap
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED, N_PCA, N_OPTUNA = 42, 8, 10

ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "analysis_output" / "regression_cv" / "no_leakage"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

COL_SA = "sascore  (1–10, lower = easier to synthesize)"
COL_SC = "scscore (1–5, lower = easier to synthesize)"
COL_SYBA = "syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)"
COL_FT = "fsscore_ft"
SCORES = [COL_SA, COL_SC, COL_SYBA, COL_FT]
NAMES = ["SA score", "SCScore", "SYBA", "FT-FSscore"] + [f"PCA-{i+1}" for i in range(N_PCA)]

df = pd.read_csv(CV_DIR / "scored_500_molecules.csv")
fp_cols = [c for c in df.columns if c.startswith("fp_")]
df = df.dropna(subset=SCORES + ["consensus_score"]).reset_index(drop=True)
y = df["consensus_score"].values.astype(float)

# Full-data fit: scalers and PCA on all 500, matching "trained on all 500 molecules"
s = StandardScaler().fit_transform(df[SCORES].values.astype(float))
f = StandardScaler().fit_transform(df[fp_cols].values.astype(float))
f = PCA(n_components=N_PCA, random_state=SEED).fit_transform(f)
X = np.hstack([s, f])
print(f"{X.shape[0]} molecules x {X.shape[1]} features | mean(y) = {y.mean():.4f}  (paper E[f(X)] = 3.618)")

tr, va = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)


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
    m = XGBRegressor(**p).fit(X[tr], y[tr])
    return mean_squared_error(y[va], m.predict(X[va]))


study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_OPTUNA)
model = XGBRegressor(random_state=SEED, verbosity=0, **study.best_params).fit(X, y)

# TreeSHAP via XGBoost's native pred_contribs. This is the same exact TreeSHAP
# algorithm the shap package would call, and it sidesteps a shap<->xgboost
# incompatibility: shap 0.49 cannot parse xgboost 3.2's array-valued base_score
# ("could not convert string to float: '[3.6177502E0]'").
import xgboost as xgb

contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
sv = contribs[:, :-1]          # per-feature SHAP values
base = float(contribs[0, -1])  # bias term == E[f(X)], identical for every row
pred = model.predict(X)
print(f"E[f(X)] = {base:.4f}   (paper 3.618 / 3.616)")

out = pd.DataFrame(sv, columns=[f"shap_{n}" for n in NAMES])
out.insert(0, "smiles", df["smiles"].values)
out["consensus"] = y
out["predicted"] = pred
out.to_csv(OUT / "shap_values.csv", index=False)

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=NAMES).sort_values(ascending=False)
print("\nmean |SHAP| (S3a beeswarm ranking):")
for n, v in mean_abs.items():
    print(f"  {n:<12}{v:.4f}")

CUBANE = "[H]C12C3([H])C4([H])C1([H])C1([H])C2([H])C3([H])C41[H]"
lines = [f"E[f(X)] = {base:.4f}", "", "mean |SHAP|:"]
lines += [f"  {n:<12}{v:.4f}" for n, v in mean_abs.items()]


def case(label, mask, target_cons, target_pred):
    if not mask.any():
        lines.append(f"\n{label}: no matching molecule"); print(f"\n{label}: not found"); return
    i = int(np.argmax(mask))
    contrib = pd.Series(sv[i], index=NAMES).sort_values(key=abs, ascending=False)
    hdr = (f"\n{label}: consensus={y[i]:.2f} predicted={pred[i]:.3f} "
           f"(paper consensus={target_cons} predicted={target_pred})")
    lines.append(hdr); print(hdr)
    for n, v in contrib.head(6).items():
        lines.append(f"    {n:<12}{v:+.3f}"); print(f"    {n:<12}{v:+.3f}")


case("S3b hard molecule", np.isclose(y, 14/3, atol=0.02), 4.67, 4.661)
case("S3c easy molecule", np.isclose(y, 2.0, atol=1e-9) & (pred < 2.4), 2.00, 1.864)
case("S3d failure case (cubane)", (df["smiles"].values == CUBANE), 4.33, 2.30)

(OUT / "shap_summary.txt").write_text("\n".join(lines) + "\n")
print(f"\nSaved -> {OUT/'shap_values.csv'}\nSaved -> {OUT/'shap_summary.txt'}")
