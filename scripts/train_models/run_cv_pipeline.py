"""
run_cv_pipeline.py

5-Fold Cross-Validation Regression Pipeline on 500 Unique Molecules.

Steps:
  0. Select best fine-tuned FSScore checkpoint (by pairwise accuracy on 208 HQ pairs)
  1. Score all 500 molecules: baseline tools (conda subprocesses) + FT FSScore + CheMeleon FP
  2. 5-fold stratified CV: LR, RF (Optuna), XGB (Optuna), MLP (Optuna)
     Each with 3 feature configs. Scalers/PCA fit on train fold only. Inner holdout
     for Optuna search ensures no test-fold leakage.
  3. Stacking ensemble: OOF predictions as meta-features, MLP meta-learner with Optuna.
  4. Outputs: per-fold metrics, OOF predictions, summary, bar-chart plot.

Run from repo root:
    conda run -n fsscore-env python scripts/train_models/run_cv_pipeline.py
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings

# Regression CV mode: "rowwise" (legacy, leaky single FT checkpoint) or
# "nl" (no-leakage, nested 5-fold FT — one FT model per regression outer fold).
REG_CV_MODE = os.environ.get("REG_CV_MODE", "rowwise").lower()
if REG_CV_MODE not in ("rowwise", "nl"):
    raise ValueError(f"REG_CV_MODE must be 'rowwise' or 'nl', got: {REG_CV_MODE!r}")
_REG_OUT_SUFFIX = os.environ.get("REG_OUT_SUFFIX", "")  # e.g. "_3x3"
_REG_K = int(os.environ.get("REG_K", "0")) or None
_NL_FT_SUFFIX = os.environ.get("NL_FT_SUFFIX", "")     # which cv_consensus_nl* dir to source nested FT
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestRegressor
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False
    print("WARNING: RandomForest not available — skipping RF models")

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("WARNING: XGBoost not available — skipping XGB models")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR      = Path(__file__).resolve().parents[2]
FSSCORE_DIR   = REPO_DIR / "fsscore"
SCORING_DIR   = REPO_DIR / "scripts" / "scoring"
FP_SCRIPT     = REPO_DIR / "scripts" / "fingerprints" / "add_fingerprints_chemeleon.py"
ALPHA_CKPTS   = FSSCORE_DIR / "models" / "checkpoints_alpha"
HQ_PAIRS_CSV  = REPO_DIR / "data" / "output" / "dummy_consensus_pairs.csv"  # overridden from config below
BASE_FSSCORE_CKPT = FSSCORE_DIR / "models" / "pretrain_graph_GGLGGL_ep242_best_valloss.ckpt"
if REG_CV_MODE == "nl":
    OUT_DIR = REPO_DIR / "analysis_output" / "regression_cv" / f"no_leakage{_REG_OUT_SUFFIX}"
    NL_FT_DIR = REPO_DIR / f"cv_consensus_nl{_NL_FT_SUFFIX}" / "regression_outer_folds"
else:
    OUT_DIR = REPO_DIR / "analysis_output" / "regression_cv"
    NL_FT_DIR = None

with open(REPO_DIR / "config.json") as _f:
    _config = json.load(_f)
REG_CSV = REPO_DIR / _config["training_input_file"]
HQ_PAIRS_CSV = REPO_DIR / _config.get("consensus_pairs_file", "data/output/dummy_consensus_pairs.csv")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_DIR = REPO_DIR / "models" / "cv_runs" / RUN_TIMESTAMP

OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORED_CACHE      = OUT_DIR / "scored_500_molecules.csv"
BEST_CKPT_CACHE   = OUT_DIR / "best_alpha_ckpt.txt"
OOF_CACHE         = OUT_DIR / "oof_predictions.csv"

SEED = 42

# ---------------------------------------------------------------------------
# FSScore imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(FSSCORE_DIR / "src"))
from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Column name constants (must match scoring script outputs exactly)
# ---------------------------------------------------------------------------
COL_SA     = "sascore  (1\u201310, lower = easier to synthesize)"
COL_SC     = "scscore (1\u20135, lower = easier to synthesize)"
COL_SYBA   = "syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)"
COL_RA_XGB = "rascore_xgb (0\u20131, higher = easier)"
COL_FSS    = "fsscore (unbounded relative scale, higher = easier to synthesize)"
COL_FSS_FT = "fsscore_ft"

# Base FSScore excluded; only fine-tuned FSScore used
SCORE_COLS = [COL_SA, COL_SC, COL_SYBA, COL_RA_XGB, COL_FSS_FT]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(y_true, y_pred):
    mse  = float(mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def prepare_features(X_scores, X_fp, train_idx, test_idx, use_fp, use_pca, score_indices=None):
    """Fit scalers on train fold, transform both. Returns X_train, X_test, scalers."""
    Xs_all_tr = X_scores[train_idx] if score_indices is None else X_scores[train_idx][:, score_indices]
    Xs_all_te = X_scores[test_idx]  if score_indices is None else X_scores[test_idx][:, score_indices]
    score_scaler = StandardScaler()
    Xs_tr = score_scaler.fit_transform(Xs_all_tr)
    Xs_te = score_scaler.transform(Xs_all_te)

    fp_scaler = pca_model = None
    if use_fp:
        fp_scaler = StandardScaler()
        Xf_tr = fp_scaler.fit_transform(X_fp[train_idx])
        Xf_te = fp_scaler.transform(X_fp[test_idx])
        if use_pca:
            pca_model = PCA(n_components=8, random_state=SEED)
            Xf_tr = pca_model.fit_transform(Xf_tr)
            Xf_te = pca_model.transform(Xf_te)
        X_tr = np.concatenate([Xs_tr, Xf_tr], axis=1)
        X_te = np.concatenate([Xs_te, Xf_te], axis=1)
    else:
        X_tr = Xs_tr
        X_te = Xs_te

    return X_tr, X_te, score_scaler, fp_scaler, pca_model


def build_mlp(input_dim, hidden_dim, num_layers, dropout):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden_dim, 1)]
    return nn.Sequential(*layers)


def train_mlp(model, X_tensor, y_tensor, lr, n_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor).squeeze(), y_tensor)
        loss.backward()
        optimizer.step()
    return model


# ---------------------------------------------------------------------------
# STEP 0: Select best fine-tuned FSScore checkpoint
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 0 — Selecting best FSScore alpha checkpoint")
print("=" * 70)


def select_best_checkpoint():
    if BEST_CKPT_CACHE.exists():
        line = BEST_CKPT_CACHE.read_text().strip()
        ckpt_path = Path(line.split("\t")[0])
        acc = float(line.split("\t")[1]) if "\t" in line else None
        print(f"  Cached best: {ckpt_path.parent.parent.name}  acc={acc}")
        return ckpt_path

    pairs_df = pd.read_csv(HQ_PAIRS_CSV)
    target = (pairs_df["consensus_choice"] == "A").astype(int).values
    unique_smiles = list(dict.fromkeys(
        pairs_df["smiles_a"].tolist() + pairs_df["smiles_b"].tolist()
    ))

    run_dirs = sorted(d for d in ALPHA_CKPTS.iterdir() if d.is_dir())
    best_acc, best_ckpt = -1.0, None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for run_dir in run_dirs:
            ckpts = sorted(run_dir.glob("ft_epoch=*-best_val_loss.ckpt"))
            if not ckpts:
                continue
            ckpt = ckpts[-1]

            model = LitRankNet.load_from_checkpoint(str(ckpt), map_location="cpu")
            scorer = Scorer(
                model=model, num_workers=0, device="cpu", batch_size=128,
                graph_datapath=str(tmpdir / "sel_graphs.pt"), keep_graphs=True,
            )
            raw = scorer.score(unique_smiles)
            score_dict = {s: float(raw[i]) for i, s in enumerate(unique_smiles)}

            for p in [tmpdir / "sel_graphs.pt", tmpdir / "processed"]:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()

            delta = np.array([
                score_dict[a] - score_dict[b]
                for a, b in zip(pairs_df["smiles_a"], pairs_df["smiles_b"])
            ])
            acc = float(((delta > 0).astype(int) == target).mean())
            print(f"  {run_dir.name}: acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_ckpt = ckpt

    print(f"\n  BEST: {best_ckpt.parent.parent.name}  acc={best_acc:.4f}")
    BEST_CKPT_CACHE.write_text(f"{best_ckpt}\t{best_acc:.6f}")
    return best_ckpt


if REG_CV_MODE == "nl":
    print("REG_CV_MODE=nl → skipping global checkpoint selection (per-fold FT performed later)")
    # Per-fold fine-tuning gives the honest CV estimate but produces no single
    # deployable checkpoint. For prediction on new molecules we use the "production"
    # checkpoint that finetune_production_fsscore.py trained on ALL pairs, and record
    # it in run_metadata.json so run_prediction.sh can find it.
    _prod = sorted((ALPHA_CKPTS / "production").glob("*.ckpt"))
    best_ckpt = _prod[-1] if _prod else None
    if best_ckpt is None:
        print("  WARNING: no production FT checkpoint found under "
              f"{ALPHA_CKPTS / 'production'}; run_prediction.sh will not work until "
              "scripts/scoring/finetune_production_fsscore.py has been run.")
    else:
        print(f"  Production FT checkpoint for prediction: {best_ckpt}")
else:
    best_ckpt = select_best_checkpoint()

# ---------------------------------------------------------------------------
# STEP 1: Score all 500 molecules
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1 — Scoring 500 molecules")
print("=" * 70)


def run_baseline_tools(smiles_df: pd.DataFrame) -> pd.DataFrame:
    """Score via conda subprocesses. Each tool reads/writes the same CSV in-place."""
    tools = [
        ("sascore-env",  "run_sascore.py"),
        ("scscore-env",  "run_scscore.py"),
        ("syba-env",     "run_syba.py"),
        ("rascore-env",  "run_rascore_xgb.py"),
        ("fsscore-env",  "run_fsscore.py"),
    ]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        working = Path(fh.name)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        tmp_out = Path(fh.name)

    smiles_df.to_csv(working, index=False)

    for env, script in tools:
        script_path = SCORING_DIR / script
        print(f"  [{script}] conda run -n {env} ...")
        result = subprocess.run(
            ["conda", "run", "-n", env,
             "python", str(script_path), str(working), str(tmp_out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    WARNING: {script} failed (exit {result.returncode})")
            print(f"    {result.stderr[-400:]}")
        else:
            working.write_bytes(tmp_out.read_bytes())

    scored = pd.read_csv(working)
    working.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return scored


def add_fsscore_ft(df: pd.DataFrame, ckpt: Path) -> pd.DataFrame:
    """Score all molecules with the best fine-tuned checkpoint."""
    smiles_list = df["smiles"].tolist()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model = LitRankNet.load_from_checkpoint(str(ckpt), map_location="cpu")
        scorer = Scorer(
            model=model, num_workers=0, device="cpu", batch_size=128,
            graph_datapath=str(tmpdir / "ft_500.pt"), keep_graphs=True,
        )
        scores = scorer.score(smiles_list)
    df[COL_FSS_FT] = [float(s) for s in scores]
    return df


def add_chemeleon_fp(df: pd.DataFrame) -> pd.DataFrame:
    """Add CheMeleon fingerprints via subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        tmp_in = Path(fh.name)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        tmp_out = Path(fh.name)

    df.to_csv(tmp_in, index=False)
    print("  Running CheMeleon fingerprints ...")
    result = subprocess.run(
        ["conda", "run", "-n", "chemprop-env",
         "python", str(FP_SCRIPT),
         "--input", str(tmp_in),
         "--output", str(tmp_out)],
        capture_output=True, text=True,
        cwd=str(REPO_DIR),
    )
    if result.returncode != 0:
        print(f"  WARNING: CheMeleon failed — fingerprint columns will be absent")
        print(f"  {result.stderr[-400:]}")
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        return df

    out = pd.read_csv(tmp_out)
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return out


if SCORED_CACHE.exists():
    print(f"  Loading cached scored dataset from {SCORED_CACHE.name}")
    scored_df = pd.read_csv(SCORED_CACHE)
    missing = [c for c in SCORE_COLS if c not in scored_df.columns]
    if missing:
        print(f"  Cache incomplete (missing: {missing}); re-scoring.")
        SCORED_CACHE.unlink()
        scored_df = None
else:
    scored_df = None

if scored_df is None:
    reg_df = pd.read_csv(REG_CSV)[["smiles", "consensus_score"]].dropna()
    print(f"  {len(reg_df)} molecules loaded from regression dataset")

    # Baseline tool scores
    smiles_input = reg_df[["smiles"]].copy()
    scored_df = run_baseline_tools(smiles_input)

    # Merge consensus_score back
    scored_df = scored_df.merge(reg_df[["smiles", "consensus_score"]], on="smiles", how="left")

    # Fine-tuned FSScore (skipped in NL mode — per-fold FT runs later)
    if REG_CV_MODE == "nl":
        print("  Adding fine-tuned FSScore: deferred (NL mode → per-fold nested FT)")
        # Placeholder column so SCORED_CACHE round-trips and SCORE_COLS dropna doesn't drop everything.
        scored_df[COL_FSS_FT] = 0.0
    else:
        print("  Adding fine-tuned FSScore ...")
        scored_df = add_fsscore_ft(scored_df, best_ckpt)

    # CheMeleon fingerprints
    scored_df = add_chemeleon_fp(scored_df)

    scored_df.to_csv(SCORED_CACHE, index=False)
    print(f"  Scored dataset cached ({len(scored_df)} rows, {len(scored_df.columns)} cols)")

print(f"  Score cols present: {[c for c in SCORE_COLS if c in scored_df.columns]}")

# ---------------------------------------------------------------------------
# STEP 2: Build feature matrices
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Building feature matrices")
print("=" * 70)

df_clean = scored_df.dropna(subset=["consensus_score"] + SCORE_COLS).reset_index(drop=True)
print(f"  {len(df_clean)} molecules after dropping NaN in scores/target")

y_all = df_clean["consensus_score"].values.astype(float)
fp_cols = [c for c in df_clean.columns if c.startswith("fp_")]
X_scores_all = df_clean[SCORE_COLS].values.astype(float)
X_fp_all     = df_clean[fp_cols].values.astype(float) if fp_cols else np.zeros((len(df_clean), 0))
print(f"  Score features: {len(SCORE_COLS)}   FP features: {len(fp_cols)}")

# Stratification bins for StratifiedKFold
n_bins = 5
y_bins = pd.qcut(y_all, q=n_bins, labels=False, duplicates="drop")

# ---------------------------------------------------------------------------
# STEP 3: 5-Fold CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — 5-Fold Stratified CV")
print("=" * 70)

N_FOLDS     = _REG_K if _REG_K is not None else 5
N_OPTUNA    = 10   # reduced from 30 for exhaustive combo search
MLP_EPOCHS  = 100

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X_scores_all, y_bins))

# ---------------------------------------------------------------------------
# NL mode: nested FT FSscore — one fine-tune per regression outer fold.
# Each fold-k FT model is trained ONLY on consensus pairs whose endpoints
# are both absent from the regression test molecules of fold k.
# Produces ft_scores_per_fold[i, k] for every molecule i and fold k.
# In STEP 3, fold k uses ft_scores_per_fold[:, k] as the fsscore_ft feature.
# ---------------------------------------------------------------------------
if REG_CV_MODE == "nl":
    print("\n" + "=" * 70)
    print("STEP 2.5 — Nested FT FSscore (NL mode)")
    print("=" * 70)
    NL_FT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_df_full = pd.read_csv(HQ_PAIRS_CSV)
    print(f"  {len(pairs_df_full)} consensus pairs available")

    # Late imports — finetune is heavy
    from fsscore.finetuning import finetune

    fss_ft_idx = SCORE_COLS.index(COL_FSS_FT)
    ft_scores_per_fold = np.full((len(df_clean), N_FOLDS), np.nan)
    all_smiles = df_clean["smiles"].tolist()

    for fold_idx, (train_idx_o, test_idx_o) in enumerate(folds):
        fold_label = f"fold_{fold_idx}"
        ft_dir = NL_FT_DIR / fold_label / "tuned"
        ckpt_marker = NL_FT_DIR / fold_label / "ft_done.ckpt_path"
        scores_csv = NL_FT_DIR / fold_label / "scores_500.csv"

        test_mols_k = set(df_clean.iloc[test_idx_o]["smiles"].tolist())
        is_clean = ~(pairs_df_full["smiles_a"].isin(test_mols_k) |
                     pairs_df_full["smiles_b"].isin(test_mols_k))
        train_pairs_k = pairs_df_full[is_clean].reset_index(drop=True)

        # Audit disjointness
        train_pair_mols = set(train_pairs_k["smiles_a"]).union(train_pairs_k["smiles_b"])
        leak = train_pair_mols & test_mols_k
        if leak:
            raise AssertionError(
                f"[fold {fold_idx}] {len(leak)} test mols leaked into FT training pairs"
            )
        print(f"  fold {fold_idx}: {len(train_pairs_k)} train pairs (filtered from {len(pairs_df_full)}); "
              f"test_mols={len(test_mols_k)}")

        (NL_FT_DIR / fold_label).mkdir(parents=True, exist_ok=True)
        train_pairs_k.to_csv(NL_FT_DIR / fold_label / "train_pairs.csv", index=False)

        if scores_csv.exists() and ckpt_marker.exists():
            print(f"    cached → loading scores from {scores_csv.name}")
            cached = pd.read_csv(scores_csv)
            score_map = dict(zip(cached["smiles"], cached["fsscore_ft"]))
            for i, sm in enumerate(all_smiles):
                ft_scores_per_fold[i, fold_idx] = score_map.get(sm, np.nan)
            continue

        # Fine-tune
        smiles_train = list(zip(train_pairs_k["smiles_a"], train_pairs_k["smiles_b"]))
        target_train = (train_pairs_k["consensus_choice"] == "A").astype(int).tolist()
        finetune(
            smiles=smiles_train,
            target=target_train,
            graph_datapath=None,
            filename=None,
            save_dir=str(ft_dir),
            model_path=str(BASE_FSSCORE_CKPT),
            featurizer="graph_2D",
            n_epochs=20,
            val_size=0.0,
            earlystopping=False,
            lr=1e-4,
            batch_size=32,
            num_workers=0,
            device="cpu",
            wandb_mode="disabled",
        )
        ckpts = sorted(ft_dir.rglob("*best*.ckpt")) or sorted(ft_dir.rglob("*.ckpt"))
        if not ckpts:
            raise RuntimeError(f"No checkpoint produced for fold {fold_idx} under {ft_dir}")
        ckpt = ckpts[-1]
        ckpt_marker.write_text(str(ckpt))
        print(f"    ckpt: {ckpt.name}")

        # Score all 500 molecules
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            model = LitRankNet.load_from_checkpoint(str(ckpt), map_location="cpu")
            scorer = Scorer(
                model=model, num_workers=0, device="cpu", batch_size=128,
                graph_datapath=str(tmpdir / f"ft_fold{fold_idx}.pt"), keep_graphs=True,
            )
            raw = scorer.score(all_smiles)
        scores_500 = [float(s) for s in raw]
        for i, s in enumerate(scores_500):
            ft_scores_per_fold[i, fold_idx] = s
        pd.DataFrame({"smiles": all_smiles, "fsscore_ft": scores_500}).to_csv(scores_csv, index=False)

    # Verify no NaNs
    if np.isnan(ft_scores_per_fold).any():
        raise RuntimeError("ft_scores_per_fold has NaNs — nested FT incomplete")

    # Sanity: dump per-fold OOF column (test rows only) — this is the "true OOF" fsscore_ft
    fsscore_ft_oof = np.full(len(df_clean), np.nan)
    for fold_idx, (_, test_idx_o) in enumerate(folds):
        fsscore_ft_oof[test_idx_o] = ft_scores_per_fold[test_idx_o, fold_idx]
    df_clean[COL_FSS_FT] = fsscore_ft_oof  # OOF view (used for diagnostics + cached output)
    pd.DataFrame({
        "smiles": df_clean["smiles"],
        "fsscore_ft_oof": fsscore_ft_oof,
        **{f"fsscore_ft_fold{k}": ft_scores_per_fold[:, k] for k in range(N_FOLDS)},
    }).to_csv(OUT_DIR / "fsscore_ft_nested.csv", index=False)
    # Refresh X_scores_all so the OOF fsscore_ft appears in caches; STEP 3 will swap per-fold below.
    X_scores_all = df_clean[SCORE_COLS].values.astype(float)
else:
    ft_scores_per_fold = None

# Feature configs: (name, use_fp, use_pca, score_indices)
# Programmatically generate all non-empty subsets of the 5 tools
# x 3 FP options (no FP / raw FP / PCA FP) = 93 configs total.
TOOL_NAMES = ["sa", "sc", "syba", "ra", "fss_ft"]


def _build_all_configs():
    cfgs = []
    for r in range(1, len(TOOL_NAMES) + 1):
        for combo in combinations(range(len(TOOL_NAMES)), r):
            name = "_".join(TOOL_NAMES[i] for i in combo)
            idxs = list(combo)
            cfgs.append((name,          False, False, idxs))
            cfgs.append((name + "_fp",  True,  False, idxs))
            cfgs.append((name + "_pca", True,  True,  idxs))
    return cfgs


CONFIGS = _build_all_configs()
print(f"  Total feature configs: {len(CONFIGS)}")  # 93

# Model types: XGB + MLP only (no LR, no RF)
MODEL_TYPES = []
if _XGB_AVAILABLE:
    MODEL_TYPES.append("xgb")
MODEL_TYPES.append("mlp")

# Storage: oof_preds[model_type][config_name] = np.zeros(N)
oof_preds = {mt: {cfg: np.full(len(df_clean), np.nan) for cfg, *_ in CONFIGS} for mt in MODEL_TYPES}
fold_splits = list(folds)  # saved for per-fold stacking metrics
cv_records = []
best_params_store = {}  # (model_type, cfg_name) -> best Optuna params from last fold

for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS}  "
          f"(train={len(train_idx)}, test={len(test_idx)}) ---")

    # NL mode: swap fsscore_ft column to this fold's nested-FT scores so the
    # FT model never saw any test molecule from this regression fold.
    if REG_CV_MODE == "nl":
        fss_ft_col_idx = SCORE_COLS.index(COL_FSS_FT)
        X_scores_all = X_scores_all.copy()
        X_scores_all[:, fss_ft_col_idx] = ft_scores_per_fold[:, fold_idx]

    # y scaler fitted on train fold only
    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_all[train_idx].reshape(-1, 1)).flatten()
    y_te_scaled = y_scaler.transform(y_all[test_idx].reshape(-1, 1)).flatten()

    # Inner holdout from training fold for Optuna (no test-fold leakage)
    inner_tr_idx, inner_val_idx = train_test_split(
        np.arange(len(train_idx)), test_size=0.2, random_state=SEED
    )
    # These are positions within train_idx, so actual indices:
    abs_inner_tr  = train_idx[inner_tr_idx]
    abs_inner_val = train_idx[inner_val_idx]

    for cfg_name, use_fp, use_pca, score_indices in CONFIGS:
        if use_fp and len(fp_cols) == 0:
            print(f"    [{cfg_name}] skipping — no FP columns available")
            continue

        cfg_model_types = MODEL_TYPES

        # Full train-fold and test-fold features
        X_tr, X_te, *_ = prepare_features(
            X_scores_all, X_fp_all, train_idx, test_idx, use_fp, use_pca, score_indices
        )
        # Inner features (for Optuna) — refit scalers on inner_tr only
        X_inn_tr, X_inn_val, *_ = prepare_features(
            X_scores_all, X_fp_all, abs_inner_tr, abs_inner_val, use_fp, use_pca, score_indices
        )

        y_inn_tr  = y_scaler.transform(y_all[abs_inner_tr].reshape(-1, 1)).flatten()
        y_inn_val = y_scaler.transform(y_all[abs_inner_val].reshape(-1, 1)).flatten()

        y_te_true = y_all[test_idx]  # original scale

        for model_type in cfg_model_types:
            # ---- LR (no Optuna) ----
            if model_type == "lr":
                model = LinearRegression()
                model.fit(X_tr, y_tr_scaled)
                pred_scaled = model.predict(X_te)

            # ---- RF ----
            elif model_type == "rf" and _RF_AVAILABLE:
                def _rf_objective(trial):
                    params = {
                        "n_estimators":    trial.suggest_int("n_estimators", 50, 300),
                        "max_depth":       trial.suggest_int("max_depth", 2, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                        "random_state":    SEED,
                    }
                    m = RandomForestRegressor(**params)
                    m.fit(X_inn_tr, y_inn_tr)
                    return mean_squared_error(y_inn_val, m.predict(X_inn_val))

                study = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler(seed=SEED))
                study.optimize(_rf_objective, n_trials=N_OPTUNA)
                best_p = study.best_params
                model = RandomForestRegressor(random_state=SEED, **best_p)
                model.fit(X_tr, y_tr_scaled)
                pred_scaled = model.predict(X_te)

            # ---- XGB ----
            elif model_type == "xgb" and _XGB_AVAILABLE:
                def _xgb_objective(trial):
                    params = {
                        "n_estimators":    trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3),
                        "max_depth":       trial.suggest_int("max_depth", 2, 10),
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 5.0),
                        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
                        "random_state":    SEED,
                        "verbosity":       0,
                    }
                    m = XGBRegressor(**params)
                    m.fit(X_inn_tr, y_inn_tr)
                    return mean_squared_error(y_inn_val, m.predict(X_inn_val))

                study = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler(seed=SEED))
                study.optimize(_xgb_objective, n_trials=N_OPTUNA)
                best_p = study.best_params
                model = XGBRegressor(random_state=SEED, verbosity=0, **best_p)
                model.fit(X_tr, y_tr_scaled)
                pred_scaled = model.predict(X_te)
                best_params_store[("xgb", cfg_name)] = best_p

            # ---- MLP ----
            elif model_type == "mlp":
                X_inn_tr_t  = torch.tensor(X_inn_tr,  dtype=torch.float32)
                X_inn_val_t = torch.tensor(X_inn_val, dtype=torch.float32)
                y_inn_tr_t  = torch.tensor(y_inn_tr,  dtype=torch.float32)
                y_inn_val_t = torch.tensor(y_inn_val, dtype=torch.float32)
                input_dim = X_tr.shape[1]

                def _mlp_objective(trial):
                    set_seed(SEED)
                    hd  = trial.suggest_int("hidden_dim", 32, 128)
                    nl  = trial.suggest_int("num_layers", 1, 3)
                    do  = trial.suggest_float("dropout", 0.0, 0.5)
                    lr  = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                    m   = build_mlp(input_dim, hd, nl, do)
                    m   = train_mlp(m, X_inn_tr_t, y_inn_tr_t, lr, MLP_EPOCHS)
                    m.eval()
                    with torch.no_grad():
                        preds = m(X_inn_val_t).squeeze().numpy()
                    return mean_squared_error(y_inn_val, preds)

                study = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler(seed=SEED))
                study.optimize(_mlp_objective, n_trials=N_OPTUNA)
                bp = study.best_params
                set_seed(SEED)
                model = build_mlp(input_dim, bp["hidden_dim"], bp["num_layers"], bp["dropout"])
                X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
                y_tr_t = torch.tensor(y_tr_scaled, dtype=torch.float32)
                model  = train_mlp(model, X_tr_t, y_tr_t, bp["lr"], MLP_EPOCHS)
                model.eval()
                X_te_t = torch.tensor(X_te, dtype=torch.float32)
                with torch.no_grad():
                    pred_scaled = model(X_te_t).squeeze().numpy()
                best_params_store[("mlp", cfg_name)] = {**bp, "input_dim": X_tr.shape[1]}

            else:
                continue

            # Inverse-transform predictions
            y_pred = y_scaler.inverse_transform(
                np.array(pred_scaled).reshape(-1, 1)
            ).flatten()

            # Store OOF predictions
            oof_preds[model_type][cfg_name][test_idx] = y_pred

            metrics = compute_metrics(y_te_true, y_pred)
            cv_records.append({
                "fold":       fold_idx + 1,
                "model":      model_type,
                "config":     cfg_name,
                **metrics,
            })
            print(f"    {model_type:4s} [{cfg_name:11s}]  "
                  f"R²={metrics['r2']:+.4f}  MAE={metrics['mae']:.4f}")

# ---------------------------------------------------------------------------
# Save OOF predictions (stacking column added after stacking step)
# ---------------------------------------------------------------------------
oof_df = df_clean[["smiles", "consensus_score"]].copy()
for mt in MODEL_TYPES:
    for cfg_name, *_ in CONFIGS:
        col = f"oof_{mt}_{cfg_name}"
        oof_df[col] = oof_preds[mt].get(cfg_name, np.full(len(df_clean), np.nan))
oof_df.to_csv(OOF_CACHE, index=False)
print(f"\nOOF predictions saved ({len(oof_df)} rows)")

# ---------------------------------------------------------------------------
# STEP 4: Stacking ensemble — Top-K OOF configs + Ridge meta-learner
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — Stacking Ensemble (Top-K OOF + Ridge)")
print("=" * 70)

from sklearn.linear_model import RidgeCV

TOP_K = _config.get("top_k", 10)  # number of top OOF columns to use as meta-features

# Rank all (model, config) pairs by OOF R²
ranked = []
for mt in MODEL_TYPES:
    for cfg_name, *_ in CONFIGS:
        preds = oof_preds[mt][cfg_name]
        valid = ~np.isnan(preds)
        if valid.sum() < 10:
            continue
        r2 = float(r2_score(y_all[valid], preds[valid]))
        ranked.append((r2, mt, cfg_name, preds))

ranked.sort(key=lambda x: x[0], reverse=True)
top_k = ranked[:TOP_K]

print(f"\n  Top-{TOP_K} configs selected for stacking:")
for r2v, mt, cfg, _ in top_k:
    print(f"    {mt:4s}  {cfg:40s}  OOF R²={r2v:.4f}")

# Build meta-feature matrix from top-K OOF columns
stack_valid_mask = np.ones(len(df_clean), dtype=bool)
for _, _, _, preds in top_k:
    stack_valid_mask &= ~np.isnan(preds)

X_stack = np.column_stack([p for _, _, _, p in top_k])[stack_valid_mask]
y_stack = y_all[stack_valid_mask]
print(f"\n  Meta-feature matrix: {X_stack.shape}  valid samples: {stack_valid_mask.sum()}")

# RidgeCV with 5-fold CV to select regularisation strength
alphas = np.logspace(-3, 3, 50)
stack_x_scaler = StandardScaler()
stack_y_scaler = StandardScaler()
X_s_scaled = stack_x_scaler.fit_transform(X_stack)
y_s_scaled = stack_y_scaler.fit_transform(y_stack.reshape(-1, 1)).flatten()

ridge_meta = RidgeCV(alphas=alphas, cv=5)
ridge_meta.fit(X_s_scaled, y_s_scaled)

stack_pred_scaled = ridge_meta.predict(X_s_scaled)
stack_pred = stack_y_scaler.inverse_transform(
    stack_pred_scaled.reshape(-1, 1)
).flatten()
stacking_r2 = float(r2_score(y_stack, stack_pred))
print(f"  Ridge meta-learner: alpha={ridge_meta.alpha_:.4f}  "
      f"train R²={stacking_r2:.4f} (in-sample for meta-learner)")
print("  Honest estimate: best single-config OOF R² printed above.")

# Save OOF stacking predictions
oof_df.loc[stack_valid_mask, "oof_stacking_ridge"] = stack_pred
oof_df.to_csv(OOF_CACHE, index=False)

# Per-fold stacking metrics — apply fitted ridge to each fold's OOF slice
for fi, (_, te_idx) in enumerate(fold_splits):
    try:
        X_fold = np.column_stack(
            [p[te_idx] for _, _, _, p in top_k]
        ).astype(float)
        X_fold_scaled = stack_x_scaler.transform(X_fold)
        sp = ridge_meta.predict(X_fold_scaled)
        y_fold_pred = stack_y_scaler.inverse_transform(
            sp.reshape(-1, 1)
        ).flatten()
        fold_metrics = compute_metrics(y_all[te_idx], y_fold_pred)
        cv_records.append({
            "fold":   fi + 1,
            "model":  "stacking_ridge",
            "config": f"top{TOP_K}_oof",
            **fold_metrics,
        })
    except Exception as e:
        print(f"  WARNING: stacking fold {fi+1} metrics failed: {e}")

stacking_results = {
    "top_k_configs": [
        {"model": mt, "config": cfg, "oof_r2": r2v}
        for r2v, mt, cfg, _ in top_k
    ],
    "ridge_alpha":       float(ridge_meta.alpha_),
    "stacking_train_r2": stacking_r2,
    "note": ("stacking_train_r2 is in-sample for the meta-learner. "
             "Honest estimate is the best single-config OOF R² above."),
}
with open(OUT_DIR / "stacking_results.json", "w") as f:
    json.dump(stacking_results, f, indent=2)
print(f"  stacking_results.json saved.")

# ---------------------------------------------------------------------------
# STEP 4b: Save top-K base models + stacking ensemble to timestamped directory
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"STEP 4b — Saving checkpoints → {MODEL_SAVE_DIR}")
print("=" * 70)

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

for r2v, mt, cfg_name, _ in top_k:
    cfg_tuple = next((c for c in CONFIGS if c[0] == cfg_name), None)
    if cfg_tuple is None:
        continue
    _, use_fp, use_pca, score_indices = cfg_tuple

    if use_fp and len(fp_cols) == 0:
        print(f"  Skipping {mt}/{cfg_name} — no FP columns")
        continue

    params = best_params_store.get((mt, cfg_name))
    if params is None:
        print(f"  Skipping {mt}/{cfg_name} — no stored params")
        continue

    # Fit scalers on full dataset
    Xs_full = X_scores_all[:, score_indices]
    score_scaler_f = StandardScaler()
    Xs_scaled_f = score_scaler_f.fit_transform(Xs_full)

    fp_scaler_f = pca_model_f = None
    if use_fp:
        fp_scaler_f = StandardScaler()
        Xf_scaled_f = fp_scaler_f.fit_transform(X_fp_all)
        if use_pca:
            pca_model_f = PCA(n_components=8, random_state=SEED)
            Xf_scaled_f = pca_model_f.fit_transform(Xf_scaled_f)
        X_full = np.concatenate([Xs_scaled_f, Xf_scaled_f], axis=1)
    else:
        X_full = Xs_scaled_f

    y_scaler_f = StandardScaler()
    y_scaled_f = y_scaler_f.fit_transform(y_all.reshape(-1, 1)).flatten()

    set_seed(SEED)
    if mt == "xgb":
        final_model = XGBRegressor(random_state=SEED, verbosity=0, **params)
        final_model.fit(X_full, y_scaled_f)
    elif mt == "mlp":
        final_model = build_mlp(
            X_full.shape[1], params["hidden_dim"], params["num_layers"], params["dropout"]
        )
        X_full_t = torch.tensor(X_full, dtype=torch.float32)
        y_full_t = torch.tensor(y_scaled_f, dtype=torch.float32)
        final_model = train_mlp(final_model, X_full_t, y_full_t, params["lr"], MLP_EPOCHS)
        final_model.eval()
    else:
        continue

    model_dir = MODEL_SAVE_DIR / f"{mt}_{cfg_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if mt == "mlp":
        torch.save(final_model.state_dict(), model_dir / "model.pt")
    else:
        joblib.dump(final_model, model_dir / "model.pkl")

    joblib.dump(score_scaler_f, model_dir / "score_scaler.pkl")
    joblib.dump(y_scaler_f, model_dir / "y_scaler.pkl")
    if fp_scaler_f is not None:
        joblib.dump(fp_scaler_f, model_dir / "fp_scaler.pkl")
    if pca_model_f is not None:
        joblib.dump(pca_model_f, model_dir / "pca.pkl")

    with open(model_dir / "best_params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "model_type":  mt,
            "config":      cfg_name,
            "oof_r2":      r2v,
            "use_fp":      use_fp,
            "use_pca":     use_pca,
            "score_cols":  [SCORE_COLS[i] for i in score_indices],
        }, f, indent=2)

    print(f"  Saved {mt}/{cfg_name}  (OOF R²={r2v:.4f})")

# Save stacking ensemble
stack_dir = MODEL_SAVE_DIR / "stacking_ensemble"
stack_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(ridge_meta, stack_dir / "ridge_meta.pkl")
joblib.dump(stack_x_scaler, stack_dir / "stack_x_scaler.pkl")
joblib.dump(stack_y_scaler, stack_dir / "stack_y_scaler.pkl")
with open(stack_dir / "stacking_results.json", "w") as f:
    json.dump(stacking_results, f, indent=2)

# Save run metadata
with open(MODEL_SAVE_DIR / "run_metadata.json", "w") as f:
    json.dump({
        "timestamp":               RUN_TIMESTAMP,
        "top_k":                   TOP_K,
        "n_folds":                 N_FOLDS,
        "n_molecules":             len(df_clean),
        "seed":                    SEED,
        "best_fsscore_checkpoint": str(best_ckpt),
    }, f, indent=2)

print(f"\n  All checkpoints saved → {MODEL_SAVE_DIR}")

# Update config.json so the prediction pipeline picks up this checkpoint
with open(REPO_DIR / "config.json") as _f:
    _cfg = json.load(_f)
_cfg["checkpoint_dir"] = str(MODEL_SAVE_DIR)
with open(REPO_DIR / "config.json", "w") as _f:
    json.dump(_cfg, _f, indent=2)
print(f"  config.json updated: checkpoint_dir = {MODEL_SAVE_DIR}")

# ---------------------------------------------------------------------------
# Save per-fold results (base models + individual tools + stacking)
# ---------------------------------------------------------------------------
cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

# Summary: mean ± std per model × config
summary_rows = []
for (model_type, cfg_name), grp in cv_df.groupby(["model", "config"]):
    summary_rows.append({
        "model":    model_type,
        "config":   cfg_name,
        "r2_mean":  grp["r2"].mean(),
        "r2_std":   grp["r2"].std(),
        "mae_mean": grp["mae"].mean(),
        "mae_std":  grp["mae"].std(),
        "mse_mean": grp["mse"].mean(),
        "mse_std":  grp["mse"].std(),
    })
summary_df = pd.DataFrame(summary_rows).sort_values("r2_mean", ascending=False)
summary_df.to_csv(OUT_DIR / "cv_summary.csv", index=False)

print("\n\nCV Summary (mean R² across 5 folds):")
print(summary_df[["model", "config", "r2_mean", "r2_std", "mae_mean"]].to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 5: Plot
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Plotting")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 5))
labels, means, stds, colors_list = [], [], [], []
palette = {"xgb": "#66BB6A", "mlp": "#FFA726", "stacking_ridge": "#AB47BC"}

for _, row in summary_df.iterrows():
    labels.append(f"{row['model']}\n{row['config']}")
    means.append(row["r2_mean"])
    stds.append(row["r2_std"])
    colors_list.append(palette.get(row["model"], "#BDBDBD"))

x = np.arange(len(labels))
bars = ax.bar(x, means, 0.6, color=colors_list, zorder=3)
ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("OOF R² (mean ± std, 5 folds)", fontsize=11)
ax.set_title("5-Fold CV Regression: R² by Model and Feature Config\n"
             f"(500 molecules, target = consensus synthesizability score)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Legend for model colours
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=mt.upper()) for mt, c in palette.items()]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_DIR / "cv_plot.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'cv_plot.png'}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset:    {len(df_clean)} molecules (after NaN filtering)")
print(f"  Features:   {len(SCORE_COLS)} scores + {len(fp_cols)} FP dims")
print(f"  CV:         {N_FOLDS}-fold stratified, Optuna {N_OPTUNA} trials/model")
print(f"\n  OOF R² (honest, 5-fold):")
for _, row in summary_df.iterrows():
    print(f"    {row['model']:4s} [{row['config']:11s}]  "
          f"R² = {row['r2_mean']:+.4f} ± {row['r2_std']:.4f}  "
          f"MAE = {row['mae_mean']:.4f}")
print(f"\nAll outputs in: {OUT_DIR}")
