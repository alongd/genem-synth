import argparse, json, os, random, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

# ---------- Import project finetune() and scoring ----------
from fsscore.finetuning import finetune
from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer


# ---------- Utilities ----------
def binom_ci_halfwidth(p, n, z=1.96):
    return float(z*np.sqrt(max(p*(1-p), 1e-12)/max(n,1)))

def ensure_cols(df: pd.DataFrame, *cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

def compute_metrics(df, label_col, a_scores, b_scores):
    y = (df[label_col].astype(str).str.upper() == "A").astype(int).to_numpy()
    delta = (np.asarray(a_scores) - np.asarray(b_scores))
    pred  = (delta > 0).astype(int)

    n = len(df)
    acc = float((pred == y).mean())
    ci  = binom_ci_halfwidth(acc, n)
    auc = float(roc_auc_score(y, delta))
    return {"n": int(n), "acc": acc, "ci95_half": ci, "auc": auc, "y": y, "pred": pred, "delta": delta}

def mcnemar_from_preds(y, predA, predB):
    base_correct = (predA == y)
    ft_correct   = (predB == y)
    a = int(np.sum( base_correct &  ft_correct))
    b = int(np.sum( base_correct & ~ft_correct))
    c = int(np.sum(~base_correct &  ft_correct))
    d = int(np.sum(~base_correct & ~ft_correct))
    chi2 = float(((b - c) ** 2) / max(b + c, 1e-12))
    p_approx = float(np.exp(-chi2 / 2.0))  # chi^2(1) tail approx
    return {"a": a, "b": b, "c": c, "d": d, "chi2_no_correction": chi2, "p_approx": p_approx}

def plot_fold_roc(y, delta, title, out_png):
    fpr, tpr, _ = roc_curve(y, delta)
    plt.figure()
    plt.plot(fpr, tpr, label=title)
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close()


# ---------- Fold construction (molecule-held-out) ----------
def build_folds_molecule_heldout(df, smiles_i_col="smiles_i", smiles_j_col="smiles_j", k=5, seed=42):
    """Strict molecule-disjoint K-fold.

    Every unique molecule is assigned to exactly one bucket. For fold f:
      - test pairs:  BOTH endpoints in bucket f
      - train pairs: BOTH endpoints in any bucket other than f
      - spanning pairs (one endpoint in f, other in another bucket) are DROPPED for fold f

    This guarantees zero molecule overlap between train and test. The cost is that
    some pairs are unused in some folds. Across all k folds, every pair appears in
    train in (k-1) folds and in test for at most one fold (only if both endpoints
    happen to land in the same bucket).
    """
    ensure_cols(df, smiles_i_col, smiles_j_col)
    rng = random.Random(seed)
    uniq_mols = sorted(set(df[smiles_i_col].astype(str)) | set(df[smiles_j_col].astype(str)))
    rng.shuffle(uniq_mols)

    bucket_of = {}
    for idx, m in enumerate(uniq_mols):
        bucket_of[m] = idx % k

    folds = []
    for f in range(k):
        a_in_f = df[smiles_i_col].map(bucket_of) == f
        b_in_f = df[smiles_j_col].map(bucket_of) == f
        is_test  = a_in_f & b_in_f
        is_train = (~a_in_f) & (~b_in_f)
        df_test  = df[is_test].copy()
        df_train = df[is_train].copy()
        folds.append((df_train, df_test, f))
    return folds


def build_folds_rowwise(df: pd.DataFrame, k: int = 5, seed: int = 42, label_col: str = "Chosen_Molecule"):
    """
    Row-wise stratified K-fold split on the pair rows. This aims for ~N/k rows in each test split.
    Leakage across molecules is allowed by design (user requested).
    """
    ensure_cols(df, label_col)
    # Convert labels to binary 1 for 'A', 0 for 'B'
    y = (df[label_col].astype(str).str.upper() == "A").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    dummy_X = np.zeros((len(df), 1), dtype=np.int8)
    for fidx, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y)):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        folds.append((df_train, df_test, fidx))
    return folds

# ---------- Adapters ----------
@dataclass
class FineTuneAdapter:
    """
    Wraps project finetune(...) and scoring.
    """
    model_path: str
    featurizer: str = "graph_2D"
    batch_size: int = 32
    n_epochs: int = 20
    lr: float = 3e-4
    earlystopping: bool = True
    patience: int = 4
    device: str = "auto"
    track_improvement: bool = False
    wandb_mode: str = "offline"
    num_workers: Optional[int] = None

    # Column names
    label_col: str = "Chosen_Molecule"
    smiles_i_col: str = "Molecule_A"
    smiles_j_col: str = "Molecule_B"

    # If you already have finetuned scores in the CSV set this True.
    use_csv_scores: bool = False
    ft_a_col: str = "smiles_i_score_l"
    ft_b_col: str = "smiles_j_score_l"

    # Optional callback if you need to produce scores from the tuned model
    score_callback: Optional[Callable[[List[str], object], np.ndarray]] = None

    # internal per-fold store
    _fitted_artifacts: dict = None  # e.g., paths, in-memory model, etc.

    def _find_best_checkpoint(self, save_dir: Path) -> Optional[str]:
        ckpt_root = Path(save_dir) / "checkpoints"
        if not ckpt_root.exists():
            return None
        # search all ckpt files
        ckpts = list(ckpt_root.rglob("*.ckpt"))
        if not ckpts:
            return None
        # prefer ones with "-best" in name
        best = [p for p in ckpts if "-best" in p.name]
        candidates = best if best else ckpts
        # pick most recent by mtime
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    def fit(self, train_df: pd.DataFrame, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Build pairwise training targets: 1 if A preferred, 0 if B
        y_train = (train_df[self.label_col].astype(str).str.upper() == "A").astype(int).tolist()
        pairs_train: List[Tuple[str, str]] = list(
            zip(train_df[self.smiles_i_col].astype(str), train_df[self.smiles_j_col].astype(str))
        )

        # Call project finetune(...)
        finetune(
            smiles=pairs_train,
            target=y_train,
            graph_datapath=None,
            filename=None,
            save_dir=str(save_dir),
            featurizer=self.featurizer,
            model_path=self.model_path,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            track_improvement=self.track_improvement,
            earlystopping=self.earlystopping,
            patience=self.patience,
            device=self.device,
            num_workers=self.num_workers,
            wandb_mode=self.wandb_mode,
        )

        ckpt = self._find_best_checkpoint(save_dir)
        self._fitted_artifacts = {
            "save_dir": str(save_dir),
            "ckpt": ckpt,
        }

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # If finetuned scores already live in CSV, just read them
        if self.use_csv_scores:
            ensure_cols(test_df, self.ft_a_col, self.ft_b_col)
            return test_df[self.ft_a_col].to_numpy(), test_df[self.ft_b_col].to_numpy()

        artifacts = self._fitted_artifacts or {}
        ckpt_path = artifacts.get("ckpt")

        # Build default scorer if no callback provided
        def default_score(smiles_list: List[str]) -> np.ndarray:
            if not ckpt_path or not os.path.exists(ckpt_path):
                # Try to re-discover in case not set
                new_ckpt = self._find_best_checkpoint(Path(artifacts.get("save_dir", ".")))
                if new_ckpt:
                    artifacts["ckpt"] = new_ckpt
            model = LitRankNet.load_from_checkpoint(artifacts.get("ckpt"), map_location="cpu")
            scorer = Scorer(model=model, num_workers=0, device="cpu")
            return scorer.score(smiles_list)

        scorer_fn = self.score_callback or default_score

        a_scores = scorer_fn(test_df[self.smiles_i_col].astype(str).tolist())
        b_scores = scorer_fn(test_df[self.smiles_j_col].astype(str).tolist())
        return np.asarray(a_scores), np.asarray(b_scores)


@dataclass
class BaselineAdapter:
    """
    Read baseline (untuned) scores from CSV, e.g., columns `smiles_i_score_og`, `smiles_j_score_og`.
    """
    a_col: str = "smiles_i_score_og"
    b_col: str = "smiles_j_score_og"

    def predict(self, df: pd.DataFrame):
        ensure_cols(df, self.a_col, self.b_col)
        return df[self.a_col].to_numpy(), df[self.b_col].to_numpy()


@dataclass
class BaselineModelAdapter:
    """
    Compute baseline scores using the original (untuned) model checkpoint.
    This is used when --use_csv_baseline is not provided.
    """
    checkpoint: str
    smiles_i_col: str = "Molecule_A"
    smiles_j_col: str = "Molecule_B"
    device: str = "cpu"
    num_workers: int = 0
    _scorer: Optional[Scorer] = None

    def _get_scorer(self) -> Scorer:
        if self._scorer is None:
            # Load original model on CPU by default for safety
            model = LitRankNet.load_from_checkpoint(self.checkpoint, map_location="cpu" if self.device == "cpu" else None)
            self._scorer = Scorer(model=model, num_workers=self.num_workers, device=self.device)
        return self._scorer

    def predict(self, df: pd.DataFrame):
        scorer = self._get_scorer()
        a_scores = scorer.score(df[self.smiles_i_col].astype(str).tolist())
        b_scores = scorer.score(df[self.smiles_j_col].astype(str).tolist())
        return np.asarray(a_scores), np.asarray(b_scores)


# ---------- Main CV ----------
def run_cv(
    df: pd.DataFrame,
    outdir: Path,
    k: int,
    seed: int,
    ft: FineTuneAdapter,
    base: Optional[BaselineAdapter],
    label_col="Chosen_Molecule",
    smiles_i_col="smiles_i",
    smiles_j_col="smiles_j",
):
    outdir.mkdir(parents=True, exist_ok=True)
    folds = build_folds_rowwise(df, k=k, seed=seed, label_col=label_col)

    fold_summaries = []
    acc_ft, auc_ft, acc_base, auc_base = [], [], [], []
    # Accumulate pooled test labels and score deltas for average ROC across folds
    ft_y_all, ft_delta_all = [], []
    base_y_all, base_delta_all = [], []

    for df_train, df_test, fidx in folds:
        fold_dir = outdir / f"fold_{fidx}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 1) Fine-tune on train
        ft.fit(df_train, save_dir=fold_dir / "tuned")

        # 2) Predict on test
        ft_a, ft_b = ft.predict(df_test)
        m_ft = compute_metrics(df_test, label_col, ft_a, ft_b)
        # collect finetuned fold outputs for pooled ROC
        ft_y_all.append(m_ft["y"])
        ft_delta_all.append(m_ft["delta"])

        # 3) Baseline on test (optional)
        m_base, mcn = None, None
        if base is not None:
            base_a, base_b = base.predict(df_test)
            m_base = compute_metrics(df_test, label_col, base_a, base_b)
            mcn = mcnemar_from_preds(m_base["y"], m_base["pred"], m_ft["pred"])
            # collect baseline fold outputs for pooled ROC
            base_y_all.append(m_base["y"])
            base_delta_all.append(m_base["delta"])

            # Plots
            plot_fold_roc(m_base["y"], m_base["delta"], f"Baseline ROC — fold {fidx}", str(fold_dir / "roc_baseline.png"))
        plot_fold_roc(m_ft["y"], m_ft["delta"], f"Finetuned ROC — fold {fidx}", str(fold_dir / "roc_finetuned.png"))

        # Keep per-fold result
        fold_summaries.append({
            "fold": int(fidx),
            "n_test": int(m_ft["n"]),
            "baseline": None if m_base is None else {"acc": m_base["acc"], "ci95_half": m_base["ci95_half"], "auc": m_base["auc"]},
            "finetuned": {"acc": m_ft["acc"], "ci95_half": m_ft["ci95_half"], "auc": m_ft["auc"]},
            "mcnemar": mcn,
        })

        acc_ft.append(m_ft["acc"]);  auc_ft.append(m_ft["auc"])
        if m_base is not None:
            acc_base.append(m_base["acc"]); auc_base.append(m_base["auc"])

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_summaries[-1], f, indent=2)

    def mean_sd(x):
        x = np.asarray(x, float)
        return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)

    ft_acc_mu, ft_acc_sd = mean_sd(acc_ft)
    ft_auc_mu, ft_auc_sd = mean_sd(auc_ft)
    base_acc_mu = base_acc_sd = base_auc_mu = base_auc_sd = None
    if acc_base:
        base_acc_mu, base_acc_sd = mean_sd(acc_base)
        base_auc_mu, base_auc_sd = mean_sd(auc_base)

    summary = {
        "k_folds": k,
        "finetuned": {"acc_mean": ft_acc_mu, "acc_sd": ft_acc_sd, "auc_mean": ft_auc_mu, "auc_sd": ft_auc_sd},
        "baseline": {"acc_mean": base_acc_mu, "acc_sd": base_acc_sd, "auc_mean": base_auc_mu, "auc_sd": base_auc_sd},
        "folds": fold_summaries,
    }
    # Plot average ROC across folds (pooled predictions)
    try:
        y_ft = np.concatenate(ft_y_all) if ft_delta_all else None
        d_ft = np.concatenate(ft_delta_all) if ft_delta_all else None
        y_ba = np.concatenate(base_y_all) if base_delta_all else None
        d_ba = np.concatenate(base_delta_all) if base_delta_all else None
        if (y_ft is not None) or (y_ba is not None):
            plt.figure()
            if (y_ba is not None) and (d_ba is not None):
                fpr_b, tpr_b, _ = roc_curve(y_ba, d_ba)
                auc_b = roc_auc_score(y_ba, d_ba)
                plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC={auc_b:.3f})", color="tab:orange")
            if (y_ft is not None) and (d_ft is not None):
                fpr_f, tpr_f, _ = roc_curve(y_ft, d_ft)
                auc_f = roc_auc_score(y_ft, d_ft)
                plt.plot(fpr_f, tpr_f, label=f"Finetuned (AUC={auc_f:.3f})", color="tab:blue")
            plt.plot([0,1],[0,1],"--", color="gray")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("Average ROC across folds")
            plt.legend()
            plt.savefig(str(outdir / "roc_average.png"), dpi=180, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"[warn] Failed to plot average ROC: {e}")

    with open(outdir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pairs CSV; must include two SMILES columns and a label column with 'A'/'B'.")
    ap.add_argument("--outdir", default="cv_out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_path", required=True, help="Path to base checkpoint for fine-tuning")

    # Column controls (defaults match data/input/l_data.csv)
    ap.add_argument("--smiles_i_col", default="Molecule_A", help="Name of first SMILES column")
    ap.add_argument("--smiles_j_col", default="Molecule_B", help="Name of second SMILES column")
    ap.add_argument("--label_col", default="Chosen_Molecule", help="Name of the label column with 'A'/'B'")

    # Optional precomputed score columns
    ap.add_argument("--use_csv_finetuned", action="store_true", help="If set, read finetuned scores from CSV columns.")
    ap.add_argument("--ft_a_col", default="smiles_i_score_l")
    ap.add_argument("--ft_b_col", default="smiles_j_score_l")
    ap.add_argument("--use_csv_baseline", action="store_true", help="If set, read baseline OG scores from CSV columns.")
    ap.add_argument("--base_a_col", default="smiles_i_score_og")
    ap.add_argument("--base_b_col", default="smiles_j_score_og")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Build adapters
    ft = FineTuneAdapter(
        model_path=args.model_path,
        use_csv_scores=args.use_csv_finetuned,
        ft_a_col=args.ft_a_col,
        ft_b_col=args.ft_b_col,
        label_col=args.label_col,
        smiles_i_col=args.smiles_i_col,
        smiles_j_col=args.smiles_j_col,
    )

    base = (
        BaselineAdapter(a_col=args.base_a_col, b_col=args.base_b_col)
        if args.use_csv_baseline
        else BaselineModelAdapter(
            checkpoint=args.model_path,
            smiles_i_col=args.smiles_i_col,
            smiles_j_col=args.smiles_j_col,
            device="cpu",
            num_workers=0,
        )
    )

    run_cv(
        df=df,
        outdir=Path(args.outdir),
        k=args.folds,
        seed=args.seed,
        ft=ft,
        base=base,
        label_col=args.label_col,
        smiles_i_col=args.smiles_i_col,
        smiles_j_col=args.smiles_j_col,
    )

if __name__ == "__main__":
    main()
