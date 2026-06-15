"""
pipeline_predict.py

Runs the full synthesizability prediction pipeline for a new set of molecular
SMILES. Invoked by run_prediction.sh inside fsscore-env.

Stages (each runs as an isolated subprocess for conda-env compatibility):
  1. Baseline synthesizability scores       (scripts/scoring/get_scores.py)
  2. CheMeleon molecular fingerprints        (chemprop-env)
  3. Fine-tuned FSscore column (fsscore_ft)  (fsscore-env, production checkpoint)
  4. Stacking-ensemble prediction            (fsscore-env, predict_from_cv_checkpoint.py)

Reads from config.json:
  - prediction_input_file : raw SMILES CSV (column: smiles)
  - output_file           : final predictions CSV (smiles, predicted_synth_score)
  - checkpoint_dir         : a trained cv_run directory (per-model subdirs + stacking_ensemble/)

The fine-tuned FSscore checkpoint used for scoring is the "production" checkpoint
trained on ALL consensus pairs by scripts/scoring/finetune_production_fsscore.py
(the per-fold no-leakage checkpoints are only for honest CV evaluation, not
deployment). Run ./run_training.sh first to create both the production checkpoint
and the cv_run directory.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
os.chdir(REPO_DIR)

# -------- Load config --------
with open("config.json") as f:
    config = json.load(f)

input_file     = config["prediction_input_file"]
output_file    = config["output_file"]
checkpoint_dir = Path(config.get("checkpoint_dir", "").strip())

scored_file    = "data/processed/prediction_scored.csv"
scored_fp_file = "data/processed/prediction_scored_with_chemeleon.csv"
prod_ckpt      = REPO_DIR / "fsscore" / "models" / "checkpoints_alpha" / "production" / "production_ft.ckpt"

# -------- Guard rails --------
if not str(checkpoint_dir) or not (checkpoint_dir / "run_metadata.json").exists():
    sys.exit(
        f"ERROR: config.json 'checkpoint_dir' ({checkpoint_dir}) is not a trained cv_run "
        f"directory (no run_metadata.json). Run ./run_training.sh first."
    )
if not prod_ckpt.exists():
    sys.exit(
        f"ERROR: production FSscore checkpoint not found at {prod_ckpt}. "
        f"Run ./run_training.sh first (Stage 1 creates it)."
    )

def banner(msg):
    print("=" * 66, flush=True)
    print(msg, flush=True)
    print("=" * 66, flush=True)


# -------- Stage 1: Baseline scores --------
banner("[1/4] Baseline synthesizability scores (get_scores.py) ...")
subprocess.run([
    "python", "scripts/scoring/get_scores.py",
    "--input", input_file,
    "--output", scored_file,
], check=True)

# -------- Stage 2: CheMeleon fingerprints (chemprop-env) --------
banner("[2/4] CheMeleon fingerprints (chemprop-env) ...")
subprocess.run([
    "conda", "run", "--no-capture-output", "-n", "chemprop-env",
    "python", "scripts/fingerprints/add_fingerprints_chemeleon.py",
    "--input", scored_file,
    "--output", scored_fp_file,
], check=True)

# -------- Stage 3: Fine-tuned FSscore column (fsscore-env) --------
banner("[3/4] Fine-tuned FSscore (production checkpoint) ...")
subprocess.run([
    "conda", "run", "--no-capture-output", "-n", "fsscore-env",
    "python", "scripts/scoring/run_fsscore_ft.py",
    "--input", scored_fp_file,
    "--output", scored_fp_file,
    "--checkpoint", str(prod_ckpt),
], check=True)

# -------- Stage 4: Stacking-ensemble prediction (fsscore-env) --------
banner("[4/4] Stacking-ensemble prediction (predict_from_cv_checkpoint.py) ...")
subprocess.run([
    "conda", "run", "--no-capture-output", "-n", "fsscore-env",
    "python", "scripts/prediction/predict_from_cv_checkpoint.py",
], check=True)

print(f"\n✅ Prediction pipeline completed → {output_file}", flush=True)
