"""
predict_from_cv_checkpoint.py

Generates predictions using a checkpoint saved by run_cv_pipeline.py.
Reads checkpoint_dir and output_file from config.json.

Modes (selected by pipeline_predict.py via --single-model flag):
  Default (stacking): checkpoint_dir contains per-model subdirs + stacking_ensemble/.
                      All base models are run and predictions are combined via ridge meta-learner.
  --single-model:     checkpoint_dir IS the model directory (contains config.json + model files).
                      A single model's prediction is written directly without stacking.

Input data must already contain all required score columns (including fsscore_ft if needed)
and CheMeleon fingerprint columns (fp_0 ... fp_N).
"""

import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--single-model", action="store_true",
                    help="Treat checkpoint_dir as a single model directory, skip stacking.")
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parents[2]

with open(REPO_DIR / "config.json") as f:
    config = json.load(f)

checkpoint_dir = Path(config["checkpoint_dir"])
if not checkpoint_dir.is_absolute():
    checkpoint_dir = REPO_DIR / checkpoint_dir

output_file = config["output_file"]

input_path = REPO_DIR / "data" / "processed" / "prediction_scored_with_chemeleon.csv"
df = pd.read_csv(input_path)
fp_cols = [c for c in df.columns if c.startswith("fp_")]


def build_mlp(input_dim, hidden_dim, num_layers, dropout):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden_dim, 1)]
    return nn.Sequential(*layers)


# Discover model dirs: single model or all base models in a cv_run
if args.single_model:
    model_dirs = [checkpoint_dir]
else:
    model_dirs = sorted([
        d for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name != "stacking_ensemble"
    ])

if not model_dirs:
    print(f"ERROR: No model directories found in {checkpoint_dir}")
    sys.exit(1)

meta_features = []

for model_dir in model_dirs:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        continue

    with open(cfg_path) as f:
        model_cfg = json.load(f)

    mt         = model_cfg["model_type"]
    score_cols = model_cfg["score_cols"]
    use_fp     = model_cfg["use_fp"]

    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Skipping {model_dir.name} — missing columns: {missing}")
        continue

    score_scaler = joblib.load(model_dir / "score_scaler.pkl")
    y_scaler     = joblib.load(model_dir / "y_scaler.pkl")
    X_scores_scaled = score_scaler.transform(df[score_cols].values.astype(float))

    if use_fp:
        if not fp_cols:
            print(f"  WARNING: Skipping {model_dir.name} — no fp columns in data")
            continue
        fp_scaler = joblib.load(model_dir / "fp_scaler.pkl")
        X_fp_scaled = fp_scaler.transform(df[fp_cols].values.astype(float))
        if (model_dir / "pca.pkl").exists():
            pca = joblib.load(model_dir / "pca.pkl")
            X_fp_scaled = pca.transform(X_fp_scaled)
        X_input = np.concatenate([X_scores_scaled, X_fp_scaled], axis=1)
    else:
        X_input = X_scores_scaled

    if mt == "mlp":
        with open(model_dir / "best_params.json") as f:
            params = json.load(f)
        model = build_mlp(params["input_dim"], params["hidden_dim"], params["num_layers"], params["dropout"])
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.tensor(X_input, dtype=torch.float32)).squeeze().numpy()
    else:
        model = joblib.load(model_dir / "model.pkl")
        y_pred_scaled = model.predict(X_input)

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    meta_features.append(y_pred)
    print(f"  {model_dir.name}: done")

if not meta_features:
    print("ERROR: No base model predictions could be generated.")
    sys.exit(1)

if args.single_model:
    # Single model: use the one prediction directly
    preds = meta_features[0]
else:
    # Apply ridge meta-learner
    X_meta = np.column_stack(meta_features)

    stack_dir      = checkpoint_dir / "stacking_ensemble"
    ridge_meta     = joblib.load(stack_dir / "ridge_meta.pkl")
    stack_x_scaler = joblib.load(stack_dir / "stack_x_scaler.pkl")
    stack_y_scaler = joblib.load(stack_dir / "stack_y_scaler.pkl")

    X_meta_scaled  = stack_x_scaler.transform(X_meta)
    preds_scaled   = ridge_meta.predict(X_meta_scaled)
    preds          = stack_y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

# Save predictions
output_path = REPO_DIR / output_file
output_path.parent.mkdir(parents=True, exist_ok=True)
df_out = df[["smiles"]].copy()
df_out["predicted_synth_score"] = preds
df_out.to_csv(output_path, index=False)

print(f"\nPredictions saved to {output_path}")
print(f"Checkpoint: {checkpoint_dir}")
print("\nTop 10 Predictions:")
print(df_out.head(10).to_string(index=False))
