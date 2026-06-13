"""
predict_from_model.py

Purpose:
--------
Generates synthesizability predictions using a single trained model
(traditional ML, XGB or MLP) based on the configuration in `config.json`.

Process:
--------
1. Loads configuration from `config.json`:
     - model_type (e.g., "rf_models", "mlp_models")
     - variant (e.g., "fp_True_pca_False")
     - output_file (path for predictions)
2. Loads preprocessed prediction input file:
       data/processed/prediction_scored_with_chemeleon.csv
3. Selects and scales the baseline synthesizability scores.
4. If specified in variant:
     - Loads and scales CheMeleon fingerprints.
     - Optionally applies PCA transformation.
5. Loads the trained model:
     - MLP models: build architecture from `best_params.json` and load weights.
     - Traditional ML models: load `.pkl` model file.
6. Produces predictions and saves them to the configured output path.

Inputs:
-------
- config.json
  Must define:
     model_type   : str  (e.g., "rf_models")
     variant      : str  (e.g., "fp_True_pca_False")
     output_file  : str  (path to save predictions)
- data/processed/prediction_scored_with_chemeleon.csv
  Must contain:
     - 6 baseline score columns
     - CheMeleon fingerprints (fp_0, ..., fp_2047) if variant uses fp=True

Outputs:
--------
- User-defined output_file path containing:
     smiles
     predicted_synth_score

Notes:
------
- The model type and variant determine feature selection and preprocessing.
- Supports both models with/without fingerprints and with/without PCA.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Current working directory:", os.getcwd())

# -------- Load config --------
with open("config.json", "r") as f:
    config = json.load(f)

model_type = config["model_type"]         # e.g., "rf_models"
variant = config["variant"]               # e.g., "fp_True_pca_False"
output_file = config["output_file"]       # e.g., "data/predictions/predicted_results.csv"

# -------- Determine input file path --------
scored_fp_file = "data/processed/prediction_scored_with_chemeleon.csv"

# -------- Load data --------
df = pd.read_csv(scored_fp_file)
X = None

# -------- Select features --------
score_cols = [
    "sascore  (1–10, lower = easier to synthesize)", 
    "syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)", 
    "scscore (1–5, lower = easier to synthesize)", 
    "rascore_xgb (0–1, higher = easier)", 
    "rascore_tf (0–1, higher = easier)", 
    "fsscore (unbounded relative scale, higher = easier to synthesize)"
]

use_fp = "fp_True" in variant
use_pca = "pca_True" in variant

# Load and scale score columns
X_scores = df[score_cols].values
score_scaler_path = f"models/{model_type}/{variant}/score_scaler.pkl"
score_scaler = joblib.load(score_scaler_path)
X_scores = score_scaler.transform(X_scores)

if use_fp:
    # Load raw fingerprint features
    fp_cols = [col for col in df.columns if col.startswith("fp_") and not col.startswith("fp_pca_")]
    X_fp = df[fp_cols].values

    # Scale the fingerprints
    fp_scaler_path = f"models/{model_type}/{variant}/fp_scaler.pkl"
    fp_scaler = joblib.load(fp_scaler_path)
    X_fp = fp_scaler.transform(X_fp)

    # Apply PCA if needed
    if use_pca:
        pca_path = f"models/{model_type}/{variant}/pca.pkl"
        pca = joblib.load(pca_path)
        X_fp = pca.transform(X_fp)

    # Concatenate scores and fingerprints
    X = np.concatenate([X_scores, X_fp], axis=1)
else:
    X = X_scores

# -------- Load model --------
model_dir = f"models/{model_type}/{variant}"

if model_type == "mlp_models":
    # Load MLP hyperparameters and architecture
    with open(os.path.join(model_dir, "best_params.json")) as f:
        params = json.load(f)
    
    input_dim = X.shape[1]
    
    # Define MLP architecture
    def build_mlp(input_dim, hidden_dim, num_layers, dropout):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, 1)]
        return nn.Sequential(*layers)

    # Build and load model
    model = build_mlp(input_dim, params["hidden_dim"], params["num_layers"], params["dropout"])
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(X_tensor).numpy().flatten()

else:
    # Load traditional model (RF, XGB, LR)
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    preds = model.predict(X)

# -------- Save predictions --------
df["predicted_synth_score"] = preds
df[["smiles", "predicted_synth_score"]].to_csv(output_file, index=False)

print(f"\nPredictions saved to {output_file}")

# Print model used
print(f"\nLoaded model from: {model_type}")
print(f"Using model configuration: {variant}")

# Print top predictions
print("\nTop 10 Predictions:")
print(df[["smiles", "predicted_synth_score"]].head(10).to_string(index=False))
