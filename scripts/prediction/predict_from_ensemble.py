"""
predict_from_ensemble.py

Purpose:
--------
Generates synthesizability predictions using the trained stacking MLP
ensemble model.

Process:
--------
1. Loads the trained ensemble configuration (`ensemble_config.json`) and
   all base models (both traditional ML and MLP variants).
2. For each base model:
   - Loads its preprocessing scalers (scores, fingerprints, PCA if applicable).
   - Scales input features from `prediction_scored_with_chemeleon.csv`.
   - Runs the model to produce predictions.
   - Inversely scales predictions back to the original target range.
3. Stacks base model predictions into meta-features.
4. Loads the final stacking MLP model and produces ensemble predictions.
5. Saves results (SMILES + predicted score) to:
       data/predictions/predicted_results.csv

Inputs:
-------
- data/processed/prediction_scored_with_chemeleon.csv
  Must contain:
    - 6 baseline synthesizability scores
    - CheMeleon fingerprints (fp_0, ..., fp_2047)

Outputs:
--------
- data/predictions/predicted_results.csv
  Contains:
    - smiles
    - predicted_synth_score

Notes:
------
- Supports base models with or without fingerprints, and with or without PCA.
- Automatically applies correct scaling and transformation for each model type.
"""

import os, json, joblib
import numpy as np
import pandas as pd
import torch

# -------- Paths --------
base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
ensemble_config_path = os.path.abspath(os.path.join(base_dir, "..", "..", "models", "ensemble_model", "ensemble_config.json"))
ensemble_model_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "models", "ensemble_model"))

# -------- Load ensemble config --------
with open(ensemble_config_path) as f:
    ensemble_config = json.load(f)

# -------- Load prediction input --------
prediction_input_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "processed", "prediction_scored_with_chemeleon.csv"))
df = pd.read_csv(prediction_input_path)

X_scores = df[[
    "sascore  (1–10, lower = easier to synthesize)",
    "syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)",
    "scscore (1–5, lower = easier to synthesize)",
    "rascore_xgb (0–1, higher = easier)",
    "rascore_tf (0–1, higher = easier)",
    "fsscore (unbounded relative scale, higher = easier to synthesize)"
]]
X_fp = df[[col for col in df.columns if col.startswith("fp_")]]

# -------- Build meta features --------
meta_features = []
for model_info in ensemble_config["models"]:
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "..", model_info["model_dir"]))
    variant = model_info["variant"]
    model_type = model_info["model_type"]
    use_fp = "fp_True" in variant
    use_pca = "pca_True" in variant

    score_scaler = joblib.load(os.path.join(model_dir, "score_scaler.pkl"))
    y_scaler_model = joblib.load(os.path.join(model_dir, "y_scaler.pkl"))
    X_scores_scaled = score_scaler.transform(X_scores)

    if use_fp:
        fp_scaler = joblib.load(os.path.join(model_dir, "fp_scaler.pkl"))
        X_fp_scaled = fp_scaler.transform(X_fp)
        if use_pca:
            pca = joblib.load(os.path.join(model_dir, "pca.pkl"))
            X_fp_scaled = pca.transform(X_fp_scaled)
        X_input = np.concatenate([X_scores_scaled, X_fp_scaled], axis=1)
    else:
        X_input = X_scores_scaled

    if model_type == "mlp_models":
        with open(os.path.join(model_dir, "best_params.json")) as f:
            params = json.load(f)
        def build_mlp(input_dim, hidden_dim, num_layers, dropout):
            layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            for _ in range(num_layers - 1):
                layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            layers += [torch.nn.Linear(hidden_dim, 1)]
            return torch.nn.Sequential(*layers)

        model = build_mlp(
            input_dim=X_input.shape[1],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            y_pred_scaled = model(X_tensor).numpy().flatten()
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        y_pred_scaled = model.predict(X_input)

    y_pred = y_scaler_model.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    meta_features.append(y_pred)

X_meta = np.stack(meta_features, axis=1)

# -------- Predict using stacking MLP --------
y_scaler_final = joblib.load(os.path.join(ensemble_model_dir, "y_scaler.pkl"))
with open(os.path.join(ensemble_model_dir, "best_params.json")) as f:
    best_params = json.load(f)

layers = []
input_dim = X_meta.shape[1]
for _ in range(best_params["num_layers"]):
    layers.append(torch.nn.Linear(input_dim, best_params["hidden_dim"]))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Dropout(best_params["dropout"]))
    input_dim = best_params["hidden_dim"]
layers.append(torch.nn.Linear(input_dim, 1))
stacking_model = torch.nn.Sequential(*layers)

stacking_model.load_state_dict(torch.load(os.path.join(ensemble_model_dir, "stacking_mlp.pt")))
stacking_model.eval()

with torch.no_grad():
    preds_scaled = stacking_model(torch.tensor(X_meta, dtype=torch.float32)).squeeze().numpy()
    preds = y_scaler_final.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

# -------- Save predictions --------
df["predicted_synth_score"] = preds  # unify column name
df_out = df[["smiles", "predicted_synth_score"]]

output_path = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "predictions", "predicted_results.csv"))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"\nPredictions saved to {output_path}")

# Print model used
print(f"\nLoaded ensemble model: stacking_mlp")

# Print top predictions
print("\nTop 10 Predictions:")
print(df_out.head(10).to_string(index=False))
