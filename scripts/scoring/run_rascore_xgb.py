import sys
import os
import pandas as pd
from RAscore import RAscore_XGB  # XGB model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_rascore_xgb.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load input CSV
    df = pd.read_csv(input_csv)

    # Model setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    XGB_MODEL_PATH = os.path.join(
        current_dir,
        "..",
        "tools",
        "rascore",
        "models",
        "XGB_chembl_ecfp_counts",
        "model.pkl"
    )

    xgb_scorer = RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)

    # Compute scores for all smiles
    scores = []
    for smiles in df['smiles']:
        try:
            score = xgb_scorer.predict(smiles)
            scores.append(round(score, 4))
        except Exception:
            scores.append(None)  # Optional: sentinel value (e.g. -1) if preferred

    # Add / overwrite 'rascore_xgb' column
    df['rascore_xgb (0–1, higher = easier)'] = scores

    # Write output CSV
    df.to_csv(output_csv, index=False)
