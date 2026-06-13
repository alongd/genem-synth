import sys
import os
import contextlib
import pandas as pd

# Setup SCScore import path
current_dir = os.path.dirname(os.path.abspath(__file__))
scscore_path = os.path.join(current_dir, "..", "tools", "scscore", "scscore")
sys.path.append(scscore_path)

from standalone_model_numpy import SCScorer

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python run_scscore.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Initialize model
    model = SCScorer()
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        model.restore(
            os.path.join(
                current_dir,
                "..",
                "tools",
                "scscore",
                "models",
                "full_reaxys_model_1024bool",
                "model.ckpt-10654.as_numpy.json.gz"
            )
        )

    # Read input CSV
    df = pd.read_csv(input_csv)

    scores = []
    for smiles in df['smiles']:
        try:
            original_smi, score = model.get_score_from_smi(smiles)
            scores.append(round(score, 2))
        except Exception:
            scores.append(None)  # Or use a sentinel value (e.g. -1)

    # Add / overwrite 'scscore' column
    df['scscore (1–5, lower = easier to synthesize)'] = scores

    # Write output CSV
    df.to_csv(output_csv, index=False)
