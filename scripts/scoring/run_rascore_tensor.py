import sys
import os
import pandas as pd
from RAscore import RAscore_NN  # TensorFlow / Keras model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_rascore_tensor.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load input CSV
    df = pd.read_csv(input_csv)

    # Model setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    NN_MODEL_PATH = os.path.join(
        current_dir,
        "..",
        "tools",
        "rascore",
        "models",
        "DNN_chembl_fcfp_counts",
        "model.tf"
    )

    nn_scorer = RAscore_NN.RAScorerNN(model_path=NN_MODEL_PATH)

    # Compute scores for all smiles
    scores = []
    for smiles in df['smiles']:
        try:
            score = nn_scorer.predict(smiles)
            scores.append(round(score, 4))
        except Exception:
            scores.append(None)  # Optional: handle invalid SMILES or prediction failure

    # Add / overwrite 'rascore_tf' column
    df['rascore_tf (0–1, higher = easier)'] = scores

    # Write output CSV
    df.to_csv(output_csv, index=False)
