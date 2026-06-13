import os
import sys
import pandas as pd

# Setup import path for vendored syba
current_dir = os.path.dirname(os.path.abspath(__file__))
syba_path = os.path.join(current_dir, "..", "tools", "syba")
sys.path.append(syba_path)

from rdkit import Chem
from syba import SybaClassifier

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_syba.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Read input
    df = pd.read_csv(input_csv)

    # Initialize SYBA model once
    syba = SybaClassifier()
    syba.fitDefaultScore()

    scores = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scores.append(None)
        else:
            try:
                score = syba.predict(mol=mol)
                scores.append(round(score, 2))
            except Exception:
                scores.append(None)

    # Add / overwrite 'syba' column
    df['syba (positive = more likely easy to synthesize, negative = more likely hard to synthesize)'] = scores

    # Write output
    df.to_csv(output_csv, index=False)
