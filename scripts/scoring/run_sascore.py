import os
import sys
import pandas as pd
from rdkit import Chem

# Setup sascore import path
current_dir = os.path.dirname(os.path.abspath(__file__))
sascore_path = os.path.join(current_dir, "..", "tools", "sascore")
sys.path.append(sascore_path)
import sascorer

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python run_sascore.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Read input CSV
    df = pd.read_csv(input_csv)

    scores = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            score = None  # Optional: record missing if SMILES invalid
        else:
            score = round(sascorer.calculateScore(mol), 2)
        scores.append(score)

    # Add / overwrite 'sascore' column
    df['sascore  (1–10, lower = easier to synthesize)'] = scores

    # Write updated dataframe
    df.to_csv(output_csv, index=False)
