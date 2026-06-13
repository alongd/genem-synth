import sys
import os
import logging
import warnings
import pandas as pd
from contextlib import redirect_stdout

# Suppress PyTorch Lightning logs and warnings
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'fsscore', 'src')))

from fsscore.score import Scorer
from fsscore.models.ranknet import LitRankNet
from fsscore.utils.paths import PRETRAIN_MODEL_PATH

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_fsscore.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load input data
    df = pd.read_csv(input_csv)
    smiles_list = df['smiles'].tolist()

    # Load pretrained model
    with open(os.devnull, 'w') as f, redirect_stdout(f):  # Quiet model loading logs
        model = LitRankNet.load_from_checkpoint(PRETRAIN_MODEL_PATH)
        scorer = Scorer(model=model, num_workers=0)

    # Compute scores
    try:
        scores = scorer.score(smiles_list)
        scores = [round(s, 4) for s in scores]
    except Exception:
        # Optional: fallback if scoring fails for any reason
        scores = [None] * len(smiles_list)

    # Add / overwrite 'fsscore' column
    df['fsscore (unbounded relative scale, higher = easier to synthesize)'] = scores

    # Write updated dataframe
    df.to_csv(output_csv, index=False)
