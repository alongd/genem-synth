"""
run_fsscore_ft.py

Adds fine-tuned FSScore predictions (column: fsscore_ft) to an input CSV.
Must be run in fsscore-env.

Usage:
    conda run -n fsscore-env python scripts/scoring/run_fsscore_ft.py \
        --input  data/processed/prediction_scored_with_chemeleon.csv \
        --output data/processed/prediction_scored_with_chemeleon.csv \
        --checkpoint fsscore/models/checkpoints_alpha/run_1/ft_epoch=5-best_val_loss.ckpt
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "fsscore" / "src"))

import torch
from fsscore.models.ranknet import LitRankNet
from fsscore.score import Scorer

torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--input",      required=True, help="Input CSV with smiles column")
parser.add_argument("--output",     required=True, help="Output CSV (can be same as input)")
parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned FSScore .ckpt file")
args = parser.parse_args()

df = pd.read_csv(args.input)
smiles_list = df["smiles"].tolist()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    model = LitRankNet.load_from_checkpoint(args.checkpoint, map_location="cpu")
    scorer = Scorer(
        model=model, num_workers=0, device="cpu", batch_size=128,
        graph_datapath=str(tmpdir / "ft_graphs.pt"), keep_graphs=True,
    )
    scores = scorer.score(smiles_list)

df["fsscore_ft"] = [float(s) for s in scores]
df.to_csv(args.output, index=False)
print(f"fsscore_ft scores added ({len(df)} molecules) → {args.output}")
