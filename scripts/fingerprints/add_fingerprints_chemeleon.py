"""
add_fingerprints_chemeleon.py

Purpose:
--------
Generates and appends CheMeleon molecular fingerprints (2048-dimensional
learned embeddings) to a scored molecule dataset.

Process:
--------
1. Loads a CSV file containing molecules (with a "smiles" column) and
   their baseline scores.
2. Uses the CheMeleon pretrained model (chemeleon_mp.pt) to compute
   molecular fingerprints:
       fp_0, fp_1, ..., fp_2047
   - Automatically downloads the model from Zenodo if not already present.
3. Concatenates the fingerprint columns to the input dataset.
4. Saves the combined dataset to the specified output file.

Usage:
------
python add_fingerprints_chemeleon.py --input <input_csv_path> --output <output_csv_path>

Defaults:
---------
--input   data/processed/scored.csv
--output  data/processed/scored_with_chemeleon.csv

Example:
--------
python add_fingerprints_chemeleon.py --input data/predict/scored.csv --output data/predict/scored_with_chemeleon.csv

Output:
-------
- CSV file with original data plus 2048 fingerprint columns.
"""



import pandas as pd
import argparse
from pathlib import Path
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))
from tools.chemeleon.chemeleon_fingerprint import CheMeleonFingerprint

# -------- Parse CLI args --------
parser = argparse.ArgumentParser(description="Add CheMeleon fingerprints to input CSV")
parser.add_argument("--input", type=str, default="data/processed/scored.csv", help="Path to scored input CSV")
parser.add_argument("--output", type=str, default="data/processed/scored_with_chemeleon.csv", help="Path to output CSV with fingerprints")
args = parser.parse_args()

INPUT_FILE = Path(args.input).resolve()
OUTPUT_FILE = Path(args.output).resolve()
SMILES_COLUMN = "smiles"

# CheMeleonFingerprint loads weights from ~/.chemprop/chemeleon_mp.pt, downloading
# them from Zenodo on first use. This repo ships those weights at models/chemeleon_mp.pt,
# so seed the cache from the bundled copy — works offline and avoids a network dependency.
import shutil
REPO_DIR = Path(__file__).resolve().parents[2]
_bundled = REPO_DIR / "models" / "chemeleon_mp.pt"
_cache = Path.home() / ".chemprop" / "chemeleon_mp.pt"
if _bundled.exists() and not _cache.exists():
    _cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_bundled, _cache)
    print(f"Seeded CheMeleon weights cache from {_bundled} -> {_cache}")

# -------- Load input --------
print(f"Reading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
smiles_list = df[SMILES_COLUMN].tolist()

# -------- Generate fingerprints --------
encoder = CheMeleonFingerprint()
fps = encoder(smiles_list)

# -------- Append fingerprints --------
fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(fps.shape[1])])
df = pd.concat([df, fp_df], axis=1)

# -------- Save result --------
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved merged output to {OUTPUT_FILE}")
