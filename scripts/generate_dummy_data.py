"""
generate_dummy_data.py

Generates a small, internally-consistent set of DUMMY input files so the whole
pipeline can be run end-to-end without any proprietary data. All SMILES are
common, public, valid molecules; the "consensus" synthesizability scores and
pairwise preferences are fabricated (seeded RNG) — they carry no scientific
meaning and exist only to exercise the code paths.

Produces (relative to repo root):
  data/input/dummy_regression.csv       smiles, consensus_score   (regression target, ~1-5)
  data/output/dummy_consensus_pairs.csv  smiles_a, smiles_b, consensus_choice (A/B)
  data/input/dummy_predict.csv           smiles                    (new molecules to score)

Usage:
  python scripts/generate_dummy_data.py
"""

import csv
import random
from pathlib import Path

from rdkit import Chem

REPO_DIR = Path(__file__).resolve().parents[1]
SEED = 42

# A pool of common, valid, drug-like / small public molecules (not our data).
POOL = [
    "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Oc1ccccc1C(=O)O", "OCC(O)CO",
    "c1ccc2ccccc2c1", "CC(C)NCC(O)COc1cccc2ccccc12", "CCOC(=O)c1ccccc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C1CCNCC1", "c1ccncc1", "Cc1ccccc1",
    "CC(N)C(=O)O", "OC(=O)c1ccccc1O", "C1=CC=C(C=C1)O", "CCCCCCCC(=O)O",
    "Clc1ccccc1", "Brc1ccccc1", "Fc1ccccc1", "Nc1ccccc1", "Oc1ccccc1",
    "C1CCCCC1", "C1CCOC1", "C1COCCN1", "c1ccsc1", "c1cc[nH]c1", "c1ccoc1",
    "CC(=O)Nc1ccc(O)cc1", "CCN1CCCC1=O", "COc1ccccc1", "CSc1ccccc1",
    "CC(C)(C)c1ccccc1", "CCCCO", "CCCCN", "CC#N", "C=CC=C", "CC(=O)C",
    "CN(C)C=O", "OCCO", "NCCO", "CC(O)C", "ClCCl", "FC(F)(F)c1ccccc1",
    "c1ccc(cc1)c1ccccc1", "Cc1ccc(cc1)S(=O)(=O)O", "NC(=O)c1ccccc1",
    "CC1=CC(=O)CC(C)(C)C1",
]


def canonical_valid(smiles):
    """Return canonical SMILES if valid, else None."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def main():
    rng = random.Random(SEED)

    # Validate + dedupe the pool
    mols, seen = [], set()
    for s in POOL:
        c = canonical_valid(s)
        if c and c not in seen:
            seen.add(c)
            mols.append(c)
    assert len(mols) >= 40, f"Need >=40 valid molecules, got {len(mols)}"

    # --- regression dataset: smiles, consensus_score (1.0-5.0) ---
    scores = {m: round(rng.uniform(1.0, 5.0), 3) for m in mols}
    reg_path = REPO_DIR / "data" / "input" / "dummy_regression.csv"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "consensus_score"])
        for m in mols:
            w.writerow([m, scores[m]])

    # --- pairwise consensus: smiles_a, smiles_b, consensus_choice (A/B) ---
    # "A" means molecule A was judged easier to synthesize (lower score here),
    # with a little label noise so both classes are well represented.
    pairs = set()
    n_pairs = 90
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        a, b = rng.sample(mols, 2)
        key = (a, b) if a < b else (b, a)
        if key in pairs:
            continue
        pairs.add(key)
    pairs_path = REPO_DIR / "data" / "output" / "dummy_consensus_pairs.csv"
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles_a", "smiles_b", "consensus_choice"])
        for a, b in sorted(pairs):
            easier_is_a = scores[a] <= scores[b]
            if rng.random() < 0.15:  # 15% label noise
                easier_is_a = not easier_is_a
            w.writerow([a, b, "A" if easier_is_a else "B"])

    # --- prediction input: smiles only (subset of pool) ---
    pred_mols = rng.sample(mols, 10)
    pred_path = REPO_DIR / "data" / "input" / "dummy_predict.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for m in pred_mols:
            w.writerow([m])

    print(f"Wrote {reg_path}  ({len(mols)} molecules)")
    print(f"Wrote {pairs_path}  ({len(pairs)} pairs)")
    print(f"Wrote {pred_path}  ({len(pred_mols)} molecules)")


if __name__ == "__main__":
    main()
