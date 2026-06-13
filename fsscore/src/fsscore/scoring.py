from pathlib import Path
import argparse
import pandas as pd
from fsscore.score import Scorer
from fsscore.models.ranknet import LitRankNet
from fsscore.utils.paths import PRETRAIN_MODEL_PATH


def load_model_from_checkpoint(checkpoint_path: str):
    cp = Path(checkpoint_path).expanduser()
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")
    # load on CPU by default; Lightning will move params to GPU if available
    model = LitRankNet.load_from_checkpoint(str(cp), map_location="cpu")
    return model


def score_unique(scorer: Scorer, smiles_iter):
    smiles_list = [s for s in smiles_iter if isinstance(s, str) and s.strip()]
    if not smiles_list:
        return {}
    scores = scorer.score(smiles_list)
    return {s: float(sc) for s, sc in zip(smiles_list, scores)}


def main():
    parser = argparse.ArgumentParser(description="Score SMILES using RankNet checkpoint. Supports single mode or CSV pairs mode.")

    # Legacy single scoring mode
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint (.ckpt) file (single mode)")
    parser.add_argument("--smiles", "-s", help="Single SMILES string to score (single mode)")
    parser.add_argument("--smiles-file", "-f", help="File with one SMILES per line (single mode)")

    # CSV pairs mode
    parser.add_argument("--input-csv", help="Path to input CSV with two SMILES columns (pairs mode)")
    parser.add_argument("--output-csv", help="Path to output CSV (if omitted, overwrites input)")
    parser.add_argument("--col1", help="Name of first SMILES column in CSV (pairs mode)")
    parser.add_argument("--col2", help="Name of second SMILES column in CSV (pairs mode)")
    parser.add_argument("--checkpoint-a", help="Path to first checkpoint (.ckpt) file (pairs mode)")
    parser.add_argument("--checkpoint-b", help="Path to second checkpoint (.ckpt) file (pairs mode)")

    args = parser.parse_args()

    # If CSV pairs mode is provided, run it
    if args.input_csv and args.col1 and args.col2 and args.checkpoint_a and args.checkpoint_b:
        # Load models
        model_a = load_model_from_checkpoint(args.checkpoint_a)
        model_b = load_model_from_checkpoint(args.checkpoint_b)

        scorer_a = Scorer(model=model_a, num_workers=0)
        scorer_b = Scorer(model=model_b, num_workers=0)

        # Load CSV
        in_path = Path(args.input_csv)
        if not in_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {in_path}")
        df = pd.read_csv(in_path)
        if args.col1 not in df.columns or args.col2 not in df.columns:
            raise SystemExit(f"Input CSV must contain columns '{args.col1}' and '{args.col2}'. Found: {list(df.columns)}")

        # Deduplicate across both columns
        smiles_all = pd.concat([df[args.col1], df[args.col2]], ignore_index=True).dropna().astype(str).str.strip()
        smiles_unique = smiles_all[smiles_all != ""].drop_duplicates().tolist()

        # Score using both checkpoints
        map_a = score_unique(scorer_a, smiles_unique)
        map_b = score_unique(scorer_b, smiles_unique)

        # Create 4 new columns
        df[f"{args.col1}_score_fsscore"] = df[args.col1].map(map_a)
        df[f"{args.col1}_score_levi"] = df[args.col1].map(map_b)
        df[f"{args.col2}_score_fsscore"] = df[args.col2].map(map_a)
        df[f"{args.col2}_score_levi"] = df[args.col2].map(map_b)

        out_path = Path(args.output_csv) if args.output_csv else in_path
        df.to_csv(out_path, index=False)
        print(f"Wrote scores to: {out_path}")
        return

    # Otherwise, fallback to legacy single scoring mode
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint)
    else:
        # fallback to a bundled PRETRAIN_MODEL_PATH variable if present
        try:
            PRETRAIN_MODEL_PATH  # noqa: F821
        except NameError:
            raise SystemExit("No checkpoint provided and PRETRAIN_MODEL_PATH is not defined. Use --checkpoint")
        model = LitRankNet.load_from_checkpoint(PRETRAIN_MODEL_PATH)

    scorer = Scorer(model=model)

    # collect smiles
    if args.smiles:
        smiles_list = [args.smiles]
    elif args.smiles_file:
        p = Path(args.smiles_file)
        if not p.exists():
            raise FileNotFoundError(f"SMILES file not found: {p}")
        smiles_list = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    else:
        # default example SMILES
        smiles_list = [
            "c1c(c(cc(c1c1c(cc(cc1N(=O)=O)N(=O)=O)N(=O)=O)N(=O)=O)c1c(cc(cc1N(=O)=O)N(=O)=O)N(=O)=O)N(=O)=O"
        ]

    scores = scorer.score(smiles_list)
    print(scores)


if __name__ == "__main__":
    main()
