"""
finetune_production_fsscore.py

Fine-tunes a single "production" FSscore model on ALL consensus pairs and saves
one deployable checkpoint. run_prediction.sh uses this checkpoint to score new
molecules (the per-fold checkpoints produced by the no-leakage CV are only for
honest evaluation, not deployment).

Reads the pairwise file from config.json -> consensus_pairs_file
(columns: smiles_a, smiles_b, consensus_choice with values "A"/"B").

Output:
  fsscore/models/checkpoints_alpha/production/production_ft.ckpt

Run inside fsscore-env:
  conda run -n fsscore-env python scripts/scoring/finetune_production_fsscore.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
FSSCORE_DIR = REPO_DIR / "fsscore"
BASE_FSSCORE_CKPT = FSSCORE_DIR / "models" / "pretrain_graph_GGLGGL_ep242_best_valloss.ckpt"
PROD_DIR = FSSCORE_DIR / "models" / "checkpoints_alpha" / "production"

sys.path.insert(0, str(FSSCORE_DIR / "src"))
import torch  # noqa: E402
from fsscore.finetuning import finetune  # noqa: E402

torch.set_float32_matmul_precision("medium")


def main():
    with open(REPO_DIR / "config.json") as f:
        config = json.load(f)
    pairs_csv = REPO_DIR / config.get(
        "consensus_pairs_file", "data/output/dummy_consensus_pairs.csv"
    )

    pairs = pd.read_csv(pairs_csv)
    smiles = list(zip(pairs["smiles_a"], pairs["smiles_b"]))
    target = (pairs["consensus_choice"] == "A").astype(int).tolist()
    print(f"Loaded {len(smiles)} pairs from {pairs_csv}")

    PROD_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_dir = tmp / "tuned"
        save_dir.mkdir(parents=True, exist_ok=True)
        graph_datapath = tmp / "prod_ft.pt"

        finetune(
            smiles=smiles,
            target=target,
            graph_datapath=str(graph_datapath),
            filename="production",
            save_dir=str(save_dir),
            model_path=str(BASE_FSSCORE_CKPT),
            featurizer="graph_2D",
            n_epochs=20,
            val_size=0.0,
            earlystopping=False,
            lr=1e-4,
            batch_size=32,
            num_workers=0,
            device="cpu",
            wandb_mode="disabled",
        )

        ckpts = sorted(save_dir.rglob("*best*.ckpt")) or sorted(save_dir.rglob("*.ckpt"))
        if not ckpts:
            raise RuntimeError(f"No checkpoint produced under {save_dir}")
        produced = ckpts[-1]
        dest = PROD_DIR / "production_ft.ckpt"
        shutil.copyfile(produced, dest)
        print(f"Production FSscore checkpoint saved -> {dest}")


if __name__ == "__main__":
    main()
