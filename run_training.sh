#!/bin/bash
# Train the synthesizability regression pipeline end-to-end.
#
#   Stage 1: Fine-tune a single "production" FSscore model on ALL consensus pairs.
#            This checkpoint is what run_prediction.sh uses to score new molecules.
#   Stage 2: Run the no-leakage (NL) cross-validation pipeline. For each CV fold it
#            fine-tunes FSscore on only the pairs that do not touch that fold's test
#            molecules, scores all molecules, and trains the regression + stacking
#            ensemble. This is the honest evaluation (reported R^2).
#
# Both stages run inside the fsscore-env conda environment.
set -e

echo "=================================================================="
echo "[1/2] Fine-tuning production FSscore on all consensus pairs ..."
echo "=================================================================="
conda run --no-capture-output -n fsscore-env python scripts/scoring/finetune_production_fsscore.py

echo "=================================================================="
echo "[2/2] Running no-leakage CV regression + stacking pipeline ..."
echo "=================================================================="
REG_CV_MODE=nl conda run --no-capture-output -n fsscore-env python scripts/train_models/run_cv_pipeline.py

echo "Done. Trained checkpoints are under models/cv_runs/ and config.json now points to the latest."
