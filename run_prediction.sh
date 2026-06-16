#!/bin/bash
# Predict synthesizability for new molecules listed in config.json -> prediction_input_file.
# Requires that ./run_training.sh has been run first (it sets checkpoint_dir in config.json).
# Each stage runs in its own conda environment internally; the driver runs in fsscore-env.
set -e

echo "Running the prediction pipeline ..."
conda run --no-capture-output -n fsscore-env python scripts/pipeline_predict.py

echo "Done! Predictions written to the output_file in config.json."
