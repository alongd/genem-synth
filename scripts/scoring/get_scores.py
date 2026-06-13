"""
get_scores.py

Description:
------------
This script is the main orchestrator for computing synthesizability scores for molecules in batch.
It reads an input CSV file containing SMILES strings, runs multiple tools (SAScore, SCScore, RAscore-TF, 
RAscore-XGB, FSscore, SYBA), and writes an output CSV with all scores as new columns.

Each tool runs in its own isolated Conda environment using `conda run -n <env>`, ensuring compatibility
between tools that have conflicting dependencies.
"""

import shutil
import subprocess
import pandas as pd
import argparse
from pathlib import Path

# -------- Function: Preprocess reviewer scores --------
def preprocess_input(raw_path, cleaned_path):
    """
    Reads a raw input CSV containing SMILES strings and reviewer score columns,
    computes the highest score per molecule (ignoring missing values),
    and writes a cleaned CSV with only 'smiles' and 'score' columns.
    """
    print(f"🔧 Preprocessing input file: {raw_path}")

    df_raw = pd.read_csv(raw_path)
    reviewer_cols = [col for col in df_raw.columns if col != 'smiles']

    df_raw['target'] = df_raw[reviewer_cols].max(axis=1, skipna=True)
    df_cleaned = df_raw[['smiles', 'target']]

    df_cleaned.to_csv(cleaned_path, index=False)

    print(f"✅ Preprocessing complete. Cleaned file written to: {cleaned_path}")


# -------- Parse CLI args --------
parser = argparse.ArgumentParser(description="Get baseline synthesizability scores")
parser.add_argument(
    "--input",
    type=str,
    default="data/input/training_molecules.csv",
    help="Path to input CSV with SMILES"
)
parser.add_argument(
    "--output",
    type=str,
    default="data/processed/working.csv",
    help="Path to output CSV with synthesizability scores"
)
args = parser.parse_args()

# -------- File Paths --------
RAW_INPUT_FILE = Path(args.input).resolve()
CLEANED_INPUT_FILE = Path("data/input/molecules_cleaned.csv").resolve()
WORKING_FILE = Path(args.output).resolve()
SCRIPTS_DIR = Path("scripts/scoring").resolve()

# -------- Tools configuration --------
# List of (conda_env, script_name) tuples for each tool to run
TOOLS = [
    ("sascore-env", "run_sascore.py"),
    ("scscore-env", "run_scscore.py"),
    ("syba-env", "run_syba.py"),
    ("rascore-env", "run_rascore_tensor.py"),
    ("rascore-env", "run_rascore_xgb.py"),
    ("fsscore-env", "run_fsscore.py")
]

# -------- Preprocess reviewer scores --------
preprocess_input(RAW_INPUT_FILE, CLEANED_INPUT_FILE)

# -------- Prepare working file --------
print("📦 Copying cleaned input file to working file...")
shutil.copy(CLEANED_INPUT_FILE, WORKING_FILE) # Overwrites working.csv each time script runs

# -------- Run each tool --------
for env, script in TOOLS:
    print(f"🔹 Running {script} in {env}...")
    result = subprocess.run([
        "conda", "run", "-n", env,
        "python", f"{SCRIPTS_DIR}/{script}",
        WORKING_FILE, WORKING_FILE
    ])
    if result.returncode != 0:
        print(f"Error running {script} in {env} — exiting early.")
        break
    else:
        print(f"✅ {script} completed successfully.")

# -------- Final output --------
shutil.copy(WORKING_FILE, "data/processed/scored.csv")
print("\n📋 Full contents of scored.csv after all tools:")
df = pd.read_csv(WORKING_FILE)
print(df.to_string(index=False))
