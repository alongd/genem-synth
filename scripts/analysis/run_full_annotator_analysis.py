"""
run_full_annotator_analysis.py

Master script that runs the complete annotator agreement analysis pipeline:
1. Inter-annotator agreement metrics (Krippendorff's α, Kendall's W)
2. Pairwise Spearman correlations
3. Consensus dataset generation with quality tiers
4. Structural disagreement analysis

Usage:
    python run_full_annotator_analysis.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path("scripts/analysis").resolve()

scripts = [
    "annotator_agreement_analysis.py",
    "generate_consensus_dataset.py",
    "disagreement_structural_analysis.py",
    "repeated_molecule_consistency.py",
    "create_regression_dataset.py"
]

print("="*80)
print("RUNNING COMPLETE ANNOTATOR AGREEMENT ANALYSIS PIPELINE")
print("="*80)
print()

for i, script in enumerate(scripts, 1):
    script_path = SCRIPTS_DIR / script
    print(f"\n{'='*80}")
    print(f"PART {i}/{len(scripts)}: {script}")
    print(f"{'='*80}\n")

    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"\n❌ Error running {script}")
        sys.exit(1)

print("\n" + "="*80)
print("FULL ANALYSIS PIPELINE COMPLETE")
print("="*80)