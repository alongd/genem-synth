"""
compare_moses_em_distributions.py

Description:
------------
Compares synthesizability score distributions between MOSES drug-like molecules
and energetic materials (EMs) to quantify how differently each tool treats the two domains.

Steps:
1. Sample 10,000 molecules from MOSES training set (seed=42)
2. Score them with all five synthesizability tools
3. Load 416 EM scores from scored.csv
4. Create visualization comparing distributions (KDE/violin plots)
5. Calculate percentile positions of EM scores within MOSES distribution
6. Calculate fraction of EMs incorrectly flagged as hard to synthesize

Output:
- Visualization plots saved to data/output/
- Statistical summary saved to data/output/moses_em_comparison.txt
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os
import subprocess
import shutil
import sys
from scipy import stats

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# File paths
MOSES_TRAIN_PATH = Path("data/input/data_moses_train.csv").resolve()
EM_SCORES_PATH = Path("data/processed/scored.csv").resolve()
MOSES_SAMPLE_PATH = Path("data/processed/moses_sample_10k.csv").resolve()
MOSES_SCORED_PATH = Path("data/processed/moses_sample_10k_scored.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()
SCRIPTS_DIR = Path("scripts/scoring").resolve()

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Score column names and their interpretations
SCORE_COLUMNS = {
    'sascore': {
        'name': 'SA Score',
        'range': '(1-10, lower = easier)',
        'direction': 'lower_better'
    },
    'scscore': {
        'name': 'SCScore',
        'range': '(1-5, lower = easier)',
        'direction': 'lower_better'
    },
    'syba': {
        'name': 'SYBA',
        'range': '(positive = easier)',
        'direction': 'higher_better'
    },
    'rascore_tf': {
        'name': 'RAscore',
        'range': '(0-1, higher = easier)',
        'direction': 'higher_better'
    },
    'fsscore': {
        'name': 'FSscore',
        'range': '(unbounded, higher = easier)',
        'direction': 'higher_better'
    }
}

# Tools configuration for scoring
TOOLS = [
    ("sascore-env", "run_sascore.py"),
    ("scscore-env", "run_scscore.py"),
    ("syba-env", "run_syba.py"),
    ("rascore-env", "run_rascore_tensor.py"),
    ("fsscore-env", "run_fsscore.py")
]

print("="*80)
print("STEP 1: Sample 10,000 MOSES molecules")
print("="*80)

# Load MOSES training set
print(f"Loading MOSES training data from {MOSES_TRAIN_PATH}...")
df_moses = pd.read_csv(MOSES_TRAIN_PATH)
print(f"Total MOSES molecules: {len(df_moses):,}")

# Sample 10,000 molecules with fixed seed
df_moses_sample = df_moses.sample(n=10000, random_state=RANDOM_SEED)
print(f"Sampled {len(df_moses_sample):,} molecules (seed={RANDOM_SEED})")

# Prepare for scoring: create target column (empty for MOSES)
df_moses_sample['target'] = ''
df_moses_sample = df_moses_sample[['SMILES', 'target']]
df_moses_sample.columns = ['smiles', 'target']

# Save sample
df_moses_sample.to_csv(MOSES_SAMPLE_PATH, index=False)
print(f"Saved sample to {MOSES_SAMPLE_PATH}\n")

print("="*80)
print("STEP 2: Score MOSES sample with all 5 tools")
print("="*80)

# Copy sample to working file
shutil.copy(MOSES_SAMPLE_PATH, MOSES_SCORED_PATH)

# Run each scoring tool
CONDA_PATH = os.environ.get("CONDA_EXE", "conda")  # was hardcoded to one machine's miniforge3 path
for env, script in TOOLS:
    print(f"🔹 Running {script} in {env}...")
    result = subprocess.run([
        CONDA_PATH, "run", "-n", env,
        "python", f"{SCRIPTS_DIR}/{script}",
        str(MOSES_SCORED_PATH), str(MOSES_SCORED_PATH)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error running {script} in {env}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ {script} completed successfully.")

print(f"\nMOSES sample scored and saved to {MOSES_SCORED_PATH}\n")

print("="*80)
print("STEP 3: Load scored data")
print("="*80)

# Load MOSES scores
df_moses_scored = pd.read_csv(MOSES_SCORED_PATH)
print(f"Loaded {len(df_moses_scored):,} scored MOSES molecules")

# Load EM scores
df_em = pd.read_csv(EM_SCORES_PATH)
print(f"Loaded {len(df_em):,} scored energetic materials\n")

# Standardize column names
def standardize_column_name(col):
    """Convert column name to standard short form"""
    col_lower = col.lower().strip()
    if 'sascore' in col_lower:
        return 'sascore'
    elif 'scscore' in col_lower:
        return 'scscore'
    elif 'syba' in col_lower:
        return 'syba'
    elif 'rascore_tf' in col_lower or 'rascore (0–1, higher = easier)' in col_lower:
        return 'rascore_tf'
    elif 'fsscore' in col_lower:
        return 'fsscore'
    return col

# Rename columns in both dataframes
df_moses_scored.columns = [standardize_column_name(col) for col in df_moses_scored.columns]
df_em.columns = [standardize_column_name(col) for col in df_em.columns]

print("="*80)
print("STEP 4: Visualize distributions")
print("="*80)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (score_key, score_info) in enumerate(SCORE_COLUMNS.items()):
    ax = axes[idx]

    # Get data for this score
    moses_scores = df_moses_scored[score_key].dropna()
    em_scores = df_em[score_key].dropna()

    # Create violin plots
    data_to_plot = [moses_scores, em_scores]
    positions = [1, 2]

    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)

    # Color MOSES in blue, EM in red
    for pc, color in zip(parts['bodies'], ['#3498db', '#e74c3c']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Set labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(['MOSES\n(n=10,000)', 'Energetic Materials\n(n=416)'])
    ax.set_ylabel('Score')
    ax.set_title(f"{score_info['name']} {score_info['range']}", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# Remove extra subplot
axes[5].axis('off')

plt.tight_layout()
plot_path = OUTPUT_DIR / "moses_em_distributions.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved distribution plot to {plot_path}\n")
plt.close()

# Create KDE plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (score_key, score_info) in enumerate(SCORE_COLUMNS.items()):
    ax = axes[idx]

    # Get data for this score
    moses_scores = df_moses_scored[score_key].dropna()
    em_scores = df_em[score_key].dropna()

    # Plot KDE
    moses_scores.plot.kde(ax=ax, label='MOSES (n=10,000)', color='#3498db', linewidth=2)
    em_scores.plot.kde(ax=ax, label='Energetic Materials (n=416)', color='#e74c3c', linewidth=2)

    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title(f"{score_info['name']} {score_info['range']}", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# Remove extra subplot
axes[5].axis('off')

plt.tight_layout()
plot_path = OUTPUT_DIR / "moses_em_kde.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved KDE plot to {plot_path}\n")
plt.close()

print("="*80)
print("STEP 5: Calculate percentile statistics")
print("="*80)

results = []

for score_key, score_info in SCORE_COLUMNS.items():
    # Get data
    moses_scores = df_moses_scored[score_key].dropna()
    em_scores = df_em[score_key].dropna()

    # Calculate median EM score
    em_median = em_scores.median()

    # Calculate what percentile this falls at in MOSES distribution
    percentile = stats.percentileofscore(moses_scores, em_median)

    # Calculate statistics
    moses_mean = moses_scores.mean()
    moses_std = moses_scores.std()
    moses_median = moses_scores.median()

    em_mean = em_scores.mean()
    em_std = em_scores.std()

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((moses_scores.var() + em_scores.var()) / 2)
    cohens_d = (em_mean - moses_mean) / pooled_std

    results.append({
        'score': score_info['name'],
        'moses_mean': moses_mean,
        'moses_std': moses_std,
        'moses_median': moses_median,
        'em_mean': em_mean,
        'em_std': em_std,
        'em_median': em_median,
        'percentile': percentile,
        'cohens_d': cohens_d
    })

    print(f"\n{score_info['name']}:")
    print(f"  MOSES: μ={moses_mean:.2f}, σ={moses_std:.2f}, median={moses_median:.2f}")
    print(f"  EM:    μ={em_mean:.2f}, σ={em_std:.2f}, median={em_median:.2f}")
    print(f"  → EM median falls at {percentile:.1f}th percentile of MOSES distribution")
    print(f"  → Effect size (Cohen's d): {cohens_d:.2f}")

df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("STEP 6: Calculate misclassification rates (awaiting thresholds)")
print("="*80)
print("\nPlease provide the standard thresholds for each tool to calculate")
print("the fraction of known-synthesizable EMs that are incorrectly flagged as hard.\n")
print("Expected format:")
print("  - SA score: > X is hard")
print("  - SCScore: > X is hard")
print("  - SYBA: < X is hard")
print("  - RAscore: < X is hard")
print("  - FSscore: < X is hard")

# Save summary to file
summary_path = OUTPUT_DIR / "moses_em_comparison.txt"
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MOSES vs Energetic Materials: Synthesizability Score Comparison\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"MOSES molecules sampled: 10,000 (seed={RANDOM_SEED})\n")
    f.write(f"Energetic materials: {len(df_em)}\n\n")

    f.write("STATISTICAL COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION NOTES\n")
    f.write("-" * 80 + "\n")
    f.write("- Percentile: Where the median EM score falls in the MOSES distribution\n")
    f.write("- Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)\n")
    f.write("- Lower percentiles indicate EMs are scored as harder to synthesize\n")
    f.write("  than typical drug-like molecules\n\n")

    f.write("THRESHOLDS FOR MISCLASSIFICATION ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write("Please provide standard thresholds to calculate the fraction of\n")
    f.write("known-synthesizable EMs incorrectly flagged as hard to synthesize.\n")

print(f"\n✅ Summary saved to {summary_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to {OUTPUT_DIR}/")
print("  - moses_em_distributions.png (violin plots)")
print("  - moses_em_kde.png (KDE plots)")
print("  - moses_em_comparison.txt (statistical summary)")
print("\nNext step: Provide thresholds for misclassification analysis")
