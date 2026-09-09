"""
create_regression_dataset.py

Description:
------------
Creates a clean dataset for regression tasks by extracting unique molecules
with senior consensus synthesizability difficulty scores.

Process:
1. Extract all unique molecules from pairwise comparisons
2. For each molecule, collect all scores from senior annotators across all pairs
3. Average scores per annotator (handles molecules appearing in multiple pairs)
4. Average across senior annotators to get consensus difficulty score
5. Calculate quality metrics (std, n_appearances, agreement)
6. Output clean CSV: SMILES, consensus_score, quality_metrics

Output is suitable for:
- Regression modeling (predict synthesizability difficulty 1-5)
- Transfer learning / fine-tuning existing models
- Correlation analysis with computational tools
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
INPUT_PATH = Path("data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()

print("="*80)
print("CREATING REGRESSION DATASET: UNIQUE MOLECULES WITH CONSENSUS SCORES")
print("="*80)
print("\nExtracting unique molecules from pairwise comparisons...")

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} pairwise comparisons")

# Define senior annotators
SENIOR_ANNOTATORS = ['asaf', 'elena', 'levi']

# ==============================================================================
# STEP 1: Extract All Molecule Scores
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: EXTRACTING MOLECULE SCORES")
print("="*80)

# Collect all scores for each molecule from each annotator
from collections import defaultdict
molecule_scores = defaultdict(lambda: defaultdict(list))  # {smiles: {annotator: [scores]}}

for idx, row in df.iterrows():
    smiles_a = row['Molecule A SMILES']
    smiles_b = row['Molecule B SMILES']

    for annotator in SENIOR_ANNOTATORS:
        score_a_col = f"{annotator} - Score A"
        score_b_col = f"{annotator} - Score B"

        score_a = row.get(score_a_col)
        score_b = row.get(score_b_col)

        # Process molecule A
        if pd.notna(score_a) and score_a != '':
            molecule_scores[smiles_a][annotator].append(float(score_a))

        # Process molecule B
        if pd.notna(score_b) and score_b != '':
            molecule_scores[smiles_b][annotator].append(float(score_b))

print(f"\nFound {len(molecule_scores)} unique molecules")

# ==============================================================================
# STEP 2: Calculate Per-Molecule Consensus Scores
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: CALCULATING CONSENSUS SCORES")
print("="*80)

regression_data = []

for smiles, annotator_scores in molecule_scores.items():

    # Step 1: Average scores per annotator (for molecules appearing multiple times)
    annotator_means = {}
    annotator_stds = {}

    for annotator in SENIOR_ANNOTATORS:
        scores = annotator_scores.get(annotator, [])
        if len(scores) > 0:
            annotator_means[annotator] = np.mean(scores)
            annotator_stds[annotator] = np.std(scores) if len(scores) > 1 else 0.0

    # Step 2: Average across senior annotators to get consensus
    if len(annotator_means) == 0:
        continue  # No senior ratings

    consensus_score = np.mean(list(annotator_means.values()))
    consensus_std = np.std(list(annotator_means.values()))

    # Calculate additional quality metrics
    n_senior_annotators = len(annotator_means)

    # Total appearances (max across annotators)
    n_appearances = max([len(annotator_scores.get(ann, [])) for ann in SENIOR_ANNOTATORS])

    # Within-annotator variance (how much does same annotator vary for this molecule)
    within_annotator_variance = np.mean([annotator_stds.get(ann, 0) for ann in SENIOR_ANNOTATORS if ann in annotator_stds])

    # Between-annotator variance (how much do annotators disagree)
    between_annotator_variance = consensus_std

    # Determine quality tier based on agreement
    if n_senior_annotators == 3 and consensus_std < 0.5:
        quality = 'high'
    elif n_senior_annotators >= 2 and consensus_std < 1.0:
        quality = 'medium'
    elif n_senior_annotators >= 2 and consensus_std < 1.5:
        quality = 'low'
    else:
        quality = 'very_low'

    regression_data.append({
        'smiles': smiles,
        'consensus_score': consensus_score,
        'consensus_std': consensus_std,
        'n_senior_annotators': n_senior_annotators,
        'n_appearances': n_appearances,
        'within_annotator_variance': within_annotator_variance,
        'between_annotator_variance': between_annotator_variance,
        'quality': quality,
        # Individual annotator means (for transparency)
        'asaf_mean': annotator_means.get('asaf', np.nan),
        'elena_mean': annotator_means.get('elena', np.nan),
        'levi_mean': annotator_means.get('levi', np.nan),
    })

df_regression = pd.DataFrame(regression_data)

print(f"\nCreated regression dataset with {len(df_regression)} molecules")
print(f"Score range: [{df_regression['consensus_score'].min():.2f}, {df_regression['consensus_score'].max():.2f}]")
print(f"Mean consensus score: {df_regression['consensus_score'].mean():.2f} ± {df_regression['consensus_score'].std():.2f}")

# ==============================================================================
# STEP 3: Quality Distribution
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: QUALITY DISTRIBUTION")
print("="*80)

quality_counts = df_regression['quality'].value_counts().sort_index()
print("\nQuality tiers:")
for quality, count in quality_counts.items():
    pct = 100 * count / len(df_regression)
    print(f"  {quality:10s}: {count:4d} molecules ({pct:5.1f}%)")

print("\nAnnotator coverage:")
coverage_counts = df_regression['n_senior_annotators'].value_counts().sort_index()
for n_ann, count in coverage_counts.items():
    pct = 100 * count / len(df_regression)
    print(f"  {int(n_ann)} senior annotators: {count:4d} molecules ({pct:5.1f}%)")

print("\nMolecule appearances:")
appearance_counts = df_regression['n_appearances'].value_counts().sort_index()
for n_app, count in appearance_counts.items():
    pct = 100 * count / len(df_regression)
    print(f"  Appears {int(n_app)} time(s): {count:4d} molecules ({pct:5.1f}%)")

# ==============================================================================
# STEP 4: Statistics by Quality Tier
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: STATISTICS BY QUALITY TIER")
print("="*80)

print("\nMean consensus score by quality:")
for quality in ['high', 'medium', 'low', 'very_low']:
    subset = df_regression[df_regression['quality'] == quality]
    if len(subset) > 0:
        mean_score = subset['consensus_score'].mean()
        std_score = subset['consensus_score'].std()
        mean_std = subset['consensus_std'].mean()
        print(f"  {quality:10s}: score={mean_score:.2f}±{std_score:.2f}, mean_std={mean_std:.2f}, n={len(subset)}")

# ==============================================================================
# STEP 5: Save Datasets
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING DATASETS")
print("="*80)

# Full dataset
full_path = OUTPUT_DIR / "unique_molecules_regression_full.csv"
df_regression.to_csv(full_path, index=False)
print(f"✅ Saved full dataset to {full_path}")
print(f"   {len(df_regression)} molecules")

# High quality only (recommended for training)
high_quality = df_regression[df_regression['quality'].isin(['high', 'medium'])]
hq_path = OUTPUT_DIR / "unique_molecules_regression_high_quality.csv"
high_quality.to_csv(hq_path, index=False)
print(f"✅ Saved high-quality subset to {hq_path}")
print(f"   {len(high_quality)} molecules ({100*len(high_quality)/len(df_regression):.1f}% of total)")

# Simple format (just SMILES and score for easy use)
simple = df_regression[['smiles', 'consensus_score']].copy()
simple_path = OUTPUT_DIR / "unique_molecules_regression_simple.csv"
simple.to_csv(simple_path, index=False)
print(f"✅ Saved simple format to {simple_path}")
print(f"   {len(simple)} molecules (SMILES + score only)")

# High quality simple format
hq_simple = high_quality[['smiles', 'consensus_score']].copy()
hq_simple_path = OUTPUT_DIR / "unique_molecules_regression_hq_simple.csv"
hq_simple.to_csv(hq_simple_path, index=False)
print(f"✅ Saved high-quality simple format to {hq_simple_path}")
print(f"   {len(hq_simple)} molecules")

# ==============================================================================
# STEP 6: Create Summary Report
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: CREATING SUMMARY REPORT")
print("="*80)

report_path = OUTPUT_DIR / "regression_dataset_summary.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("REGRESSION DATASET: UNIQUE MOLECULES WITH CONSENSUS SCORES\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET DESCRIPTION\n")
    f.write("-" * 80 + "\n")
    f.write("This dataset contains unique molecules extracted from pairwise\n")
    f.write("synthesizability comparisons, with consensus difficulty scores\n")
    f.write("from senior annotators (Asaf, Elena, Levi).\n\n")
    f.write("Score scale: 1 = easiest to synthesize, 5 = hardest to synthesize\n\n")

    f.write("PROCESSING PIPELINE\n")
    f.write("-" * 80 + "\n")
    f.write("1. Extract all unique molecules from pairwise data\n")
    f.write("2. For molecules appearing multiple times:\n")
    f.write("   - Average scores per annotator across appearances\n")
    f.write("3. Average across senior annotators for consensus score\n")
    f.write("4. Calculate quality metrics (std, coverage, variance)\n\n")

    f.write("DATASET STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique molecules: {len(df_regression)}\n")
    f.write(f"Score range: [{df_regression['consensus_score'].min():.2f}, {df_regression['consensus_score'].max():.2f}]\n")
    f.write(f"Mean score: {df_regression['consensus_score'].mean():.2f} ± {df_regression['consensus_score'].std():.2f}\n")
    f.write(f"Median score: {df_regression['consensus_score'].median():.2f}\n\n")

    f.write("QUALITY TIERS\n")
    f.write("-" * 80 + "\n")
    for quality, count in quality_counts.items():
        pct = 100 * count / len(df_regression)
        f.write(f"{quality:10s}: {count:4d} molecules ({pct:5.1f}%)\n")
    f.write("\n")

    f.write("Quality tier definitions:\n")
    f.write("  high:     3 senior annotators, std < 0.5\n")
    f.write("  medium:   ≥2 senior annotators, std < 1.0\n")
    f.write("  low:      ≥2 senior annotators, std < 1.5\n")
    f.write("  very_low: Otherwise\n\n")

    f.write("ANNOTATOR COVERAGE\n")
    f.write("-" * 80 + "\n")
    for n_ann, count in coverage_counts.items():
        pct = 100 * count / len(df_regression)
        f.write(f"{int(n_ann)} senior annotators: {count:4d} molecules ({pct:5.1f}%)\n")
    f.write("\n")

    f.write("MOLECULE APPEARANCES (in pairwise data)\n")
    f.write("-" * 80 + "\n")
    for n_app, count in appearance_counts.items():
        pct = 100 * count / len(df_regression)
        f.write(f"Appears {int(n_app)} time(s): {count:4d} molecules ({pct:5.1f}%)\n")
    f.write("\n")

    f.write("RECOMMENDED USAGE\n")
    f.write("-" * 80 + "\n")
    f.write("For regression modeling:\n")
    f.write(f"  Conservative: Use high-quality only ({len(high_quality)} molecules)\n")
    f.write(f"  Balanced:     Use high + medium quality ({len(df_regression[df_regression['quality'].isin(['high', 'medium', 'low'])])} molecules)\n")
    f.write(f"  Liberal:      Use all molecules ({len(df_regression)} molecules)\n\n")

    f.write("FILES GENERATED\n")
    f.write("-" * 80 + "\n")
    f.write("  unique_molecules_regression_full.csv\n")
    f.write("    - All molecules with full metadata\n")
    f.write("  unique_molecules_regression_high_quality.csv\n")
    f.write("    - High + medium quality subset (recommended)\n")
    f.write("  unique_molecules_regression_simple.csv\n")
    f.write("    - SMILES + score only (all molecules)\n")
    f.write("  unique_molecules_regression_hq_simple.csv\n")
    f.write("    - SMILES + score only (high quality subset)\n")

print(f"✅ Saved summary report to {report_path}")

# ==============================================================================
# STEP 7: Create Score Distribution Visualization
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: CREATING VISUALIZATION")
print("="*80)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall score distribution
ax = axes[0, 0]
ax.hist(df_regression['consensus_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Consensus Difficulty Score', fontsize=11)
ax.set_ylabel('Number of Molecules', fontsize=11)
ax.set_title('Distribution of Synthesizability Scores', fontsize=12, fontweight='bold')
ax.axvline(df_regression['consensus_score'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {df_regression["consensus_score"].mean():.2f}')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Score distribution by quality
ax = axes[0, 1]
for quality in ['high', 'medium', 'low', 'very_low']:
    subset = df_regression[df_regression['quality'] == quality]
    if len(subset) > 0:
        ax.hist(subset['consensus_score'], bins=15, alpha=0.5, label=quality, edgecolor='black')
ax.set_xlabel('Consensus Difficulty Score', fontsize=11)
ax.set_ylabel('Number of Molecules', fontsize=11)
ax.set_title('Score Distribution by Quality Tier', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Consensus std distribution
ax = axes[1, 0]
ax.hist(df_regression['consensus_std'], bins=20, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Consensus Std Dev (annotator disagreement)', fontsize=11)
ax.set_ylabel('Number of Molecules', fontsize=11)
ax.set_title('Distribution of Annotator Agreement', fontsize=12, fontweight='bold')
ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='High quality threshold')
ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='Medium quality threshold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Score vs appearances
ax = axes[1, 1]
for n_app in sorted(df_regression['n_appearances'].unique()):
    subset = df_regression[df_regression['n_appearances'] == n_app]
    ax.scatter(subset['consensus_std'], subset['consensus_score'],
               alpha=0.6, s=30, label=f'{int(n_app)} appearance(s)')
ax.set_xlabel('Consensus Std Dev', fontsize=11)
ax.set_ylabel('Consensus Score', fontsize=11)
ax.set_title('Score vs Agreement (by appearances)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
viz_path = OUTPUT_DIR / "regression_dataset_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization to {viz_path}")
plt.close()

print("\n" + "="*80)
print("REGRESSION DATASET CREATION COMPLETE")
print("="*80)
print(f"\n✅ Created dataset with {len(df_regression)} unique molecules")
print(f"✅ Recommended for training: {len(high_quality)} molecules (high + medium quality)")
print(f"\nReady for downstream regression tasks!")
