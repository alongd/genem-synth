"""
repeated_molecule_consistency.py

Description:
------------
Analyzes annotator consistency for molecules that appear in multiple pairs.

Key question: Do annotators give consistent scores to the same molecule across
different pairwise comparisons, or are their scores influenced by what they're
being compared against?

Analyses:
1. Identify molecules appearing in multiple pairs
2. Calculate score variance per molecule per annotator
3. Identify molecules with high score variance (context-dependent scoring)
4. Compare variance across annotators (who is most/least consistent)
5. Analyze if pair context (difficulty of comparison molecule) affects scoring
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# File paths
INPUT_PATH = Path("data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()

print("="*80)
print("REPEATED MOLECULE CONSISTENCY ANALYSIS")
print("="*80)
print("\nAnalyzing whether annotators score the same molecule consistently")
print("across different pairwise comparisons.\n")

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} pairwise comparisons")

# Define annotators
SENIOR_ANNOTATORS = ['asaf', 'elena', 'levi']
JUNIOR_ANNOTATORS = ['roman']
ALL_ANNOTATORS = SENIOR_ANNOTATORS + JUNIOR_ANNOTATORS

# ==============================================================================
# STEP 1: Identify Repeated Molecules
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: IDENTIFYING REPEATED MOLECULES")
print("="*80)

# Collect all molecule scores across all pairs
molecule_scores = defaultdict(lambda: defaultdict(list))  # {smiles: {annotator: [scores]}}
molecule_contexts = defaultdict(list)  # {smiles: [(pair_id, paired_with_smiles, score)]}

for idx, row in df.iterrows():
    pair_id = row['Pair ID']
    smiles_a = row['Molecule A SMILES']
    smiles_b = row['Molecule B SMILES']

    for annotator in ALL_ANNOTATORS:
        score_a_col = f"{annotator} - Score A"
        score_b_col = f"{annotator} - Score B"

        # Handle roman's different column name
        if annotator == 'roman':
            score_b_col = "roman- Score B"

        score_a = row.get(score_a_col)
        score_b = row.get(score_b_col)

        # Process molecule A
        if pd.notna(score_a) and score_a != '':
            score_a = float(score_a)
            molecule_scores[smiles_a][annotator].append(score_a)

            # Store context (what it was paired with)
            if annotator == 'asaf':  # Only store once per pair
                score_b_val = float(score_b) if pd.notna(score_b) and score_b != '' else np.nan
                molecule_contexts[smiles_a].append({
                    'pair_id': pair_id,
                    'paired_with': smiles_b,
                    'paired_score': score_b_val,
                    'position': 'A'
                })

        # Process molecule B
        if pd.notna(score_b) and score_b != '':
            score_b = float(score_b)
            molecule_scores[smiles_b][annotator].append(score_b)

            # Store context
            if annotator == 'asaf':
                score_a_val = float(score_a) if pd.notna(score_a) and score_a != '' else np.nan
                molecule_contexts[smiles_b].append({
                    'pair_id': pair_id,
                    'paired_with': smiles_a,
                    'paired_score': score_a_val,
                    'position': 'B'
                })

# Find repeated molecules
repeated_molecules = {}
for smiles, annotator_scores in molecule_scores.items():
    # Count how many times this molecule appears (use asaf as reference)
    if 'asaf' in annotator_scores:
        n_appearances = len(annotator_scores['asaf'])
        if n_appearances > 1:
            repeated_molecules[smiles] = n_appearances

print(f"\nTotal unique molecules: {len(molecule_scores)}")
print(f"Molecules appearing >1 time: {len(repeated_molecules)}")

# Distribution of repetitions
repetition_counts = defaultdict(int)
for n in repeated_molecules.values():
    repetition_counts[n] += 1

print("\nRepetition distribution:")
for n in sorted(repetition_counts.keys()):
    count = repetition_counts[n]
    print(f"  Appears {n} times: {count} molecules")

# ==============================================================================
# STEP 2: Calculate Score Variance Per Molecule
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: CALCULATING SCORE VARIANCE FOR REPEATED MOLECULES")
print("="*80)

variance_results = []

for smiles in repeated_molecules.keys():
    contexts = molecule_contexts[smiles]

    for annotator in ALL_ANNOTATORS:
        scores = molecule_scores[smiles].get(annotator, [])

        if len(scores) < 2:
            continue

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        score_range = max_score - min_score

        variance_results.append({
            'smiles': smiles,
            'annotator': annotator,
            'n_appearances': len(scores),
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'score_range': score_range,
            'all_scores': scores
        })

df_variance = pd.DataFrame(variance_results)

print(f"\nAnalyzed {len(df_variance)} molecule-annotator combinations")
print(f"Average std across all repeated molecules: {df_variance['std_score'].mean():.3f}")
print(f"Median std: {df_variance['std_score'].median():.3f}")

# Top 10 most inconsistent molecules (highest variance)
print("\nTop 10 molecules with highest score variance (any annotator):")
print("-" * 80)
top_inconsistent = df_variance.nlargest(10, 'std_score')
for idx, row in top_inconsistent.iterrows():
    print(f"{row['annotator']:6s}: std={row['std_score']:.2f}, range=[{row['min_score']:.0f}-{row['max_score']:.0f}], " +
          f"n={row['n_appearances']}, SMILES={row['smiles'][:50]}...")

# ==============================================================================
# STEP 3: Annotator-Level Consistency Comparison
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: ANNOTATOR-LEVEL CONSISTENCY")
print("="*80)

# Calculate average variance per annotator
annotator_consistency = []

for annotator in ALL_ANNOTATORS:
    ann_data = df_variance[df_variance['annotator'] == annotator]

    if len(ann_data) == 0:
        continue

    annotator_consistency.append({
        'annotator': annotator,
        'n_molecules': len(ann_data),
        'mean_std': ann_data['std_score'].mean(),
        'median_std': ann_data['std_score'].median(),
        'mean_range': ann_data['score_range'].mean(),
        'pct_std_gt_1': 100 * (ann_data['std_score'] > 1.0).sum() / len(ann_data),
        'pct_std_gt_15': 100 * (ann_data['std_score'] > 1.5).sum() / len(ann_data),
        'max_std': ann_data['std_score'].max()
    })

df_consistency = pd.DataFrame(annotator_consistency)

print("\nAnnotator Consistency Statistics:")
print("=" * 80)
print("(Lower values = more consistent scoring across different pairs)")
print("-" * 80)
print(df_consistency.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# ==============================================================================
# STEP 4: Context Effect Analysis
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: CONTEXT EFFECT ANALYSIS")
print("="*80)
print("\nDoes the difficulty of the paired molecule affect scoring?")

context_effects = []

for smiles in repeated_molecules.keys():
    contexts = molecule_contexts[smiles]

    if len(contexts) < 2:
        continue

    for annotator in ALL_ANNOTATORS:
        scores = molecule_scores[smiles].get(annotator, [])

        if len(scores) < 2:
            continue

        # Get paired molecule scores (context difficulty)
        paired_scores = [ctx['paired_score'] for ctx in contexts]

        # Remove NaN pairs
        valid_pairs = [(s, ps) for s, ps in zip(scores, paired_scores) if not np.isnan(ps)]

        if len(valid_pairs) < 2:
            continue

        own_scores = [p[0] for p in valid_pairs]
        context_scores = [p[1] for p in valid_pairs]

        # Calculate correlation between own score and paired molecule difficulty
        if len(own_scores) > 2 and np.std(own_scores) > 0 and np.std(context_scores) > 0:
            correlation = np.corrcoef(own_scores, context_scores)[0, 1]

            context_effects.append({
                'smiles': smiles,
                'annotator': annotator,
                'n_pairs': len(own_scores),
                'own_score_std': np.std(own_scores),
                'context_score_mean': np.mean(context_scores),
                'correlation': correlation
            })

df_context = pd.DataFrame(context_effects)

if len(df_context) > 0:
    print(f"\nAnalyzed context effects for {len(df_context)} cases")
    print(f"Mean correlation (own score vs paired difficulty): {df_context['correlation'].mean():.3f}")
    print(f"Median correlation: {df_context['correlation'].median():.3f}")

    # Positive correlation = scoring relative to pair (harder when paired with easy molecule)
    # Negative correlation = inverse relationship
    # Near zero = independent scoring

    print("\nContext effect by annotator:")
    print("-" * 80)
    for annotator in ALL_ANNOTATORS:
        ann_ctx = df_context[df_context['annotator'] == annotator]
        if len(ann_ctx) > 0:
            mean_corr = ann_ctx['correlation'].mean()
            pct_positive = 100 * (ann_ctx['correlation'] > 0.3).sum() / len(ann_ctx)
            print(f"  {annotator:6s}: mean_corr={mean_corr:6.3f}, {pct_positive:.1f}% show positive context effect")

# ==============================================================================
# STEP 5: Visualizations
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Annotator consistency comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean std by annotator
ax = axes[0]
annotators = df_consistency['annotator']
mean_stds = df_consistency['mean_std']
colors = ['lightblue' if ann in SENIOR_ANNOTATORS else 'lightcoral'
          for ann in df_consistency['annotator']]

bars = ax.bar(range(len(annotators)), mean_stds, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(annotators)))
ax.set_xticklabels([a.capitalize() for a in annotators])
ax.set_ylabel('Mean Score Std Dev\n(for repeated molecules)', fontsize=11)
ax.set_title('Annotator Consistency\n(Lower = More Consistent)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, mean_stds)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontweight='bold')

# Percentage with high variance
ax = axes[1]
pct_high = df_consistency['pct_std_gt_1']
bars = ax.bar(range(len(annotators)), pct_high, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(range(len(annotators)))
ax.set_xticklabels([a.capitalize() for a in annotators])
ax.set_ylabel('% of Molecules with Std > 1.0', fontsize=11)
ax.set_title('High Variance Molecules\n(Inconsistent Scoring)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, pct_high)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
consistency_plot_path = OUTPUT_DIR / "annotator_consistency_repeated_molecules.png"
plt.savefig(consistency_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved consistency plot to {consistency_plot_path}")
plt.close()

# Plot 2: Distribution of score variance
fig, ax = plt.subplots(figsize=(10, 6))

for annotator in ALL_ANNOTATORS:
    ann_data = df_variance[df_variance['annotator'] == annotator]
    if len(ann_data) > 0:
        ax.hist(ann_data['std_score'], bins=20, alpha=0.5, label=annotator.capitalize(),
                edgecolor='black')

ax.set_xlabel('Score Standard Deviation', fontsize=12)
ax.set_ylabel('Number of Molecules', fontsize=12)
ax.set_title('Distribution of Score Variance for Repeated Molecules', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Std = 1.0', alpha=0.7)

plt.tight_layout()
dist_plot_path = OUTPUT_DIR / "score_variance_distribution.png"
plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved variance distribution to {dist_plot_path}")
plt.close()

# Plot 3: Context effect (if available)
if len(df_context) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    for annotator in ALL_ANNOTATORS:
        ann_ctx = df_context[df_context['annotator'] == annotator]
        if len(ann_ctx) > 0:
            ax.scatter(ann_ctx['context_score_mean'], ann_ctx['correlation'],
                      alpha=0.6, s=50, label=annotator.capitalize())

    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Moderate positive effect')
    ax.axhline(-0.3, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Moderate negative effect')

    ax.set_xlabel('Mean Difficulty of Paired Molecules', fontsize=12)
    ax.set_ylabel('Correlation (Own Score vs Paired Difficulty)', fontsize=12)
    ax.set_title('Context Effect: Does Pair Difficulty Affect Scoring?', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    context_plot_path = OUTPUT_DIR / "context_effect_analysis.png"
    plt.savefig(context_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved context effect plot to {context_plot_path}")
    plt.close()

# ==============================================================================
# STEP 6: Save Results
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING RESULTS")
print("="*80)

# Save variance data
variance_path = OUTPUT_DIR / "repeated_molecule_variance.csv"
df_variance.to_csv(variance_path, index=False)
print(f"✅ Saved variance data to {variance_path}")

# Save consistency summary
consistency_path = OUTPUT_DIR / "annotator_consistency_summary.csv"
df_consistency.to_csv(consistency_path, index=False)
print(f"✅ Saved consistency summary to {consistency_path}")

# Save context effects
if len(df_context) > 0:
    context_path = OUTPUT_DIR / "context_effect_analysis.csv"
    df_context.to_csv(context_path, index=False)
    print(f"✅ Saved context effects to {context_path}")

# Save detailed report
report_path = OUTPUT_DIR / "repeated_molecule_consistency_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("REPEATED MOLECULE CONSISTENCY ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("QUESTION\n")
    f.write("-" * 80 + "\n")
    f.write("Do annotators give consistent scores to the same molecule across\n")
    f.write("different pairwise comparisons, or are their scores influenced by\n")
    f.write("what they're being compared against?\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total unique molecules: {len(molecule_scores)}\n")
    f.write(f"Molecules appearing >1 time: {len(repeated_molecules)}\n\n")

    f.write("Repetition distribution:\n")
    for n in sorted(repetition_counts.keys()):
        count = repetition_counts[n]
        f.write(f"  Appears {n} times: {count} molecules\n")

    f.write("\n" + "ANNOTATOR CONSISTENCY\n")
    f.write("-" * 80 + "\n")
    f.write("(Lower values = more consistent scoring across different pairs)\n\n")
    f.write(df_consistency.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 80 + "\n")
    f.write("mean_std: Average score std dev across all repeated molecules\n")
    f.write("pct_std_gt_1: Percentage of molecules with std > 1.0 (high inconsistency)\n")
    f.write("pct_std_gt_15: Percentage with std > 1.5 (very high inconsistency)\n\n")

    if len(df_context) > 0:
        f.write("CONTEXT EFFECT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Correlation between own score and paired molecule difficulty:\n")
        f.write("  Positive (+): Score increases when paired with easier molecules (relative scoring)\n")
        f.write("  Negative (-): Score decreases when paired with easier molecules\n")
        f.write("  Near zero:    Scoring is independent of pair context\n\n")

        for annotator in ALL_ANNOTATORS:
            ann_ctx = df_context[df_context['annotator'] == annotator]
            if len(ann_ctx) > 0:
                mean_corr = ann_ctx['correlation'].mean()
                pct_positive = 100 * (ann_ctx['correlation'] > 0.3).sum() / len(ann_ctx)
                f.write(f"  {annotator:6s}: mean_corr={mean_corr:6.3f}, {pct_positive:.1f}% show positive context effect\n")

    f.write("\n" + "TOP 10 MOST INCONSISTENT MOLECULES\n")
    f.write("-" * 80 + "\n")
    for idx, row in top_inconsistent.iterrows():
        f.write(f"\n{row['annotator']:6s}: std={row['std_score']:.2f}, range=[{row['min_score']:.0f}-{row['max_score']:.0f}]\n")
        f.write(f"  SMILES: {row['smiles']}\n")
        f.write(f"  Scores: {row['all_scores']}\n")

print(f"✅ Saved detailed report to {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Finding:")
if len(df_consistency) > 0:
    best_annotator = df_consistency.loc[df_consistency['mean_std'].idxmin(), 'annotator']
    worst_annotator = df_consistency.loc[df_consistency['mean_std'].idxmax(), 'annotator']
    print(f"  Most consistent: {best_annotator.capitalize()}")
    print(f"  Least consistent: {worst_annotator.capitalize()}")

print(f"\nOverall mean std for repeated molecules: {df_variance['std_score'].mean():.3f}")
print("(Std > 1.0 indicates scoring varies significantly based on pair context)")
