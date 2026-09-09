"""
annotator_agreement_analysis.py

Description:
------------
Analyzes inter-annotator agreement for synthesizability pairwise comparisons.

Annotators:
- Senior: Asaf, Elena, Levi
- Junior: Roman

Analyses:
1. Inter-annotator agreement: Krippendorff's α, Kendall's W (all 4 & senior 3)
2. Pairwise Spearman rank correlations
3. Structural regions of greatest disagreement
4. Senior consensus scores and quality tiers

Score interpretation: 1 = easiest to synthesize, 5 = hardest to synthesize
UNSURE handling: Scores are used as-is; choice treated as missing data for ranking
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import kendalltau
from itertools import combinations
from collections import defaultdict

# File paths
INPUT_PATH = Path("data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"\nLoaded {len(df)} pairwise comparisons")
print(f"Columns: {df.columns.tolist()}\n")

# Define annotators
SENIOR_ANNOTATORS = ['asaf', 'elena', 'levi']
JUNIOR_ANNOTATORS = ['roman']
ALL_ANNOTATORS = SENIOR_ANNOTATORS + JUNIOR_ANNOTATORS

print("Annotator groups:")
print(f"  Senior: {', '.join(SENIOR_ANNOTATORS)}")
print(f"  Junior: {', '.join(JUNIOR_ANNOTATORS)}")
print(f"  All: {', '.join(ALL_ANNOTATORS)}\n")

# ==============================================================================
# STEP 1: Data Processing & Extraction
# ==============================================================================

print("="*80)
print("STEP 1: EXTRACTING SCORES AND CHOICES")
print("="*80)

def extract_annotator_data(df, annotator):
    """Extract choice and scores for an annotator"""
    choice_col = f"{annotator} - Choice"
    score_a_col = f"{annotator} - Score A"
    score_b_col = f"{annotator} - Score B"

    # Handle roman's different column name format
    if annotator == 'roman':
        score_b_col = "roman- Score B"  # Note the hyphen difference

    data = []
    for idx, row in df.iterrows():
        choice = row.get(choice_col, np.nan)
        score_a = row.get(score_a_col, np.nan)
        score_b = row.get(score_b_col, np.nan)

        # Handle empty strings as NaN
        if pd.isna(choice) or choice == '':
            choice = np.nan
        if pd.isna(score_a) or score_a == '':
            score_a = np.nan
        else:
            score_a = float(score_a)
        if pd.isna(score_b) or score_b == '':
            score_b = np.nan
        else:
            score_b = float(score_b)

        data.append({
            'pair_id': row['Pair ID'],
            'annotator': annotator,
            'choice': choice,
            'score_a': score_a,
            'score_b': score_b
        })

    return pd.DataFrame(data)

# Extract data for all annotators
annotator_data = []
for annotator in ALL_ANNOTATORS:
    annotator_data.append(extract_annotator_data(df, annotator))

df_annotators = pd.concat(annotator_data, ignore_index=True)

print(f"Extracted {len(df_annotators)} annotator responses")
print(f"Missing choices: {df_annotators['choice'].isna().sum()}")
print(f"Missing score A: {df_annotators['score_a'].isna().sum()}")
print(f"Missing score B: {df_annotators['score_b'].isna().sum()}")

# Count UNSURE responses
unsure_counts = {}
for annotator in ALL_ANNOTATORS:
    annotator_df = df_annotators[df_annotators['annotator'] == annotator]
    unsure_count = (annotator_df['choice'] == 'UNSURE').sum()
    unsure_counts[annotator] = unsure_count

print("\nUNSURE response counts:")
for annotator, count in unsure_counts.items():
    pct = 100 * count / len(df)
    print(f"  {annotator:6s}: {count:3d} ({pct:4.1f}%)")

# ==============================================================================
# STEP 2: Krippendorff's Alpha
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: KRIPPENDORFF'S ALPHA")
print("="*80)

def krippendorff_alpha_ordinal(data_matrix):
    """
    Calculate Krippendorff's alpha for ordinal data.

    data_matrix: numpy array of shape (n_annotators, n_items)
                 NaN values represent missing data
    """
    # Convert to numpy array
    data = np.array(data_matrix, dtype=float)
    n_annotators, n_items = data.shape

    # Calculate observed disagreement
    n_pairs = 0
    disagreement_sum = 0

    for item_idx in range(n_items):
        values = data[:, item_idx]
        valid_values = values[~np.isnan(values)]

        if len(valid_values) < 2:
            continue

        # For each pair of annotators rating this item
        for i in range(len(valid_values)):
            for j in range(i + 1, len(valid_values)):
                # Ordinal distance: squared difference
                disagreement_sum += (valid_values[i] - valid_values[j]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return np.nan

    observed_disagreement = disagreement_sum / n_pairs

    # Calculate expected disagreement (marginal distribution)
    all_values = data[~np.isnan(data)]
    n_values = len(all_values)

    if n_values < 2:
        return np.nan

    expected_disagreement_sum = 0
    for i in range(n_values):
        for j in range(i + 1, n_values):
            expected_disagreement_sum += (all_values[i] - all_values[j]) ** 2

    expected_disagreement = expected_disagreement_sum / (n_values * (n_values - 1) / 2)

    if expected_disagreement == 0:
        return 1.0  # Perfect agreement

    alpha = 1 - (observed_disagreement / expected_disagreement)
    return alpha

# Prepare data matrix for all annotators
# For each pair, we have 2 molecules, so 2*n_pairs items
n_pairs = len(df)
n_items = 2 * n_pairs  # Each pair has 2 molecules

# All annotators matrix
data_all = np.full((len(ALL_ANNOTATORS), n_items), np.nan)

for ann_idx, annotator in enumerate(ALL_ANNOTATORS):
    ann_data = df_annotators[df_annotators['annotator'] == annotator]

    for seq_idx, (_, row) in enumerate(ann_data.iterrows()):
        mol_a_idx = seq_idx * 2
        mol_b_idx = seq_idx * 2 + 1

        if not np.isnan(row['score_a']):
            data_all[ann_idx, mol_a_idx] = row['score_a']
        if not np.isnan(row['score_b']):
            data_all[ann_idx, mol_b_idx] = row['score_b']

# Senior annotators matrix
data_senior = np.full((len(SENIOR_ANNOTATORS), n_items), np.nan)

for ann_idx, annotator in enumerate(SENIOR_ANNOTATORS):
    ann_data = df_annotators[df_annotators['annotator'] == annotator]

    for seq_idx, (_, row) in enumerate(ann_data.iterrows()):
        mol_a_idx = seq_idx * 2
        mol_b_idx = seq_idx * 2 + 1

        if not np.isnan(row['score_a']):
            data_senior[ann_idx, mol_a_idx] = row['score_a']
        if not np.isnan(row['score_b']):
            data_senior[ann_idx, mol_b_idx] = row['score_b']

# Calculate Krippendorff's alpha
alpha_all = krippendorff_alpha_ordinal(data_all)
alpha_senior = krippendorff_alpha_ordinal(data_senior)

print(f"\nKrippendorff's α (ordinal):")
print(f"  All 4 annotators:    {alpha_all:.4f}")
print(f"  Senior 3 annotators: {alpha_senior:.4f}")

print("\nInterpretation:")
print("  α > 0.800: Highly reliable")
print("  α > 0.667: Tentatively reliable")
print("  α > 0.400: Low reliability")
print("  α < 0.400: Unreliable")

# ==============================================================================
# STEP 3: Kendall's W
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: KENDALL'S W (Coefficient of Concordance)")
print("="*80)

def calculate_kendalls_w(rankings_matrix):
    """
    Calculate Kendall's W for rankings.

    rankings_matrix: numpy array of shape (n_annotators, n_items)
                    NaN values represent missing data
    """
    data = np.array(rankings_matrix, dtype=float)
    n_annotators, n_items = data.shape

    # Remove items with too much missing data
    valid_items_mask = np.sum(~np.isnan(data), axis=0) >= 2
    data_clean = data[:, valid_items_mask]

    if data_clean.shape[1] < 2:
        return np.nan, np.nan

    # For each annotator, rank the items (handling ties)
    ranked_data = np.full_like(data_clean, np.nan)

    for ann_idx in range(n_annotators):
        values = data_clean[ann_idx, :]
        valid_mask = ~np.isnan(values)

        if np.sum(valid_mask) == 0:
            continue

        # Rank valid values (1 = easiest/lowest score, higher rank = harder)
        ranks = stats.rankdata(values[valid_mask], method='average')
        ranked_data[ann_idx, valid_mask] = ranks

    # Calculate sum of ranks for each item
    rank_sums = np.nansum(ranked_data, axis=0)

    # Calculate W
    n_items_clean = data_clean.shape[1]
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    # Maximum possible S
    S_max = (n_annotators ** 2 * (n_items_clean ** 3 - n_items_clean)) / 12

    if S_max == 0:
        return np.nan, np.nan

    W = S / S_max

    # Chi-square test for significance
    chi_square = n_annotators * (n_items_clean - 1) * W
    df = n_items_clean - 1
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    return W, p_value

W_all, p_all = calculate_kendalls_w(data_all)
W_senior, p_senior = calculate_kendalls_w(data_senior)

print(f"\nKendall's W:")
print(f"  All 4 annotators:    W = {W_all:.4f}, p = {p_all:.4e}")
print(f"  Senior 3 annotators: W = {W_senior:.4f}, p = {p_senior:.4e}")

print("\nInterpretation:")
print("  W = 1.0: Perfect agreement")
print("  W = 0.0: No agreement")
print("  p < 0.05: Significant agreement")

# ==============================================================================
# STEP 4: Pairwise Spearman Correlations
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: PAIRWISE SPEARMAN CORRELATIONS")
print("="*80)

# Create correlation matrix
correlation_matrix = np.full((len(ALL_ANNOTATORS), len(ALL_ANNOTATORS)), np.nan)
p_value_matrix = np.full((len(ALL_ANNOTATORS), len(ALL_ANNOTATORS)), np.nan)

for i, ann1 in enumerate(ALL_ANNOTATORS):
    for j, ann2 in enumerate(ALL_ANNOTATORS):
        if i == j:
            correlation_matrix[i, j] = 1.0
            p_value_matrix[i, j] = 0.0
            continue

        # Get scores for both annotators
        ann1_data = df_annotators[df_annotators['annotator'] == ann1]
        ann2_data = df_annotators[df_annotators['annotator'] == ann2]

        # Flatten all scores (both molecule A and B)
        scores1 = []
        scores2 = []

        for idx in range(len(df)):
            s1a = ann1_data.iloc[idx]['score_a']
            s1b = ann1_data.iloc[idx]['score_b']
            s2a = ann2_data.iloc[idx]['score_a']
            s2b = ann2_data.iloc[idx]['score_b']

            if not np.isnan(s1a) and not np.isnan(s2a):
                scores1.append(s1a)
                scores2.append(s2a)

            if not np.isnan(s1b) and not np.isnan(s2b):
                scores1.append(s1b)
                scores2.append(s2b)

        if len(scores1) > 2:
            rho, p_val = stats.spearmanr(scores1, scores2)
            correlation_matrix[i, j] = rho
            p_value_matrix[i, j] = p_val

print("\nSpearman Correlation Matrix:")
print("=" * 60)
corr_df = pd.DataFrame(correlation_matrix,
                       index=ALL_ANNOTATORS,
                       columns=ALL_ANNOTATORS)
print(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))

print("\n\nPairwise Correlations:")
print("-" * 60)
for i, ann1 in enumerate(ALL_ANNOTATORS):
    for j, ann2 in enumerate(ALL_ANNOTATORS):
        if i < j:
            rho = correlation_matrix[i, j]
            p_val = p_value_matrix[i, j]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {ann1:6s} vs {ann2:6s}: ρ = {rho:.4f}, p = {p_val:.4e} {sig}")

# Create correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(correlation_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(ALL_ANNOTATORS)))
ax.set_yticks(range(len(ALL_ANNOTATORS)))
ax.set_xticklabels(ALL_ANNOTATORS)
ax.set_yticklabels(ALL_ANNOTATORS)

# Add correlation values as text
for i in range(len(ALL_ANNOTATORS)):
    for j in range(len(ALL_ANNOTATORS)):
        value = correlation_matrix[i, j]
        if not np.isnan(value):
            text = ax.text(j, i, f'{value:.3f}',
                          ha='center', va='center',
                          color='white' if value < 0.5 else 'black',
                          fontsize=11, fontweight='bold')

ax.set_title('Pairwise Spearman Correlations\nBetween Annotators',
             fontsize=13, fontweight='bold', pad=15)
plt.colorbar(im, ax=ax, label='Spearman ρ')
plt.tight_layout()

corr_plot_path = OUTPUT_DIR / "annotator_correlations.png"
plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved correlation heatmap to {corr_plot_path}")
plt.close()

# Save correlation matrix to CSV
corr_csv_path = OUTPUT_DIR / "pairwise_correlations.csv"
corr_df.to_csv(corr_csv_path)
print(f"✅ Saved correlation matrix to {corr_csv_path}")

# ==============================================================================
# Additional: Individual Annotator Score Distributions
# ==============================================================================

print("\n" + "="*80)
print("INDIVIDUAL ANNOTATOR SCORE DISTRIBUTIONS")
print("="*80)

# Calculate distribution statistics for each annotator
annotator_stats = []

for annotator in ALL_ANNOTATORS:
    ann_data = df_annotators[df_annotators['annotator'] == annotator]

    # Combine all scores (A and B)
    all_scores = []
    for _, row in ann_data.iterrows():
        if not np.isnan(row['score_a']):
            all_scores.append(row['score_a'])
        if not np.isnan(row['score_b']):
            all_scores.append(row['score_b'])

    all_scores = np.array(all_scores)

    stats_dict = {
        'annotator': annotator,
        'n_scores': len(all_scores),
        'mean': np.mean(all_scores),
        'std': np.std(all_scores),
        'median': np.median(all_scores),
        'min': np.min(all_scores),
        'max': np.max(all_scores),
        'q25': np.percentile(all_scores, 25),
        'q75': np.percentile(all_scores, 75)
    }
    annotator_stats.append(stats_dict)

df_stats = pd.DataFrame(annotator_stats)

print("\nScore Distribution Statistics:")
print("=" * 80)
print(df_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Create distribution plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, annotator in enumerate(ALL_ANNOTATORS):
    ax = axes[idx]
    ann_data = df_annotators[df_annotators['annotator'] == annotator]

    # Combine all scores
    all_scores = []
    for _, row in ann_data.iterrows():
        if not np.isnan(row['score_a']):
            all_scores.append(row['score_a'])
        if not np.isnan(row['score_b']):
            all_scores.append(row['score_b'])

    # Plot histogram
    ax.hist(all_scores, bins=np.arange(0.5, 6.5, 1), color='steelblue',
            edgecolor='black', alpha=0.7)

    # Add mean line
    mean_val = np.mean(all_scores)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f}')

    # Styling
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{annotator.capitalize()} (n={len(all_scores)})',
                fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.5, 5.5)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
dist_plot_path = OUTPUT_DIR / "individual_annotator_distributions.png"
plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved annotator distributions to {dist_plot_path}")
plt.close()

# Create box plot comparison
fig, ax = plt.subplots(figsize=(10, 6))

annotator_scores = []
annotator_labels = []

for annotator in ALL_ANNOTATORS:
    ann_data = df_annotators[df_annotators['annotator'] == annotator]
    scores = []
    for _, row in ann_data.iterrows():
        if not np.isnan(row['score_a']):
            scores.append(row['score_a'])
        if not np.isnan(row['score_b']):
            scores.append(row['score_b'])
    annotator_scores.append(scores)
    annotator_labels.append(annotator.capitalize())

bp = ax.boxplot(annotator_scores, labels=annotator_labels, patch_artist=True,
                showmeans=True, meanline=True)

# Color senior vs junior differently
colors = ['lightblue' if ann in SENIOR_ANNOTATORS else 'lightcoral' for ann in ALL_ANNOTATORS]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Annotator', fontsize=12)
ax.set_title('Score Distribution Comparison Across Annotators', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0.5, 5.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', label='Senior Annotators'),
    Patch(facecolor='lightcoral', label='Junior Annotators')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
box_plot_path = OUTPUT_DIR / "annotator_score_boxplots.png"
plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved box plot comparison to {box_plot_path}")
plt.close()

# Save statistics to CSV
stats_csv_path = OUTPUT_DIR / "annotator_score_statistics.csv"
df_stats.to_csv(stats_csv_path, index=False)
print(f"✅ Saved statistics to {stats_csv_path}")

print("\n" + "="*80)
print("STEP 1-4 COMPLETE: Basic agreement metrics calculated")
print("="*80)

# Save comprehensive summary
summary_path = OUTPUT_DIR / "inter_annotator_agreement_summary.txt"
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("INTER-ANNOTATOR AGREEMENT ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total pairwise comparisons: {len(df)}\n")
    f.write(f"Annotators: {', '.join(ALL_ANNOTATORS)}\n")
    f.write(f"Senior annotators: {', '.join(SENIOR_ANNOTATORS)}\n\n")

    f.write("UNSURE RESPONSE RATES\n")
    f.write("-" * 80 + "\n")
    for annotator, count in unsure_counts.items():
        pct = 100 * count / len(df)
        f.write(f"  {annotator:6s}: {count:3d} ({pct:4.1f}%)\n")

    f.write("\n" + "KRIPPENDORFF'S ALPHA (Ordinal)\n")
    f.write("-" * 80 + "\n")
    f.write(f"All 4 annotators:    α = {alpha_all:.4f}\n")
    f.write(f"Senior 3 annotators: α = {alpha_senior:.4f}\n\n")
    f.write("Interpretation:\n")
    f.write("  α > 0.800: Highly reliable\n")
    f.write("  α > 0.667: Tentatively reliable\n")
    f.write("  α > 0.400: Low reliability\n")
    f.write("  α < 0.400: Unreliable\n")

    f.write("\n" + "KENDALL'S W (Coefficient of Concordance)\n")
    f.write("-" * 80 + "\n")
    f.write(f"All 4 annotators:    W = {W_all:.4f}, p = {p_all:.4e}\n")
    f.write(f"Senior 3 annotators: W = {W_senior:.4f}, p = {p_senior:.4e}\n\n")
    f.write("Interpretation:\n")
    f.write("  W = 1.0: Perfect agreement\n")
    f.write("  W = 0.0: No agreement\n")
    f.write("  p < 0.05: Significant agreement\n")

    f.write("\n" + "PAIRWISE SPEARMAN CORRELATIONS\n")
    f.write("-" * 80 + "\n")
    for i, ann1 in enumerate(ALL_ANNOTATORS):
        for j, ann2 in enumerate(ALL_ANNOTATORS):
            if i < j:
                rho = correlation_matrix[i, j]
                p_val = p_value_matrix[i, j]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                f.write(f"  {ann1:6s} vs {ann2:6s}: ρ = {rho:.4f}, p = {p_val:.4e} {sig}\n")

    f.write("\n" + "INDIVIDUAL ANNOTATOR SCORE DISTRIBUTIONS\n")
    f.write("-" * 80 + "\n")
    f.write(df_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    f.write("\n")

    f.write("\n" + "="*80 + "\n")
    f.write("FILES GENERATED\n")
    f.write("="*80 + "\n")
    f.write("  - annotator_correlations.png: Correlation heatmap\n")
    f.write("  - pairwise_correlations.csv: Correlation matrix\n")
    f.write("  - individual_annotator_distributions.png: Histograms\n")
    f.write("  - annotator_score_boxplots.png: Box plot comparison\n")
    f.write("  - annotator_score_statistics.csv: Distribution statistics\n")
    f.write("  - inter_annotator_agreement_summary.txt: This file\n")

print(f"\n✅ Saved agreement summary to {summary_path}")

print("\nProceeding to disagreement analysis and consensus generation...")
