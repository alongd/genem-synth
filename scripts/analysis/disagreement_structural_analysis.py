"""
disagreement_structural_analysis.py

Description:
------------
Part 3 of annotator agreement analysis:
- Identifies molecules/pairs with highest disagreement
- Analyzes functional group enrichment in disagreement cases
- Compares senior vs all-annotator disagreement patterns

Requires: generate_consensus_dataset.py outputs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from collections import defaultdict

# File paths
INPUT_PATH = Path("data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv").resolve()
CONSENSUS_PATH = Path("data/output/senior_consensus_dataset.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()

print("="*80)
print("STRUCTURAL DISAGREEMENT ANALYSIS")
print("="*80)

# Load data
df_original = pd.read_csv(INPUT_PATH)
df_consensus = pd.read_csv(CONSENSUS_PATH)

print(f"\nLoaded {len(df_original)} pairs with annotations")
print(f"Loaded {len(df_consensus)} consensus pairs\n")

# Define functional families (same as previous structural analysis)
FUNCTIONAL_FAMILIES = {
    'nitro_aromatic': {
        'name': 'Aromatic Nitro',
        'smarts': '[c][N+](=O)[O-]',
    },
    'nitro_aliphatic': {
        'name': 'Aliphatic Nitro',
        'smarts': '[C;!$([cH])]([N+](=O)[O-])',
    },
    'nitramine': {
        'name': 'Nitramine',
        'smarts': '[N;!$([N+])]([N+](=O)[O-])',
    },
    'nitrate_ester': {
        'name': 'Nitrate Ester',
        'smarts': '[O][N+](=O)[O-]',
    },
    'azide': {
        'name': 'Azide',
        'smarts': '[N-]=[N+]=[N]',
    },
    'tetrazole': {
        'name': 'Tetrazole',
        'smarts': 'c1nnnn1',
    },
    'triazole': {
        'name': 'Triazole',
        'smarts': 'c1nncn1',
    },
    'triazole_123': {
        'name': '1,2,3-Triazole',
        'smarts': 'c1cnnn1',
    },
    'gem_dinitro': {
        'name': 'gem-Dinitro',
        'smarts': '[C]([N+](=O)[O-])([N+](=O)[O-])',
    }
}

def identify_functional_groups(smiles):
    """Returns dict of functional group presence for a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    groups = {}
    for key, family in FUNCTIONAL_FAMILIES.items():
        pattern = Chem.MolFromSmarts(family['smarts'])
        if pattern is not None:
            groups[key] = mol.HasSubstructMatch(pattern)
        else:
            groups[key] = False

    return groups

# ==============================================================================
# STEP 1: Identify High Disagreement Pairs
# ==============================================================================

print("="*80)
print("STEP 1: IDENTIFYING HIGH DISAGREEMENT PAIRS")
print("="*80)

# Define high disagreement as combined_std > 1.5 or quality_tier = noise
high_disagreement = df_consensus[
    (df_consensus['combined_std'] > 1.5) |
    (df_consensus['quality_tier'] == 'tier_4_noise')
].copy()

low_disagreement = df_consensus[
    (df_consensus['combined_std'] <= 1.0) &
    (df_consensus['quality_tier'].isin(['tier_1_high', 'tier_2_moderate']))
].copy()

print(f"\nHigh disagreement pairs (std > 1.5 or tier 4): {len(high_disagreement)} ({100*len(high_disagreement)/len(df_consensus):.1f}%)")
print(f"Low disagreement pairs (std <= 1.0 and tier 1-2): {len(low_disagreement)} ({100*len(low_disagreement)/len(df_consensus):.1f}%)")

# Top 20 most disagreed pairs
top_disagreement = df_consensus.nlargest(20, 'combined_std')
print(f"\nTop 20 pairs with highest disagreement (by combined std):")
print("-" * 80)
for idx, row in top_disagreement.iterrows():
    print(f"Pair {row['pair_id']}: std = {row['combined_std']:.2f}, sep = {row['score_separation']:.2f}, tier = {row['quality_tier']}")

# ==============================================================================
# STEP 2: Functional Group Analysis of Disagreement
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: FUNCTIONAL GROUP ENRICHMENT IN DISAGREEMENT")
print("="*80)

# Annotate all molecules with functional groups
print("\nAnnotating molecules with functional groups...")
mol_annotations = {}

for idx, row in df_consensus.iterrows():
    for mol_type, smiles_col in [('a', 'smiles_a'), ('b', 'smiles_b')]:
        smiles = row[smiles_col]
        if smiles not in mol_annotations:
            groups = identify_functional_groups(smiles)
            if groups is not None:
                mol_annotations[smiles] = groups

print(f"Annotated {len(mol_annotations)} unique molecules")

# Count functional groups in high vs low disagreement pairs
high_disagree_fg_counts = defaultdict(int)
low_disagree_fg_counts = defaultdict(int)

for idx, row in high_disagreement.iterrows():
    for smiles_col in ['smiles_a', 'smiles_b']:
        smiles = row[smiles_col]
        if smiles in mol_annotations:
            for fg_key, present in mol_annotations[smiles].items():
                if present:
                    high_disagree_fg_counts[fg_key] += 1

for idx, row in low_disagreement.iterrows():
    for smiles_col in ['smiles_a', 'smiles_b']:
        smiles = row[smiles_col]
        if smiles in mol_annotations:
            for fg_key, present in mol_annotations[smiles].items():
                if present:
                    low_disagree_fg_counts[fg_key] += 1

# Calculate enrichment
n_high_mols = len(high_disagreement) * 2
n_low_mols = len(low_disagreement) * 2

print(f"\nFunctional group enrichment in disagreement:")
print("=" * 80)
print(f"{'Group':20s} {'High Disagree':>15s} {'Low Disagree':>15s} {'Enrichment':>12s}")
print("-" * 80)

enrichment_results = []

for fg_key, fg_info in FUNCTIONAL_FAMILIES.items():
    high_count = high_disagree_fg_counts[fg_key]
    low_count = low_disagree_fg_counts[fg_key]

    high_rate = high_count / n_high_mols if n_high_mols > 0 else 0
    low_rate = low_count / n_low_mols if n_low_mols > 0 else 0

    enrichment = high_rate / low_rate if low_rate > 0 else 0

    enrichment_results.append({
        'functional_group': fg_info['name'],
        'high_count': high_count,
        'low_count': low_count,
        'high_rate': high_rate,
        'low_rate': low_rate,
        'enrichment': enrichment
    })

    print(f"{fg_info['name']:20s} {high_count:6d} ({high_rate*100:5.1f}%)  {low_count:6d} ({low_rate*100:5.1f}%)  {enrichment:8.2f}x")

df_enrichment = pd.DataFrame(enrichment_results)

# ==============================================================================
# STEP 3: Visualization
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Disagreement distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combined std distribution
axes[0].hist(df_consensus['combined_std'].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(1.5, color='red', linestyle='--', linewidth=2, label='High disagreement threshold')
axes[0].set_xlabel('Combined Standard Deviation', fontsize=11)
axes[0].set_ylabel('Number of Pairs', fontsize=11)
axes[0].set_title('Distribution of Annotator Disagreement', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Score separation distribution
axes[1].hist(df_consensus['score_separation'].dropna(), bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Score Separation |A - B|', fontsize=11)
axes[1].set_ylabel('Number of Pairs', fontsize=11)
axes[1].set_title('Distribution of Score Separation', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
dist_plot_path = OUTPUT_DIR / "disagreement_distribution.png"
plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved disagreement distribution to {dist_plot_path}")
plt.close()

# Plot 2: Functional group enrichment heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data for heatmap
fg_names = [r['functional_group'] for r in enrichment_results]
enrichments = [r['enrichment'] for r in enrichment_results]
high_rates = [r['high_rate'] * 100 for r in enrichment_results]
low_rates = [r['low_rate'] * 100 for r in enrichment_results]

# Create bar plot with two bars per group
x = np.arange(len(fg_names))
width = 0.35

bars1 = ax.barh(x - width/2, high_rates, width, label='High Disagreement', color='#e74c3c', alpha=0.8)
bars2 = ax.barh(x + width/2, low_rates, width, label='Low Disagreement', color='#3498db', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(fg_names, fontsize=10)
ax.set_xlabel('Occurrence Rate (%)', fontsize=11)
ax.set_title('Functional Group Occurrence: High vs Low Disagreement Pairs',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)

# Add enrichment values as text
for i, enrich in enumerate(enrichments):
    if enrich > 1.2 or enrich < 0.8:
        color = 'red' if enrich > 1.2 else 'green'
        ax.text(max(high_rates[i], low_rates[i]) + 1, i, f'{enrich:.2f}x',
                va='center', color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
fg_plot_path = OUTPUT_DIR / "disagreement_functional_groups.png"
plt.savefig(fg_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved functional group analysis to {fg_plot_path}")
plt.close()

# ==============================================================================
# STEP 4: Save Results
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: SAVING DISAGREEMENT ANALYSIS")
print("="*80)

# Save enrichment analysis
enrichment_path = OUTPUT_DIR / "disagreement_functional_enrichment.csv"
df_enrichment.to_csv(enrichment_path, index=False)
print(f"✅ Saved enrichment analysis to {enrichment_path}")

# Save high disagreement pairs
high_disagree_path = OUTPUT_DIR / "high_disagreement_pairs.csv"
high_disagreement.to_csv(high_disagree_path, index=False)
print(f"✅ Saved high disagreement pairs to {high_disagree_path}")

# Save detailed analysis report
report_path = OUTPUT_DIR / "disagreement_analysis.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("STRUCTURAL DISAGREEMENT ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("DISAGREEMENT CLASSIFICATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"High disagreement (std > 1.5): {len(high_disagreement)} pairs ({100*len(high_disagreement)/len(df_consensus):.1f}%)\n")
    f.write(f"Low disagreement (std <= 1.0): {len(low_disagreement)} pairs ({100*len(low_disagreement)/len(df_consensus):.1f}%)\n\n")

    f.write("FUNCTIONAL GROUP ENRICHMENT\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Group':20s} {'High Disagree':>15s} {'Low Disagree':>15s} {'Enrichment':>12s}\n")
    f.write("-" * 80 + "\n")
    for _, row in df_enrichment.iterrows():
        f.write(f"{row['functional_group']:20s} {row['high_count']:6.0f} ({row['high_rate']*100:5.1f}%)  ")
        f.write(f"{row['low_count']:6.0f} ({row['low_rate']*100:5.1f}%)  {row['enrichment']:8.2f}x\n")

    f.write("\n" + "TOP 20 MOST DISAGREED PAIRS\n")
    f.write("-" * 80 + "\n")
    for idx, row in top_disagreement.iterrows():
        f.write(f"\nPair {row['pair_id']}:\n")
        f.write(f"  Combined std: {row['combined_std']:.2f}\n")
        f.write(f"  Score separation: {row['score_separation']:.2f}\n")
        f.write(f"  Quality tier: {row['quality_tier']}\n")
        f.write(f"  Votes: A={row['vote_a']}, B={row['vote_b']}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*80 + "\n\n")
    f.write("Enrichment > 1.0 indicates functional group is over-represented in\n")
    f.write("high-disagreement pairs, suggesting this motif is difficult for\n")
    f.write("annotators to evaluate consistently.\n\n")
    f.write("Enrichment < 1.0 indicates functional group appears more often in\n")
    f.write("low-disagreement pairs, suggesting clearer consensus on molecules\n")
    f.write("with these features.\n")

print(f"✅ Saved disagreement report to {report_path}")

print("\n" + "="*80)
print("DISAGREEMENT ANALYSIS COMPLETE")
print("="*80)
