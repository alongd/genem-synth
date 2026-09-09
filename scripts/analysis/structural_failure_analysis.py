"""
structural_failure_analysis.py

Description:
------------
Performs structural analysis of systematic failure cases by grouping EMs by
functional family to identify specific chemical features responsible for
mis-scoring by synthesizability tools.

Functional families analyzed:
- Nitro groups (aromatic and aliphatic C-NO2)
- Nitrate ester groups (O-NO2)
- Nitramine groups (N-NO2)
- Azide groups
- Tetrazole rings
- Triazole rings
- gem-Dinitro motifs

Analysis identifies:
1. Molecules with poor scores despite confirmed synthesis (misclassified as hard)
2. Molecules with very good scores despite specialized chemistry (opposite error)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import defaultdict

# File paths
EM_SCORES_PATH = Path("data/processed/scored.csv").resolve()
MOSES_SCORED_PATH = Path("data/processed/moses_sample_10k_scored.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()

print("="*80)
print("STRUCTURAL FAILURE ANALYSIS")
print("="*80)
print("\nAnalyzing mis-scoring patterns by functional family\n")

# Load data
df_em = pd.read_csv(EM_SCORES_PATH)
df_moses = pd.read_csv(MOSES_SCORED_PATH)

# Standardize column names
def standardize_column_name(col):
    col_lower = col.lower().strip()
    if 'sascore' in col_lower:
        return 'sascore'
    elif 'scscore' in col_lower:
        return 'scscore'
    elif 'syba' in col_lower:
        return 'syba'
    elif 'rascore_tf' in col_lower:
        return 'rascore_tf'
    elif 'rascore_xgb' in col_lower:
        return 'rascore_xgb'
    elif 'fsscore' in col_lower:
        return 'fsscore'
    return col

df_em.columns = [standardize_column_name(col) for col in df_em.columns]
df_moses.columns = [standardize_column_name(col) for col in df_moses.columns]

print(f"Loaded {len(df_em)} energetic materials")
print(f"Loaded {len(df_moses)} MOSES molecules\n")

# Define SMARTS patterns for functional families
FUNCTIONAL_FAMILIES = {
    'nitro_aromatic': {
        'name': 'Aromatic Nitro',
        'smarts': '[c][N+](=O)[O-]',
        'description': 'Nitro group attached to aromatic carbon'
    },
    'nitro_aliphatic': {
        'name': 'Aliphatic Nitro',
        'smarts': '[C;!$([cH])]([N+](=O)[O-])',
        'description': 'Nitro group attached to aliphatic carbon'
    },
    'nitramine': {
        'name': 'Nitramine',
        'smarts': '[N;!$([N+])]([N+](=O)[O-])',
        'description': 'N-NO2 group (nitramine)'
    },
    'nitrate_ester': {
        'name': 'Nitrate Ester',
        'smarts': '[O][N+](=O)[O-]',
        'description': 'O-NO2 group (nitrate ester)'
    },
    'azide': {
        'name': 'Azide',
        'smarts': '[N-]=[N+]=[N]',
        'description': 'Azide group (-N3)'
    },
    'tetrazole': {
        'name': 'Tetrazole',
        'smarts': 'c1nnnn1',
        'description': 'Tetrazole ring'
    },
    'triazole': {
        'name': 'Triazole',
        'smarts': 'c1nncn1',
        'description': '1,2,4-Triazole ring'
    },
    'triazole_123': {
        'name': '1,2,3-Triazole',
        'smarts': 'c1cnnn1',
        'description': '1,2,3-Triazole ring'
    },
    'gem_dinitro': {
        'name': 'gem-Dinitro',
        'smarts': '[C]([N+](=O)[O-])([N+](=O)[O-])',
        'description': 'Carbon with two nitro groups (gem-dinitro)'
    }
}

print("="*80)
print("IDENTIFYING FUNCTIONAL GROUPS")
print("="*80)

# Function to identify functional groups in a molecule
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

# Identify functional groups in all EMs
print("\nScanning energetic materials for functional groups...")
em_groups = []
for idx, row in df_em.iterrows():
    groups = identify_functional_groups(row['smiles'])
    if groups is not None:
        groups['smiles'] = row['smiles']
        groups['idx'] = idx
        em_groups.append(groups)

df_em_groups = pd.DataFrame(em_groups)

# Merge with scores
df_em_analysis = df_em.merge(df_em_groups, left_index=True, right_on='idx')

print(f"Successfully analyzed {len(df_em_analysis)} molecules\n")

# Count molecules per functional family
print("Functional group distribution:")
print("-" * 80)
for key, family in FUNCTIONAL_FAMILIES.items():
    count = df_em_analysis[key].sum()
    pct = 100 * count / len(df_em_analysis)
    print(f"{family['name']:20s}: {count:3d} molecules ({pct:5.1f}%)")

print("\n" + "="*80)
print("MISCLASSIFICATION ANALYSIS BY FUNCTIONAL FAMILY")
print("="*80)

# Define thresholds
thresholds = {
    'sascore': {'threshold': 6.0, 'condition': 'greater', 'name': 'SA Score'},
    'sascore_45': {'threshold': 4.5, 'condition': 'greater', 'name': 'SA Score (4.5)'},
    'scscore': {'threshold': 3.0, 'condition': 'greater', 'name': 'SCScore'},
    'syba': {'threshold': 0.0, 'condition': 'less', 'name': 'SYBA'},
    'rascore_tf': {'threshold': 0.5, 'condition': 'less', 'name': 'RAscore'},
    'fsscore': {'threshold': df_moses['fsscore'].median(), 'condition': 'less', 'name': 'FSscore'}
}

# Define opposite error thresholds
opposite_thresholds = {
    'scscore': {'threshold': df_moses['scscore'].median(), 'condition': 'less', 'name': 'SCScore (easier)'},
    'fsscore': {'threshold': df_moses['fsscore'].median(), 'condition': 'greater', 'name': 'FSscore (easier)'}
}

# Calculate misclassification rates per functional family
results = []

for tool_key, tool_info in thresholds.items():
    score_col = tool_key.split('_')[0] if '_' in tool_key and tool_key != 'rascore_tf' else tool_key

    print(f"\n{tool_info['name']} (threshold: {tool_info['threshold']:.2f}):")
    print("-" * 80)

    # Overall misclassification rate
    scores = df_em_analysis[score_col].dropna()
    if tool_info['condition'] == 'greater':
        total_misclass = (scores > tool_info['threshold']).sum()
    else:
        total_misclass = (scores < tool_info['threshold']).sum()

    overall_rate = 100 * total_misclass / len(scores)
    print(f"Overall: {total_misclass}/{len(scores)} ({overall_rate:.1f}%)")

    # Per-family misclassification rates
    for fg_key, fg_info in FUNCTIONAL_FAMILIES.items():
        # Filter to molecules with this functional group
        fg_mols = df_em_analysis[df_em_analysis[fg_key] == True]

        if len(fg_mols) == 0:
            continue

        fg_scores = fg_mols[score_col].dropna()

        if len(fg_scores) == 0:
            continue

        # Count misclassifications
        if tool_info['condition'] == 'greater':
            fg_misclass = (fg_scores > tool_info['threshold']).sum()
        else:
            fg_misclass = (fg_scores < tool_info['threshold']).sum()

        fg_rate = 100 * fg_misclass / len(fg_scores)

        results.append({
            'tool': tool_info['name'],
            'functional_group': fg_info['name'],
            'n_molecules': len(fg_scores),
            'n_misclassified': fg_misclass,
            'misclass_rate': fg_rate,
            'overall_rate': overall_rate,
            'enrichment': fg_rate / overall_rate if overall_rate > 0 else 0,
            'error_type': 'flagged_as_hard'
        })

        print(f"  {fg_info['name']:20s}: {fg_misclass:3d}/{len(fg_scores):3d} ({fg_rate:5.1f}%) " +
              f"[enrichment: {fg_rate/overall_rate if overall_rate > 0 else 0:.2f}x]")

# Opposite errors
print("\n" + "="*80)
print("OPPOSITE ERROR ANALYSIS BY FUNCTIONAL FAMILY")
print("="*80)

for tool_key, tool_info in opposite_thresholds.items():
    print(f"\n{tool_info['name']} (threshold: {tool_info['threshold']:.2f}):")
    print("-" * 80)

    # Overall opposite error rate
    scores = df_em_analysis[tool_key].dropna()
    if tool_info['condition'] == 'greater':
        total_opposite = (scores > tool_info['threshold']).sum()
    else:
        total_opposite = (scores < tool_info['threshold']).sum()

    overall_rate = 100 * total_opposite / len(scores)
    print(f"Overall: {total_opposite}/{len(scores)} ({overall_rate:.1f}%)")

    # Per-family opposite error rates
    for fg_key, fg_info in FUNCTIONAL_FAMILIES.items():
        fg_mols = df_em_analysis[df_em_analysis[fg_key] == True]

        if len(fg_mols) == 0:
            continue

        fg_scores = fg_mols[tool_key].dropna()

        if len(fg_scores) == 0:
            continue

        if tool_info['condition'] == 'greater':
            fg_opposite = (fg_scores > tool_info['threshold']).sum()
        else:
            fg_opposite = (fg_scores < tool_info['threshold']).sum()

        fg_rate = 100 * fg_opposite / len(fg_scores)

        results.append({
            'tool': tool_info['name'],
            'functional_group': fg_info['name'],
            'n_molecules': len(fg_scores),
            'n_misclassified': fg_opposite,
            'misclass_rate': fg_rate,
            'overall_rate': overall_rate,
            'enrichment': fg_rate / overall_rate if overall_rate > 0 else 0,
            'error_type': 'rated_as_easier'
        })

        print(f"  {fg_info['name']:20s}: {fg_opposite:3d}/{len(fg_scores):3d} ({fg_rate:5.1f}%) " +
              f"[enrichment: {fg_rate/overall_rate if overall_rate > 0 else 0:.2f}x]")

# Create results dataframe
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create heatmap of misclassification rates
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Misclassification rates (flagged as hard)
misclass_data = df_results[df_results['error_type'] == 'flagged_as_hard']
pivot_misclass = misclass_data.pivot(index='functional_group', columns='tool', values='misclass_rate')

if not pivot_misclass.empty:
    im1 = axes[0].imshow(pivot_misclass.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    axes[0].set_xticks(range(len(pivot_misclass.columns)))
    axes[0].set_xticklabels(pivot_misclass.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(pivot_misclass.index)))
    axes[0].set_yticklabels(pivot_misclass.index)
    axes[0].set_title('Misclassification Rates by Functional Group (% flagged as hard)',
                      fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(len(pivot_misclass.index)):
        for j in range(len(pivot_misclass.columns)):
            value = pivot_misclass.values[i, j]
            if not np.isnan(value):
                axes[0].text(j, i, f'{value:.1f}%', ha='center', va='center',
                           color='white' if value > 50 else 'black', fontsize=8)

    plt.colorbar(im1, ax=axes[0], label='Misclassification Rate (%)')

# Plot 2: Opposite error rates
opposite_data = df_results[df_results['error_type'] == 'rated_as_easier']
pivot_opposite = opposite_data.pivot(index='functional_group', columns='tool', values='misclass_rate')

if not pivot_opposite.empty:
    im2 = axes[1].imshow(pivot_opposite.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    axes[1].set_xticks(range(len(pivot_opposite.columns)))
    axes[1].set_xticklabels(pivot_opposite.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(pivot_opposite.index)))
    axes[1].set_yticklabels(pivot_opposite.index)
    axes[1].set_title('Opposite Error Rates by Functional Group (% rated easier than typical drug)',
                      fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(len(pivot_opposite.index)):
        for j in range(len(pivot_opposite.columns)):
            value = pivot_opposite.values[i, j]
            if not np.isnan(value):
                axes[1].text(j, i, f'{value:.1f}%', ha='center', va='center',
                           color='white' if value > 50 else 'black', fontsize=8)

    plt.colorbar(im2, ax=axes[1], label='Opposite Error Rate (%)')

plt.tight_layout()
plot_path = OUTPUT_DIR / "structural_failure_heatmap.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved heatmap to {plot_path}")
plt.close()

# Save detailed results
output_path = OUTPUT_DIR / "structural_failure_analysis.txt"
with open(output_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("STRUCTURAL FAILURE ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("FUNCTIONAL GROUP DEFINITIONS\n")
    f.write("-" * 80 + "\n")
    for fg_key, fg_info in FUNCTIONAL_FAMILIES.items():
        count = df_em_analysis[fg_key].sum()
        pct = 100 * count / len(df_em_analysis)
        f.write(f"\n{fg_info['name']}:\n")
        f.write(f"  SMARTS: {fg_info['smarts']}\n")
        f.write(f"  Description: {fg_info['description']}\n")
        f.write(f"  Count: {count} molecules ({pct:.1f}%)\n")

    f.write("\n\n" + "="*80 + "\n")
    f.write("MISCLASSIFICATION ENRICHMENT ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write("Enrichment > 1.0 indicates functional group is over-represented in failures\n")
    f.write("Enrichment < 1.0 indicates functional group is under-represented in failures\n\n")

    f.write(df_results.to_string(index=False))

print(f"✅ Saved detailed analysis to {output_path}")

# Save CSV
csv_path = OUTPUT_DIR / "structural_failure_analysis.csv"
df_results.to_csv(csv_path, index=False)
print(f"✅ Saved CSV to {csv_path}")

# Save molecule-level annotations
df_em_analysis.to_csv(OUTPUT_DIR / "em_functional_group_annotations.csv", index=False)
print(f"✅ Saved molecule annotations to {OUTPUT_DIR / 'em_functional_group_annotations.csv'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
