"""
calculate_misclassification.py

Description:
------------
Calculates the fraction of known-synthesized energetic materials (EMs) that are
incorrectly flagged as "hard to synthesize" by each tool using standard thresholds.

Thresholds:
- SA score: > 6 (standard community convention)
- SCScore: > 3 (primary) and > 4 (sensitivity check)
- SYBA: < 0 (original paper's decision boundary)
- RAscore: < 0.5 (standard binary classifier threshold)
- FSscore: < median of MOSES sample (relative to typical drug-like molecules)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
MOSES_SCORED_PATH = Path("data/processed/moses_sample_10k_scored.csv").resolve()
EM_SCORES_PATH = Path("data/processed/scored.csv").resolve()
OUTPUT_DIR = Path("data/output").resolve()

print("="*80)
print("MISCLASSIFICATION RATE ANALYSIS")
print("="*80)
print("\nCalculating the fraction of 416 known-synthesized EMs incorrectly")
print("flagged as 'hard to synthesize' by each tool.\n")

# Load data
df_moses = pd.read_csv(MOSES_SCORED_PATH)
df_em = pd.read_csv(EM_SCORES_PATH)

print(f"Loaded {len(df_moses):,} MOSES molecules")
print(f"Loaded {len(df_em):,} energetic materials\n")

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

df_moses.columns = [standardize_column_name(col) for col in df_moses.columns]
df_em.columns = [standardize_column_name(col) for col in df_em.columns]

print("MOSES columns:", df_moses.columns.tolist())
print("EM columns:", df_em.columns.tolist())
print()

# Define thresholds
thresholds = {
    'sascore': {
        'name': 'SA Score',
        'threshold': 6.0,
        'condition': 'greater',  # hard if score > threshold
        'rationale': 'Standard community convention from generative modelling literature'
    },
    'sascore_45': {
        'name': 'SA Score (4.5)',
        'score_col': 'sascore',
        'threshold': 4.5,
        'condition': 'greater',
        'rationale': 'Alternative threshold for sensitivity analysis'
    },
    'scscore_primary': {
        'name': 'SCScore',
        'score_col': 'scscore',
        'threshold': 3.0,
        'condition': 'greater',
        'rationale': 'Primary cutoff (absence of canonical threshold in original paper)'
    },
    'scscore_sensitivity': {
        'name': 'SCScore (sensitivity)',
        'score_col': 'scscore',
        'threshold': 4.0,
        'condition': 'greater',
        'rationale': 'Sensitivity check with stricter threshold'
    },
    'syba': {
        'name': 'SYBA',
        'threshold': 0.0,
        'condition': 'less',  # hard if score < threshold
        'rationale': 'Decision boundary from original paper by design'
    },
    'rascore_tf': {
        'name': 'RAscore',
        'threshold': 0.5,
        'condition': 'less',
        'rationale': 'Standard probability threshold for binary classifiers'
    },
    'fsscore': {
        'name': 'FSscore',
        'threshold': df_moses['fsscore'].median(),
        'condition': 'less',
        'rationale': f'Median of MOSES sample ({df_moses["fsscore"].median():.4f}) - below typical drug-like molecule'
    }
}

# Define opposite error rates (EMs rated as easier than typical drug-like molecule)
opposite_errors = {
    'scscore_easier': {
        'name': 'SCScore (easier than typical)',
        'score_col': 'scscore',
        'threshold': df_moses['scscore'].median(),
        'condition': 'less',  # easier if score < threshold (for SCScore, lower = easier)
        'rationale': f'Fraction of EMs with score < {df_moses["scscore"].median():.4f} (MOSES median) — rated easier than typical drug-like molecule',
        'error_type': 'opposite'
    },
    'fsscore_easier': {
        'name': 'FSscore (easier than typical)',
        'score_col': 'fsscore',
        'threshold': df_moses['fsscore'].median(),
        'condition': 'greater',  # easier if score > threshold (for FSscore, higher = easier)
        'rationale': f'Fraction of EMs with score > {df_moses["fsscore"].median():.4f} (MOSES median) — rated easier than typical drug-like molecule',
        'error_type': 'opposite'
    }
}

print("="*80)
print("THRESHOLDS")
print("="*80)
print("\nMisclassification thresholds (flagged as hard):")
for key, info in thresholds.items():
    score_col = info.get('score_col', key.split('_')[0])
    direction = ">" if info['condition'] == 'greater' else "<"
    print(f"\n{info['name']}:")
    print(f"  Hard if score {direction} {info['threshold']:.4f}")
    print(f"  Rationale: {info['rationale']}")

print("\n" + "-"*80)
print("Opposite error thresholds (rated as easier than typical):")
for key, info in opposite_errors.items():
    direction = ">" if info['condition'] == 'greater' else "<"
    print(f"\n{info['name']}:")
    print(f"  Easier if score {direction} {info['threshold']:.4f}")
    print(f"  Rationale: {info['rationale']}")

print("\n" + "="*80)
print("MISCLASSIFICATION RESULTS")
print("="*80)

results = []

for key, info in thresholds.items():
    # Get the score column name, handling compound keys like 'scscore_primary'
    if 'score_col' in info:
        score_col = info['score_col']
    else:
        # For keys like 'sascore', 'syba', 'rascore_tf', 'fsscore', use the key as-is
        score_col = key

    # Get EM scores (drop NaN values)
    em_scores = df_em[score_col].dropna()
    n_total = len(em_scores)

    # Calculate misclassifications
    if info['condition'] == 'greater':
        # Hard if score > threshold
        misclassified = (em_scores > info['threshold']).sum()
    else:
        # Hard if score < threshold
        misclassified = (em_scores < info['threshold']).sum()

    fraction = misclassified / n_total if n_total > 0 else 0
    percentage = fraction * 100

    results.append({
        'tool': info['name'],
        'threshold': info['threshold'],
        'condition': info['condition'],
        'n_total': n_total,
        'n_misclassified': misclassified,
        'fraction': fraction,
        'percentage': percentage,
        'rationale': info['rationale']
    })

    print(f"\n{info['name']}:")
    print(f"  Threshold: {info['threshold']:.4f}")
    print(f"  EMs flagged as hard: {misclassified}/{n_total} ({percentage:.1f}%)")
    print(f"  → {misclassified} known-synthesizable EMs incorrectly flagged as hard to synthesize")

# Create summary table
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print()
print(df_results[['tool', 'threshold', 'n_misclassified', 'n_total', 'percentage']].to_string(index=False))

# Calculate opposite errors (EMs rated as easier than typical drug-like molecule)
print("\n" + "="*80)
print("OPPOSITE ERROR RATES")
print("="*80)
print("\nFraction of EMs rated as EASIER than typical drug-like molecule:\n")

opposite_results = []

for key, info in opposite_errors.items():
    score_col = info['score_col']

    # Get EM scores (drop NaN values)
    em_scores = df_em[score_col].dropna()
    n_total = len(em_scores)

    # Calculate how many EMs are rated as easier than typical
    if info['condition'] == 'greater':
        # Easier if score > threshold (FSscore case)
        count_easier = (em_scores > info['threshold']).sum()
    else:
        # Easier if score < threshold (SCScore case)
        count_easier = (em_scores < info['threshold']).sum()

    fraction = count_easier / n_total if n_total > 0 else 0
    percentage = fraction * 100

    opposite_results.append({
        'tool': info['name'],
        'threshold': info['threshold'],
        'condition': info['condition'],
        'n_total': n_total,
        'n_easier': count_easier,
        'fraction': fraction,
        'percentage': percentage,
        'rationale': info['rationale']
    })

    print(f"{info['name']}:")
    print(f"  Threshold: {info['threshold']:.4f}")
    print(f"  EMs rated as easier: {count_easier}/{n_total} ({percentage:.1f}%)")
    print(f"  → {count_easier} EMs scored better than the median drug-like molecule\n")

df_opposite = pd.DataFrame(opposite_results)

# Save detailed results
output_path = OUTPUT_DIR / "misclassification_rates.txt"
with open(output_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MISCLASSIFICATION RATE ANALYSIS\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total energetic materials (known synthesizable): {len(df_em)}\n\n")

    f.write("THRESHOLDS AND RATIONALE\n")
    f.write("-" * 80 + "\n")
    for _, row in df_results.iterrows():
        direction = ">" if row['condition'] == 'greater' else "<"
        f.write(f"\n{row['tool']}:\n")
        f.write(f"  Threshold: score {direction} {row['threshold']:.4f}\n")
        f.write(f"  Rationale: {row['rationale']}\n")
        f.write(f"  Misclassified: {row['n_misclassified']}/{row['n_total']} ({row['percentage']:.1f}%)\n")

    f.write("\n" + "="*80 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("="*80 + "\n\n")
    f.write(df_results[['tool', 'threshold', 'n_misclassified', 'n_total', 'percentage']].to_string(index=False))

    f.write("\n\n" + "="*80 + "\n")
    f.write("OPPOSITE ERROR RATES\n")
    f.write("="*80 + "\n\n")
    f.write("Fraction of EMs rated as EASIER than typical drug-like molecule:\n\n")
    for _, row in df_opposite.iterrows():
        f.write(f"\n{row['tool']}:\n")
        f.write(f"  Threshold: {row['threshold']:.4f}\n")
        f.write(f"  Rationale: {row['rationale']}\n")
        f.write(f"  EMs rated as easier: {row['n_easier']}/{row['n_total']} ({row['percentage']:.1f}%)\n")

    f.write("\n\n" + "="*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*80 + "\n\n")
    f.write("MISCLASSIFICATION RATES (flagged as hard):\n")
    f.write("These represent the fraction of molecules with confirmed synthetic routes\n")
    f.write("that each tool incorrectly labels as 'hard to synthesize'.\n\n")
    f.write("OPPOSITE ERROR RATES (flagged as easier than typical):\n")
    f.write("These represent the fraction of EMs that score better than the median\n")
    f.write("drug-like molecule, indicating the tool rates them as easier to synthesize\n")
    f.write("than a typical pharmaceutical compound.\n\n")
    f.write("High error rates in either direction indicate poor calibration for the\n")
    f.write("energetic materials domain, directly motivating the need for a\n")
    f.write("domain-specific synthesizability model.\n")

print(f"\n✅ Detailed results saved to {output_path}")

# Also save CSVs for easy import
csv_path = OUTPUT_DIR / "misclassification_rates.csv"
df_results.to_csv(csv_path, index=False)
print(f"✅ CSV results saved to {csv_path}")

csv_opposite_path = OUTPUT_DIR / "opposite_error_rates.csv"
df_opposite.to_csv(csv_opposite_path, index=False)
print(f"✅ Opposite error rates CSV saved to {csv_opposite_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
