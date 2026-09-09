"""
generate_consensus_dataset.py

Description:
------------
Part 2 of annotator agreement analysis:
- Identifies structural regions of greatest disagreement
- Generates senior consensus scores
- Creates quality-tiered labeled dataset

Requires: annotator_agreement_analysis.py to be run first (or run as continuation)
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
OUTPUT_DIR = Path("data/output").resolve()

print("\n" + "="*80)
print("STEP 5: GENERATING SENIOR CONSENSUS SCORES")
print("="*80)

# Load data
df = pd.read_csv(INPUT_PATH)

# Define senior annotators
SENIOR_ANNOTATORS = ['asaf', 'elena', 'levi']

def extract_senior_consensus(row):
    """Extract consensus scores and choice from senior annotators"""

    # Extract choices and scores for each senior
    senior_choices = []
    scores_a = []
    scores_b = []

    for annotator in SENIOR_ANNOTATORS:
        choice_col = f"{annotator} - Choice"
        score_a_col = f"{annotator} - Score A"
        score_b_col = f"{annotator} - Score B"

        choice = row.get(choice_col)
        score_a = row.get(score_a_col)
        score_b = row.get(score_b_col)

        # Handle missing/empty data
        if pd.isna(choice) or choice == '':
            choice = None
        if pd.isna(score_a) or score_a == '':
            score_a = None
        else:
            score_a = float(score_a)
        if pd.isna(score_b) or score_b == '':
            score_b = None
        else:
            score_b = float(score_b)

        # Store choice (excluding UNSURE for vote counting)
        if choice and choice != 'UNSURE':
            senior_choices.append(choice)

        # Store scores (including UNSURE cases)
        if score_a is not None:
            scores_a.append(score_a)
        if score_b is not None:
            scores_b.append(score_b)

    # Calculate consensus score (mean of all senior scores)
    consensus_score_a = np.mean(scores_a) if len(scores_a) > 0 else np.nan
    consensus_score_b = np.mean(scores_b) if len(scores_b) > 0 else np.nan
    std_score_a = np.std(scores_a) if len(scores_a) > 1 else np.nan
    std_score_b = np.std(scores_b) if len(scores_b) > 1 else np.nan

    # Count votes
    vote_a = senior_choices.count('A')
    vote_b = senior_choices.count('B')
    n_seniors_rated = len([s for s in scores_a if s is not None])

    # Determine consensus choice based on senior majority
    if vote_a >= 2:
        consensus_choice = 'A'
        confidence = 'strong' if vote_a == 3 else 'moderate'
    elif vote_b >= 2:
        consensus_choice = 'B'
        confidence = 'strong' if vote_b == 3 else 'moderate'
    else:
        # No clear majority (1-1-1, 0-0-3, etc.)
        consensus_choice = 'no_consensus'
        confidence = 'noise'

    # Calculate score separation
    score_separation = abs(consensus_score_a - consensus_score_b) if not np.isnan(consensus_score_a) and not np.isnan(consensus_score_b) else np.nan

    # Combined variance (average of molecule A and B variances)
    combined_variance = np.mean([v for v in [std_score_a, std_score_b] if not np.isnan(v)]) if not all(np.isnan([std_score_a, std_score_b])) else np.nan

    return {
        'consensus_score_a': consensus_score_a,
        'consensus_score_b': consensus_score_b,
        'std_score_a': std_score_a,
        'std_score_b': std_score_b,
        'combined_std': combined_variance,
        'score_separation': score_separation,
        'vote_a': vote_a,
        'vote_b': vote_b,
        'n_seniors_rated': n_seniors_rated,
        'consensus_choice': consensus_choice,
        'confidence': confidence
    }

# Generate consensus for all pairs
consensus_data = []
for idx, row in df.iterrows():
    consensus = extract_senior_consensus(row)
    consensus['pair_id'] = row['Pair ID']
    consensus['smiles_a'] = row['Molecule A SMILES']
    consensus['smiles_b'] = row['Molecule B SMILES']
    consensus_data.append(consensus)

df_consensus = pd.DataFrame(consensus_data)

print(f"\nGenerated consensus scores for {len(df_consensus)} pairs")
print(f"  Mean score A: {df_consensus['consensus_score_a'].mean():.2f} ± {df_consensus['consensus_score_a'].std():.2f}")
print(f"  Mean score B: {df_consensus['consensus_score_b'].mean():.2f} ± {df_consensus['consensus_score_b'].std():.2f}")
print(f"  Mean separation: {df_consensus['score_separation'].mean():.2f} ± {df_consensus['score_separation'].std():.2f}")
print(f"  Mean combined std: {df_consensus['combined_std'].mean():.2f}")

# ==============================================================================
# STEP 6: Quality Tier Classification
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: QUALITY TIER CLASSIFICATION")
print("="*80)

def assign_quality_tier(row):
    """
    Assign quality tier based on:
    - Vote agreement (strong 3-0, moderate 2-1, weak/noise)
    - Score separation (clear vs ambiguous)
    - Combined variance (low vs high disagreement)
    """

    vote_a = row['vote_a']
    vote_b = row['vote_b']
    separation = row['score_separation']
    combined_std = row['combined_std']
    consensus_choice = row['consensus_choice']

    # Already classified as no_consensus
    if consensus_choice == 'no_consensus':
        return 'tier_4_noise'

    # Missing critical data
    if np.isnan(separation) or np.isnan(combined_std):
        return 'tier_4_noise'

    # Tier 1: High Confidence (Strong Signal)
    # - Unanimous or one abstains (3-0 or 2-0)
    # - Clear separation (>1.0)
    # - Low variance (<1.0)
    if (vote_a == 3 or vote_b == 3 or (vote_a == 2 and vote_b == 0) or (vote_b == 2 and vote_a == 0)):
        if separation > 1.0 and combined_std < 1.0:
            return 'tier_1_high'
        elif separation > 0.5 and combined_std < 1.5:
            return 'tier_2_moderate'

    # Tier 2: Moderate Confidence (Usable Signal)
    # - Clear majority (2-1)
    # - Reasonable separation (>0.5)
    # - Moderate variance (<1.5)
    if (vote_a == 2 and vote_b == 1) or (vote_b == 2 and vote_a == 1):
        if separation > 0.5 and combined_std < 1.5:
            return 'tier_2_moderate'
        elif separation > 0.3 and combined_std < 2.0:
            return 'tier_3_low'

    # Tier 3: Low Confidence (Questionable)
    # - Weak signals: small separation or high variance
    if separation > 0.3 and combined_std < 2.0:
        return 'tier_3_low'

    # Tier 4: Noise (Exclude from Training)
    return 'tier_4_noise'

df_consensus['quality_tier'] = df_consensus.apply(assign_quality_tier, axis=1)

# Print tier distribution
print("\nQuality Tier Distribution:")
print("-" * 60)
tier_counts = df_consensus['quality_tier'].value_counts().sort_index()
for tier, count in tier_counts.items():
    pct = 100 * count / len(df_consensus)
    print(f"  {tier:20s}: {count:4d} pairs ({pct:5.1f}%)")

# Print tier definitions
print("\n" + "-" * 60)
print("Tier Definitions:")
print("-" * 60)
print("  Tier 1 (High):     3-0 or 2-0 vote, separation >1.0, std <1.0")
print("  Tier 2 (Moderate): 2-1 vote, separation >0.5, std <1.5")
print("  Tier 3 (Low):      Weak signals, separation >0.3, std <2.0")
print("  Tier 4 (Noise):    No majority, small separation, or high variance")

# ==============================================================================
# STEP 7: Save Consensus Dataset
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING CONSENSUS DATASET")
print("="*80)

# Reorganize columns for clean output
consensus_output = df_consensus[[
    'pair_id',
    'smiles_a',
    'smiles_b',
    'consensus_score_a',
    'consensus_score_b',
    'score_separation',
    'std_score_a',
    'std_score_b',
    'combined_std',
    'consensus_choice',
    'vote_a',
    'vote_b',
    'n_seniors_rated',
    'confidence',
    'quality_tier'
]].copy()

# Save full dataset
full_output_path = OUTPUT_DIR / "senior_consensus_dataset.csv"
consensus_output.to_csv(full_output_path, index=False)
print(f"✅ Saved full consensus dataset to {full_output_path}")

# Save high-quality subset (Tier 1 + 2)
high_quality = consensus_output[consensus_output['quality_tier'].isin(['tier_1_high', 'tier_2_moderate'])]
hq_output_path = OUTPUT_DIR / "senior_consensus_high_quality.csv"
high_quality.to_csv(hq_output_path, index=False)
print(f"✅ Saved high-quality subset to {hq_output_path}")
print(f"   ({len(high_quality)} pairs, {100*len(high_quality)/len(consensus_output):.1f}% of total)")

# Save Tier 1 only (highest quality)
tier1_only = consensus_output[consensus_output['quality_tier'] == 'tier_1_high']
tier1_output_path = OUTPUT_DIR / "senior_consensus_tier1_only.csv"
tier1_only.to_csv(tier1_output_path, index=False)
print(f"✅ Saved Tier 1 only to {tier1_output_path}")
print(f"   ({len(tier1_only)} pairs, {100*len(tier1_only)/len(consensus_output):.1f}% of total)")

# Create summary statistics file
summary_path = OUTPUT_DIR / "consensus_statistics.txt"
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SENIOR CONSENSUS DATASET STATISTICS\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total pairs: {len(consensus_output)}\n")
    f.write(f"Senior annotators: Asaf, Elena, Levi\n\n")

    f.write("QUALITY TIER DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    for tier, count in tier_counts.items():
        pct = 100 * count / len(df_consensus)
        f.write(f"{tier:20s}: {count:4d} pairs ({pct:5.1f}%)\n")

    f.write("\n" + "CONSENSUS SCORE STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean consensus score A: {df_consensus['consensus_score_a'].mean():.2f} ± {df_consensus['consensus_score_a'].std():.2f}\n")
    f.write(f"Mean consensus score B: {df_consensus['consensus_score_b'].mean():.2f} ± {df_consensus['consensus_score_b'].std():.2f}\n")
    f.write(f"Mean score separation: {df_consensus['score_separation'].mean():.2f} ± {df_consensus['score_separation'].std():.2f}\n")
    f.write(f"Mean combined std: {df_consensus['combined_std'].mean():.2f}\n\n")

    f.write("CONSENSUS CHOICE DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    choice_counts = df_consensus['consensus_choice'].value_counts()
    for choice, count in choice_counts.items():
        pct = 100 * count / len(df_consensus)
        f.write(f"{choice:15s}: {count:4d} ({pct:5.1f}%)\n")

    f.write("\n" + "QUALITY RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("Conservative (Tier 1 only): Use for highest quality training\n")
    f.write(f"  → {len(tier1_only)} pairs ({100*len(tier1_only)/len(consensus_output):.1f}%)\n\n")
    f.write("Moderate (Tier 1 + 2): Balance quality and quantity\n")
    f.write(f"  → {len(high_quality)} pairs ({100*len(high_quality)/len(consensus_output):.1f}%)\n\n")
    f.write("Liberal (Tier 1 + 2 + 3): Maximum data retention\n")
    liberal = consensus_output[consensus_output['quality_tier'].isin(['tier_1_high', 'tier_2_moderate', 'tier_3_low'])]
    f.write(f"  → {len(liberal)} pairs ({100*len(liberal)/len(consensus_output):.1f}%)\n")

print(f"✅ Saved summary statistics to {summary_path}")

print("\n" + "="*80)
print("CONSENSUS GENERATION COMPLETE")
print("="*80)
