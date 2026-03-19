#!/usr/bin/env python3
"""
Fixed Temporal Stability Test

The previous test was flawed because it trained independent models for each
time period. This version uses PRODUCTION factors and checks if user behavior
is consistent with their factor positions across time.

Approach:
1. Use production user/note factors (already trained on all data)
2. Split ratings by time into early/late periods
3. For each user, compute: does their factor position predict their ratings
   consistently in both early and late periods?

Usage:
    python temporal_stability_fixed.py \
        --notes data/scored_notes.tsv \
        --users data/helpfulness_scores.tsv \
        --ratings data/ratings/ \
        --outdir data/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    createdAtMillisKey,
    helpfulnessLevelKey,
)

HELPFULNESS_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


def find_factor_columns(df: pd.DataFrame, entity_type: str, num_factors: int = 6) -> List[str]:
    """Find factor columns."""
    factor_cols = []
    if entity_type == 'note':
        prefix_internal = 'internalNoteFactor'
        prefix_core = 'coreNoteFactor'
    else:
        prefix_internal = 'internalRaterFactor'
        prefix_core = 'coreRaterFactor'

    for i in range(1, num_factors + 1):
        candidates = [f"{prefix_internal}{i}", f"{prefix_core}{i}"]
        for col in candidates:
            if col in df.columns:
                factor_cols.append(col)
                break
    return factor_cols


def run_temporal_stability_fixed(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
) -> Dict:
    """
    Test temporal stability using production factors.

    Key insight: If factors capture stable user preferences, then:
    - A user's factor alignment with a note should predict their rating
    - This prediction should work equally well in early AND late periods

    We measure: correlation between (user_factor · note_factor) and rating,
    separately for early and late periods.
    """
    print("\n" + "=" * 60)
    print("TEMPORAL STABILITY (Fixed - Using Production Factors)")
    print("=" * 60)

    # Prepare ratings
    if helpfulnessLevelKey in ratings.columns:
        ratings = ratings.copy()
        ratings['helpful'] = ratings[helpfulnessLevelKey].map(HELPFULNESS_MAP)

    if createdAtMillisKey not in ratings.columns:
        return {'error': 'No timestamp column'}

    # Merge ratings with factors
    print("Merging ratings with production factors...")

    # Get note factors
    note_cols = [noteIdKey] + note_factor_cols
    note_factors = notes[note_cols].dropna()

    # Get user factors
    user_cols = [raterParticipantIdKey] + user_factor_cols
    user_factors = users[user_cols].dropna()

    # Merge
    merged = ratings.merge(note_factors, on=noteIdKey, how='inner')
    merged = merged.merge(user_factors, on=raterParticipantIdKey, how='inner')
    merged = merged.dropna(subset=['helpful', createdAtMillisKey])

    print(f"Ratings with factors: {len(merged):,}")

    # Split by time (median split)
    median_time = merged[createdAtMillisKey].median()
    early = merged[merged[createdAtMillisKey] < median_time]
    late = merged[merged[createdAtMillisKey] >= median_time]

    print(f"Early period: {len(early):,} ratings")
    print(f"Late period: {len(late):,} ratings")

    # Compute factor dot product for each rating
    def compute_dot_product(df, note_cols, user_cols):
        dot = np.zeros(len(df))
        for nc, uc in zip(note_cols, user_cols):
            dot += df[nc].values * df[uc].values
        return dot

    early['factor_dot'] = compute_dot_product(early, note_factor_cols, user_factor_cols)
    late['factor_dot'] = compute_dot_product(late, note_factor_cols, user_factor_cols)

    # Correlation between factor_dot and helpful rating
    early_corr, early_p = spearmanr(early['factor_dot'], early['helpful'])
    late_corr, late_p = spearmanr(late['factor_dot'], late['helpful'])

    print(f"\nFactor-rating correlation:")
    print(f"  Early period: r={early_corr:.4f} (p={early_p:.2e})")
    print(f"  Late period:  r={late_corr:.4f} (p={late_p:.2e})")

    # Check per-user stability
    # For users with ratings in both periods, compare their behavior
    print("\nPer-user stability analysis...")

    early_users = set(early[raterParticipantIdKey].unique())
    late_users = set(late[raterParticipantIdKey].unique())
    common_users = early_users & late_users

    print(f"Users in both periods: {len(common_users):,}")

    # For each common user, compute their mean residual in each period
    # Residual = rating - predicted (where predicted = global_mean + factor_dot * weight)

    global_mean = merged['helpful'].mean()

    user_early_means = early.groupby(raterParticipantIdKey)['helpful'].mean()
    user_late_means = late.groupby(raterParticipantIdKey)['helpful'].mean()

    # Get common users' means
    common_early = user_early_means.loc[user_early_means.index.isin(common_users)]
    common_late = user_late_means.loc[user_late_means.index.isin(common_users)]

    # Align indices
    common_idx = common_early.index.intersection(common_late.index)
    common_early = common_early.loc[common_idx]
    common_late = common_late.loc[common_idx]

    # Correlation of user means across periods
    user_stability_corr, user_stability_p = spearmanr(common_early, common_late)

    print(f"\nUser mean rating stability:")
    print(f"  Correlation (early vs late): r={user_stability_corr:.4f}")

    # Also check: do user FACTORS predict their behavior equally well in both periods?
    # Group by user and compute correlation between their factor position and rating tendency

    def user_factor_consistency(df, user_factor_cols):
        """For each user, how consistent is their factor with their ratings?"""
        user_stats = []
        for user_id, group in df.groupby(raterParticipantIdKey):
            if len(group) < 5:
                continue
            # Get user's factor (should be same for all their ratings)
            user_f = group[user_factor_cols].iloc[0].values
            user_f_norm = np.linalg.norm(user_f)
            # Mean rating
            mean_rating = group['helpful'].mean()
            user_stats.append({
                'user': user_id,
                'f_norm': user_f_norm,
                'mean_rating': mean_rating,
            })
        return pd.DataFrame(user_stats)

    early_user_stats = user_factor_consistency(early, user_factor_cols)
    late_user_stats = user_factor_consistency(late, user_factor_cols)

    # Merge to get common users
    common_stats = early_user_stats.merge(
        late_user_stats, on='user', suffixes=('_early', '_late')
    )

    if len(common_stats) > 100:
        # Correlation of factor norms
        fnorm_corr = np.corrcoef(common_stats['f_norm_early'], common_stats['f_norm_late'])[0, 1]
        print(f"  Factor norm correlation: r={fnorm_corr:.4f}")

        # Correlation of rating tendency
        rating_corr = np.corrcoef(common_stats['mean_rating_early'], common_stats['mean_rating_late'])[0, 1]
        print(f"  Rating tendency correlation: r={rating_corr:.4f}")
    else:
        fnorm_corr = None
        rating_corr = None

    # Summary
    results = {
        'n_ratings_total': len(merged),
        'n_early': len(early),
        'n_late': len(late),
        'n_common_users': len(common_users),
        'factor_rating_correlation': {
            'early': float(early_corr),
            'late': float(late_corr),
            'description': 'Correlation between factor dot product and rating',
        },
        'user_stability': {
            'mean_rating_correlation': float(user_stability_corr),
            'factor_norm_correlation': float(fnorm_corr) if fnorm_corr else None,
            'rating_tendency_correlation': float(rating_corr) if rating_corr else None,
        },
        'interpretation': (
            'Factor-rating correlation is stable across time periods if '
            'early_corr ≈ late_corr. User stability measures whether individual '
            'users behave consistently.'
        ),
        'pass': abs(early_corr - late_corr) < 0.1 and user_stability_corr > 0.3,
    }

    print(f"\n{'='*60}")
    print("RESULT:", "PASS" if results['pass'] else "NEEDS REVIEW")
    print(f"  Factor-rating corr diff: {abs(early_corr - late_corr):.4f} (threshold: <0.1)")
    print(f"  User stability: {user_stability_corr:.4f} (threshold: >0.3)")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes', required=True)
    parser.add_argument('--users', required=True)
    parser.add_argument('--ratings', required=True)
    parser.add_argument('--outdir', default='data/')
    args = parser.parse_args()

    print("Loading data...")
    notes = pd.read_csv(args.notes, sep='\t', low_memory=False)
    users = pd.read_csv(args.users, sep='\t', low_memory=False)

    ratings_path = Path(args.ratings)
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path, sep='\t', low_memory=False)
    else:
        rating_files = sorted(ratings_path.glob("ratings*.tsv"))
        ratings = pd.read_csv(rating_files[0], sep='\t', low_memory=False)

    print(f"  Notes: {len(notes):,}, Users: {len(users):,}, Ratings: {len(ratings):,}")

    note_factor_cols = find_factor_columns(notes, 'note')
    user_factor_cols = find_factor_columns(users, 'user')

    results = run_temporal_stability_fixed(
        notes, users, ratings, note_factor_cols, user_factor_cols
    )

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    outdir = Path(args.outdir)
    outpath = outdir / 'temporal_stability_fixed.json'
    with open(outpath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to: {outpath}")


if __name__ == "__main__":
    main()
