"""
Factor Correlation with Social vs Info Scores

Instead of binary classification (limited by small labeled set),
test whether factors correlate with text-derived social/info scores
across ALL CRH notes.

If factors capture "social utility", they should:
- Correlate positively with social_score
- Correlate negatively with info_score
- Or correlate with (social_score - info_score)

This tests whether factor space captures the social dimension
beyond what text heuristics alone provide.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import scoring functions from our classifier
from social_info_classifier import compute_info_score, compute_social_score


def load_crh_with_factors(scored_path: str, notes_path: str) -> pd.DataFrame:
    """Load CRH notes with both factors and summaries."""
    print("Loading scored notes...")
    scored = pd.read_csv(scored_path, sep='\t', low_memory=False)

    # Filter to CRH
    crh = scored[scored['finalRatingStatus'] == 'CURRENTLY_RATED_HELPFUL'].copy()
    print(f"  CRH notes: {len(crh)}")

    # Keep factor columns
    factor_cols = ['noteId', 'coreNoteIntercept', 'coreNoteFactor1',
                   'internalNoteFactor2', 'internalNoteFactor3',
                   'internalNoteFactor4', 'internalNoteFactor5', 'internalNoteFactor6']
    factor_cols = [c for c in factor_cols if c in crh.columns]
    crh = crh[factor_cols]

    # Load summaries
    print("Loading note summaries...")
    notes = pd.read_csv(notes_path, sep='\t', low_memory=False, usecols=['noteId', 'summary'])
    crh = crh.merge(notes, on='noteId', how='left')

    # Filter to notes with all factors
    has_factors = crh[['coreNoteIntercept', 'coreNoteFactor1', 'internalNoteFactor2',
                       'internalNoteFactor3', 'internalNoteFactor4', 'internalNoteFactor5',
                       'internalNoteFactor6']].notna().all(axis=1)
    crh = crh[has_factors].copy()
    print(f"  Notes with factors: {len(crh)}")

    return crh


def analyze_correlations(df: pd.DataFrame):
    """Analyze correlations between factors and social/info scores."""

    print("\nComputing text scores for all CRH notes...")
    info_df = df['summary'].apply(compute_info_score).apply(pd.Series)
    social_df = df['summary'].apply(compute_social_score).apply(pd.Series)

    df = pd.concat([df, info_df, social_df], axis=1)

    # Derived scores
    df['social_minus_info'] = df['social_score'] - df['info_score']
    df['social_info_ratio'] = df['social_score'] / (df['info_score'] + 1)

    print(f"\nScore distributions:")
    print(f"  info_score:  mean={df['info_score'].mean():.2f}, std={df['info_score'].std():.2f}")
    print(f"  social_score: mean={df['social_score'].mean():.2f}, std={df['social_score'].std():.2f}")
    print(f"  social - info: mean={df['social_minus_info'].mean():.2f}, std={df['social_minus_info'].std():.2f}")

    # Factor columns
    factor_cols = ['coreNoteIntercept', 'coreNoteFactor1',
                   'internalNoteFactor2', 'internalNoteFactor3',
                   'internalNoteFactor4', 'internalNoteFactor5', 'internalNoteFactor6']

    # Target scores to correlate with
    target_cols = ['social_score', 'info_score', 'social_minus_info', 'social_info_ratio']

    print("\n" + "="*70)
    print("CORRELATIONS: FACTORS vs TEXT-DERIVED SCORES")
    print("="*70)
    print("\nSpearman correlations (p-value in parentheses):")
    print(f"\n{'Factor':<25} {'social':>12} {'info':>12} {'soc-info':>12} {'ratio':>12}")
    print("-" * 75)

    results = {}
    for factor in factor_cols:
        results[factor] = {}
        row = f"{factor:<25}"
        for target in target_cols:
            corr, pval = spearmanr(df[factor], df[target])
            results[factor][target] = {'corr': corr, 'pval': pval}
            sig = "*" if pval < 0.05 else " "
            sig = "**" if pval < 0.01 else sig
            sig = "***" if pval < 0.001 else sig
            row += f" {corr:+.3f}{sig:>3}"
        print(row)

    print("\n*** p<0.001, ** p<0.01, * p<0.05")

    # Key finding: which factor correlates most with social_minus_info?
    print("\n" + "="*70)
    print("KEY FINDING: Which factor predicts 'social - info' best?")
    print("="*70)

    best_factor = max(factor_cols, key=lambda f: abs(results[f]['social_minus_info']['corr']))
    best_corr = results[best_factor]['social_minus_info']['corr']
    best_pval = results[best_factor]['social_minus_info']['pval']

    print(f"\nBest factor: {best_factor}")
    print(f"Correlation with (social - info): {best_corr:+.4f} (p={best_pval:.2e})")

    if abs(best_corr) > 0.1 and best_pval < 0.001:
        print("✓ Factor captures social dimension beyond text heuristics!")
    elif abs(best_corr) > 0.05:
        print("~ Factor shows weak signal for social dimension")
    else:
        print("✗ Factors don't capture social dimension beyond text")

    # Look at high social_minus_info notes
    print("\n" + "="*70)
    print("TOP 'SOCIAL' NOTES (high social_score - info_score)")
    print("="*70)

    top_social = df.nlargest(15, 'social_minus_info')
    print(f"\n{'intercept':>10} {'F6':>8} {'soc-inf':>8} summary")
    print("-" * 70)
    for _, row in top_social.iterrows():
        summary = str(row['summary'])[:45] if pd.notna(row['summary']) else 'NaN'
        print(f"{row['coreNoteIntercept']:>10.3f} {row['internalNoteFactor6']:>8.3f} "
              f"{row['social_minus_info']:>8.1f} {summary}...")

    # Compare to our hand-labeled humor notes
    print("\n" + "="*70)
    print("OUR HAND-LABELED HUMOR CRH NOTES (that have factors)")
    print("="*70)

    humor_ids = [
        1654229640754372613, 1692660341212070388, 1694581725014233312,
        1841513716564566398, 1858895923184558154, 1868165050403615097,
        1873504456752447866, 1881457464522281075, 1957506012820414604
    ]

    humor_notes = df[df['noteId'].isin(humor_ids)]
    if len(humor_notes) > 0:
        print(f"\n{'intercept':>10} {'F6':>8} {'soc-inf':>8} summary")
        print("-" * 70)
        for _, row in humor_notes.iterrows():
            summary = str(row['summary'])[:45] if pd.notna(row['summary']) else 'NaN'
            print(f"{row['coreNoteIntercept']:>10.3f} {row['internalNoteFactor6']:>8.3f} "
                  f"{row['social_minus_info']:>8.1f} {summary}...")

        print(f"\nHumor notes mean (social - info): {humor_notes['social_minus_info'].mean():.2f}")
        print(f"All CRH notes mean (social - info): {df['social_minus_info'].mean():.2f}")

    return results, df


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent / "data"

    df = load_crh_with_factors(
        str(base_path / "scored_notes.tsv"),
        str(base_path / "notes-00000.tsv")
    )

    results, df_with_scores = analyze_correlations(df)
