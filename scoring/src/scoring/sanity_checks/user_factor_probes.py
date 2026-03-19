#!/usr/bin/env python3
"""
User Factor Probes: Discover what each factor axis represents by analyzing
what users at the extremes rate as helpful.

For each factor:
1. Find users at top/bottom 2%
2. Analyze what notes they rate HELPFUL vs NOT_HELPFUL
3. Find "signature" notes where top/bottom users strongly disagree
4. Identify topic preferences, classification patterns

This helps discover if an axis represents:
- Left/right politics
- Snarky vs earnest tone
- Institution-trusting vs skeptical
- Humor tolerance
- Domain expertise (e.g., medical misinfo)

Usage:
    python user_factor_probes.py \
        --notes data/notes-00000.tsv \
        --users data/helpfulness_scores.tsv \
        --ratings data/ratings-00000.tsv
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    summaryKey,
    classificationKey,
    helpfulnessLevelKey,
    noteTopicKey,
)


HELPFULNESS_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


def find_user_factor_columns(df: pd.DataFrame, num_factors: int = 6) -> List[str]:
    """Find user factor columns with mixed naming convention."""
    factor_cols = []
    for i in range(1, num_factors + 1):
        candidates = [f"internalRaterFactor{i}", f"coreRaterFactor{i}"]
        for col in candidates:
            if col in df.columns:
                factor_cols.append(col)
                break
    return factor_cols


def get_extreme_users(
    users: pd.DataFrame,
    factor_col: str,
    percentile: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Get users at extreme ends of a factor axis."""
    valid = users[users[factor_col].notna()].copy()

    low_threshold = np.percentile(valid[factor_col], percentile)
    high_threshold = np.percentile(valid[factor_col], 100 - percentile)

    low_users = valid[valid[factor_col] <= low_threshold]
    high_users = valid[valid[factor_col] >= high_threshold]

    return low_users, high_users, low_threshold, high_threshold


def get_user_ratings(
    user_ids: Set,
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
) -> pd.DataFrame:
    """Get ratings from a set of users, merged with note info."""
    user_ratings = ratings[ratings[raterParticipantIdKey].isin(user_ids)].copy()

    # Convert helpfulness to numeric
    if helpfulnessLevelKey in user_ratings.columns:
        user_ratings['helpful_num'] = user_ratings[helpfulnessLevelKey].map(HELPFULNESS_MAP)

    # Merge with notes for text and metadata
    merge_cols = [noteIdKey]
    if summaryKey in notes.columns:
        merge_cols.append(summaryKey)
    if classificationKey in notes.columns:
        merge_cols.append(classificationKey)
    if noteTopicKey in notes.columns:
        merge_cols.append(noteTopicKey)

    user_ratings = user_ratings.merge(
        notes[merge_cols],
        on=noteIdKey,
        how='left'
    )

    return user_ratings


def analyze_rating_patterns(
    ratings: pd.DataFrame,
    label: str,
    n_examples: int = 8,
) -> Dict:
    """Analyze rating patterns for a group of users."""
    analysis = {
        'label': label,
        'n_ratings': len(ratings),
        'mean_helpful': ratings['helpful_num'].mean() if 'helpful_num' in ratings.columns else None,
    }

    if 'helpful_num' not in ratings.columns:
        return analysis

    helpful = ratings[ratings['helpful_num'] >= 0.5]
    not_helpful = ratings[ratings['helpful_num'] == 0]

    analysis['n_helpful'] = len(helpful)
    analysis['n_not_helpful'] = len(not_helpful)
    analysis['helpful_rate'] = len(helpful) / len(ratings) if len(ratings) > 0 else 0

    # Topic breakdown
    if noteTopicKey in ratings.columns:
        analysis['helpful_topics'] = helpful[noteTopicKey].value_counts().head(10).to_dict()
        analysis['not_helpful_topics'] = not_helpful[noteTopicKey].value_counts().head(10).to_dict()

    # Classification breakdown
    if classificationKey in ratings.columns:
        analysis['helpful_classifications'] = helpful[classificationKey].value_counts().to_dict()
        analysis['not_helpful_classifications'] = not_helpful[classificationKey].value_counts().to_dict()

    # Sample examples of notes they rate HELPFUL
    analysis['helpful_examples'] = []
    if len(helpful) > 0 and summaryKey in helpful.columns:
        sample = helpful.sample(n=min(n_examples, len(helpful)), random_state=42)
        for _, row in sample.iterrows():
            analysis['helpful_examples'].append({
                'noteId': row[noteIdKey],
                'text': str(row.get(summaryKey, ''))[:350],
                'classification': row.get(classificationKey, ''),
                'topic': row.get(noteTopicKey, ''),
            })

    # Sample examples of notes they rate NOT HELPFUL
    analysis['not_helpful_examples'] = []
    if len(not_helpful) > 0 and summaryKey in not_helpful.columns:
        sample = not_helpful.sample(n=min(n_examples, len(not_helpful)), random_state=42)
        for _, row in sample.iterrows():
            analysis['not_helpful_examples'].append({
                'noteId': row[noteIdKey],
                'text': str(row.get(summaryKey, ''))[:350],
                'classification': row.get(classificationKey, ''),
                'topic': row.get(noteTopicKey, ''),
            })

    return analysis


def find_signature_notes(
    low_ratings: pd.DataFrame,
    high_ratings: pd.DataFrame,
    notes: pd.DataFrame,
    n_examples: int = 10,
) -> Dict:
    """
    Find notes where LOW and HIGH users strongly disagree.

    "Signature" notes are those that one group rates helpful
    while the other group rates not helpful.
    """

    # Aggregate ratings by note
    low_by_note = low_ratings.groupby(noteIdKey)['helpful_num'].agg(['mean', 'count']).reset_index()
    low_by_note.columns = [noteIdKey, 'low_mean', 'low_count']

    high_by_note = high_ratings.groupby(noteIdKey)['helpful_num'].agg(['mean', 'count']).reset_index()
    high_by_note.columns = [noteIdKey, 'high_mean', 'high_count']

    # Merge to find notes rated by both groups
    merged = low_by_note.merge(high_by_note, on=noteIdKey, how='inner')

    # Require at least 3 ratings from each group
    merged = merged[(merged['low_count'] >= 3) & (merged['high_count'] >= 3)]

    # Compute disagreement score
    merged['disagreement'] = merged['high_mean'] - merged['low_mean']

    # Get note text
    merge_cols = [noteIdKey]
    if summaryKey in notes.columns:
        merge_cols.append(summaryKey)
    if classificationKey in notes.columns:
        merge_cols.append(classificationKey)
    if noteTopicKey in notes.columns:
        merge_cols.append(noteTopicKey)

    merged = merged.merge(notes[merge_cols], on=noteIdKey, how='left')

    result = {
        'n_shared_notes': len(merged),
    }

    # Notes that HIGH users like but LOW users don't
    high_prefers = merged.nlargest(n_examples, 'disagreement')
    result['high_prefers'] = []
    for _, row in high_prefers.iterrows():
        result['high_prefers'].append({
            'noteId': row[noteIdKey],
            'low_mean': row['low_mean'],
            'high_mean': row['high_mean'],
            'disagreement': row['disagreement'],
            'text': str(row.get(summaryKey, ''))[:400],
            'classification': row.get(classificationKey, ''),
            'topic': row.get(noteTopicKey, ''),
        })

    # Notes that LOW users like but HIGH users don't
    low_prefers = merged.nsmallest(n_examples, 'disagreement')
    result['low_prefers'] = []
    for _, row in low_prefers.iterrows():
        result['low_prefers'].append({
            'noteId': row[noteIdKey],
            'low_mean': row['low_mean'],
            'high_mean': row['high_mean'],
            'disagreement': row['disagreement'],
            'text': str(row.get(summaryKey, ''))[:400],
            'classification': row.get(classificationKey, ''),
            'topic': row.get(noteTopicKey, ''),
        })

    return result


def print_factor_analysis(
    factor_idx: int,
    factor_col: str,
    low_threshold: float,
    high_threshold: float,
    low_analysis: Dict,
    high_analysis: Dict,
    signature_notes: Dict,
):
    """Print comprehensive analysis for a factor."""

    print(f"\n{'='*100}")
    print(f"FACTOR {factor_idx}: {factor_col}")
    print(f"{'='*100}")
    print(f"LOW threshold: {low_threshold:.3f}, HIGH threshold: {high_threshold:.3f}")

    # ===== LOW END USERS =====
    print(f"\n{'-'*100}")
    print(f"LOW END USERS (factor <= {low_threshold:.3f})")
    print(f"{'-'*100}")
    print(f"Ratings: {low_analysis['n_ratings']:,}")
    print(f"Helpful rate: {low_analysis['helpful_rate']:.1%}")

    if low_analysis.get('helpful_topics'):
        print(f"\n  Topics they rate HELPFUL:")
        for topic, count in list(low_analysis['helpful_topics'].items())[:7]:
            print(f"    {topic}: {count}")

    if low_analysis.get('not_helpful_topics'):
        print(f"\n  Topics they rate NOT HELPFUL:")
        for topic, count in list(low_analysis['not_helpful_topics'].items())[:7]:
            print(f"    {topic}: {count}")

    if low_analysis.get('helpful_classifications'):
        print(f"\n  Classifications they rate HELPFUL:")
        for cls, count in low_analysis['helpful_classifications'].items():
            print(f"    {cls}: {count}")

    if low_analysis.get('helpful_examples'):
        print(f"\n  Example notes LOW users rate HELPFUL:")
        for i, ex in enumerate(low_analysis['helpful_examples'][:4], 1):
            cls = ex.get('classification', '')
            cls = str(cls)[:40] if cls and not (isinstance(cls, float) and np.isnan(cls)) else ''
            topic = ex.get('topic', '')
            topic = str(topic) if topic and not (isinstance(topic, float) and np.isnan(topic)) else ''
            print(f"\n    [{i}] {cls} | {topic}")
            print(f"        {ex['text'][:250]}...")

    if low_analysis.get('not_helpful_examples'):
        print(f"\n  Example notes LOW users rate NOT HELPFUL:")
        for i, ex in enumerate(low_analysis['not_helpful_examples'][:4], 1):
            cls = ex.get('classification', '')
            cls = str(cls)[:40] if cls and not (isinstance(cls, float) and np.isnan(cls)) else ''
            topic = ex.get('topic', '')
            topic = str(topic) if topic and not (isinstance(topic, float) and np.isnan(topic)) else ''
            print(f"\n    [{i}] {cls} | {topic}")
            print(f"        {ex['text'][:250]}...")

    # ===== HIGH END USERS =====
    print(f"\n{'-'*100}")
    print(f"HIGH END USERS (factor >= {high_threshold:.3f})")
    print(f"{'-'*100}")
    print(f"Ratings: {high_analysis['n_ratings']:,}")
    print(f"Helpful rate: {high_analysis['helpful_rate']:.1%}")

    if high_analysis.get('helpful_topics'):
        print(f"\n  Topics they rate HELPFUL:")
        for topic, count in list(high_analysis['helpful_topics'].items())[:7]:
            print(f"    {topic}: {count}")

    if high_analysis.get('not_helpful_topics'):
        print(f"\n  Topics they rate NOT HELPFUL:")
        for topic, count in list(high_analysis['not_helpful_topics'].items())[:7]:
            print(f"    {topic}: {count}")

    if high_analysis.get('helpful_classifications'):
        print(f"\n  Classifications they rate HELPFUL:")
        for cls, count in high_analysis['helpful_classifications'].items():
            print(f"    {cls}: {count}")

    if high_analysis.get('helpful_examples'):
        print(f"\n  Example notes HIGH users rate HELPFUL:")
        for i, ex in enumerate(high_analysis['helpful_examples'][:4], 1):
            cls = ex.get('classification', '')
            cls = str(cls)[:40] if cls and not (isinstance(cls, float) and np.isnan(cls)) else ''
            topic = ex.get('topic', '')
            topic = str(topic) if topic and not (isinstance(topic, float) and np.isnan(topic)) else ''
            print(f"\n    [{i}] {cls} | {topic}")
            print(f"        {ex['text'][:250]}...")

    if high_analysis.get('not_helpful_examples'):
        print(f"\n  Example notes HIGH users rate NOT HELPFUL:")
        for i, ex in enumerate(high_analysis['not_helpful_examples'][:4], 1):
            cls = ex.get('classification', '')
            cls = str(cls)[:40] if cls and not (isinstance(cls, float) and np.isnan(cls)) else ''
            topic = ex.get('topic', '')
            topic = str(topic) if topic and not (isinstance(topic, float) and np.isnan(topic)) else ''
            print(f"\n    [{i}] {cls} | {topic}")
            print(f"        {ex['text'][:250]}...")

    # ===== SIGNATURE NOTES =====
    print(f"\n{'='*100}")
    print(f"SIGNATURE NOTES (where LOW and HIGH users disagree)")
    print(f"{'='*100}")
    print(f"Notes rated by both groups: {signature_notes['n_shared_notes']:,}")

    print(f"\n>>> Notes that HIGH users like but LOW users don't:")
    print(f"    (These reveal what HIGH factor values mean)")
    for i, note in enumerate(signature_notes['high_prefers'][:5], 1):
        print(f"\n  [{i}] LOW={note['low_mean']:.2f}, HIGH={note['high_mean']:.2f} (Δ={note['disagreement']:.2f})")
        print(f"      Topic: {note['topic']} | Class: {note['classification']}")
        print(f"      {note['text'][:300]}...")

    print(f"\n>>> Notes that LOW users like but HIGH users don't:")
    print(f"    (These reveal what LOW factor values mean)")
    for i, note in enumerate(signature_notes['low_prefers'][:5], 1):
        print(f"\n  [{i}] LOW={note['low_mean']:.2f}, HIGH={note['high_mean']:.2f} (Δ={note['disagreement']:.2f})")
        print(f"      Topic: {note['topic']} | Class: {note['classification']}")
        print(f"      {note['text'][:300]}...")

    # ===== INTERPRETATION HINTS =====
    print(f"\n{'='*100}")
    print(f"INTERPRETATION HINTS")
    print(f"{'='*100}")

    # Compare helpful rates
    rate_diff = high_analysis['helpful_rate'] - low_analysis['helpful_rate']
    if abs(rate_diff) > 0.05:
        if rate_diff > 0:
            print(f"  • HIGH users are more generous raters (+{rate_diff:.1%} helpful rate)")
        else:
            print(f"  • LOW users are more generous raters ({rate_diff:.1%} helpful rate)")

    # Compare topic preferences
    if low_analysis.get('helpful_topics') and high_analysis.get('helpful_topics'):
        low_topics = set(low_analysis['helpful_topics'].keys())
        high_topics = set(high_analysis['helpful_topics'].keys())

        only_low = low_topics - high_topics
        only_high = high_topics - low_topics

        if only_low:
            print(f"  • Topics preferred by LOW users only: {', '.join(list(only_low)[:3])}")
        if only_high:
            print(f"  • Topics preferred by HIGH users only: {', '.join(list(only_high)[:3])}")

    # Look at signature note patterns
    if signature_notes['high_prefers']:
        high_topics = [n['topic'] for n in signature_notes['high_prefers'] if n['topic']]
        if high_topics:
            top_topic = Counter(high_topics).most_common(1)[0][0]
            print(f"  • HIGH users' signature notes often about: {top_topic}")

    if signature_notes['low_prefers']:
        low_topics = [n['topic'] for n in signature_notes['low_prefers'] if n['topic']]
        if low_topics:
            top_topic = Counter(low_topics).most_common(1)[0][0]
            print(f"  • LOW users' signature notes often about: {top_topic}")

    print(f"\n  SUGGESTED AXIS LABELS (manual review needed):")
    print(f"    - Political: left vs right")
    print(f"    - Trust: institutional vs skeptical")
    print(f"    - Tone: snarky vs earnest")
    print(f"    - Domain: specific topic expertise")
    print(f"    - Quality: strict vs lenient rating standards")


def run_user_probes(
    notes_path: str,
    users_path: str,
    ratings_path: str,
    factors_to_probe: Optional[List[int]] = None,
    percentile: float = 2.0,
):
    """Run user factor probes."""

    print("Loading data...")

    # Load notes
    notes = pd.read_csv(notes_path, sep='\t', low_memory=False)
    print(f"  Notes: {len(notes):,}")

    # Load users
    users = pd.read_csv(users_path, sep='\t')
    print(f"  Users: {len(users):,}")

    # Find factor columns
    factor_cols = find_user_factor_columns(users)
    print(f"  Found {len(factor_cols)} user factors: {factor_cols}")

    # Check how many users have factor values
    if factor_cols:
        users_with_factors = users[users[factor_cols[0]].notna()]
        print(f"  Users with factor values: {len(users_with_factors):,} ({100*len(users_with_factors)/len(users):.1f}%)")

    # Collect all extreme user IDs across all factors first
    all_extreme_user_ids = set()
    for factor_col in factor_cols:
        valid = users[users[factor_col].notna()]
        if len(valid) == 0:
            continue
        low_thresh = np.percentile(valid[factor_col], percentile)
        high_thresh = np.percentile(valid[factor_col], 100 - percentile)
        low_ids = set(valid[valid[factor_col] <= low_thresh][raterParticipantIdKey])
        high_ids = set(valid[valid[factor_col] >= high_thresh][raterParticipantIdKey])
        all_extreme_user_ids.update(low_ids)
        all_extreme_user_ids.update(high_ids)

    print(f"  Total extreme users across all factors: {len(all_extreme_user_ids):,}")

    # Load ratings - read in chunks and filter to extreme users
    print(f"  Loading ratings from extreme users...")
    ratings_path_obj = Path(ratings_path)

    if ratings_path_obj.is_dir():
        # Directory of rating files
        rating_files = sorted(ratings_path_obj.glob("ratings*.tsv"))
    else:
        rating_files = [ratings_path_obj]

    ratings_list = []
    for rf in rating_files:
        print(f"    Reading {rf.name}...")
        chunk_iter = pd.read_csv(rf, sep='\t', chunksize=500000)
        for chunk in chunk_iter:
            # Filter to extreme users only
            filtered = chunk[chunk[raterParticipantIdKey].isin(all_extreme_user_ids)]
            if len(filtered) > 0:
                ratings_list.append(filtered)

    if ratings_list:
        ratings = pd.concat(ratings_list, ignore_index=True)
    else:
        ratings = pd.DataFrame()

    print(f"  Ratings from extreme users: {len(ratings):,}")

    if factors_to_probe is None:
        factors_to_probe = list(range(1, len(factor_cols) + 1))

    for factor_idx in factors_to_probe:
        if factor_idx > len(factor_cols):
            print(f"  Skipping factor {factor_idx} (not found)")
            continue

        factor_col = factor_cols[factor_idx - 1]

        # Get extreme users
        low_users, high_users, low_thresh, high_thresh = get_extreme_users(
            users, factor_col, percentile
        )

        print(f"\nAnalyzing Factor {factor_idx} ({factor_col})...")
        print(f"  LOW users: {len(low_users):,}, HIGH users: {len(high_users):,}")

        # Get ratings for each group
        low_user_ids = set(low_users[raterParticipantIdKey])
        high_user_ids = set(high_users[raterParticipantIdKey])

        low_ratings = get_user_ratings(low_user_ids, ratings, notes)
        high_ratings = get_user_ratings(high_user_ids, ratings, notes)

        print(f"  LOW ratings: {len(low_ratings):,}, HIGH ratings: {len(high_ratings):,}")

        # Analyze patterns
        low_analysis = analyze_rating_patterns(low_ratings, f"LOW {factor_col}")
        high_analysis = analyze_rating_patterns(high_ratings, f"HIGH {factor_col}")

        # Find signature notes
        signature_notes = find_signature_notes(low_ratings, high_ratings, notes)

        # Print results
        print_factor_analysis(
            factor_idx, factor_col,
            low_thresh, high_thresh,
            low_analysis, high_analysis,
            signature_notes,
        )


def main():
    parser = argparse.ArgumentParser(description='User factor probes')
    parser.add_argument('--notes', required=True, help='Path to notes TSV')
    parser.add_argument('--users', required=True, help='Path to helpfulness_scores TSV')
    parser.add_argument('--ratings', required=True, help='Path to ratings TSV')
    parser.add_argument('--factors', type=str, default=None,
                        help='Comma-separated factor indices (default: all)')
    parser.add_argument('--percentile', type=float, default=2.0,
                        help='Percentile for extremes (default: 2.0)')
    args = parser.parse_args()

    factors = None
    if args.factors:
        factors = [int(x) for x in args.factors.split(',')]

    run_user_probes(
        args.notes,
        args.users,
        args.ratings,
        factors_to_probe=factors,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
