#!/usr/bin/env python3
"""
Factor Probes: Label axes with text/content analysis

For each factor axis:
1. Find notes at extremes (top/bottom 1-2%)
2. Print their text, metadata, topics
3. Find users at extremes and analyze their rating patterns
4. Summarize what each axis captures

This helps discover if an axis represents:
- Left/right politics
- Snarky vs earnest tone
- Institution-trusting vs skeptical
- Humor tolerance
- Domain expertise (e.g., medical misinfo)

Usage:
    python factor_probes.py \
        --notes data/notes-00000.tsv \
        --scored-notes data/scored_notes.tsv \
        --users data/helpfulness_scores.tsv \
        --ratings data/ratings/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    createdAtMillisKey,
    summaryKey,
    classificationKey,
    helpfulnessLevelKey,
    noteTopicKey,
    finalRatingStatusKey,
)


HELPFULNESS_LEVEL_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


def find_factor_columns(df: pd.DataFrame, entity_type: str, num_factors: int = 6) -> List[str]:
    """Find factor columns with mixed naming convention."""
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


def get_extreme_notes(
    scored_notes: pd.DataFrame,
    factor_col: str,
    percentile: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get notes at extreme ends of a factor axis."""
    valid = scored_notes[scored_notes[factor_col].notna()].copy()

    low_threshold = np.percentile(valid[factor_col], percentile)
    high_threshold = np.percentile(valid[factor_col], 100 - percentile)

    low_notes = valid[valid[factor_col] <= low_threshold]
    high_notes = valid[valid[factor_col] >= high_threshold]

    return low_notes, high_notes


def get_extreme_users(
    users: pd.DataFrame,
    factor_col: str,
    percentile: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get users at extreme ends of a factor axis."""
    valid = users[users[factor_col].notna()].copy()

    low_threshold = np.percentile(valid[factor_col], percentile)
    high_threshold = np.percentile(valid[factor_col], 100 - percentile)

    low_users = valid[valid[factor_col] <= low_threshold]
    high_users = valid[valid[factor_col] >= high_threshold]

    return low_users, high_users


def analyze_note_content(
    notes_subset: pd.DataFrame,
    notes_full: pd.DataFrame,
    label: str,
    n_examples: int = 10,
) -> Dict:
    """Analyze content of notes at one extreme."""

    # Merge with full notes to get text
    if summaryKey in notes_full.columns:
        merged = notes_subset.merge(
            notes_full[[noteIdKey, summaryKey]],
            on=noteIdKey,
            how='left'
        )
    else:
        merged = notes_subset.copy()
        merged[summaryKey] = "[text not available]"

    analysis = {
        'label': label,
        'count': len(notes_subset),
        'examples': [],
        'topics': {},
        'classifications': {},
        'statuses': {},
    }

    # Topic distribution
    if noteTopicKey in notes_subset.columns:
        topics = notes_subset[noteTopicKey].value_counts().head(10)
        analysis['topics'] = topics.to_dict()

    # Classification distribution
    if classificationKey in notes_subset.columns:
        classifications = notes_subset[classificationKey].value_counts()
        analysis['classifications'] = classifications.to_dict()

    # Status distribution
    if finalRatingStatusKey in notes_subset.columns:
        statuses = notes_subset[finalRatingStatusKey].value_counts()
        analysis['statuses'] = statuses.to_dict()

    # Sample examples
    sample = merged.sample(n=min(n_examples, len(merged)), random_state=42)
    for _, row in sample.iterrows():
        example = {
            'noteId': row[noteIdKey],
            'text': str(row.get(summaryKey, ''))[:500],  # Truncate
        }
        if noteTopicKey in row:
            example['topic'] = row[noteTopicKey]
        if classificationKey in row:
            example['classification'] = row[classificationKey]
        if finalRatingStatusKey in row:
            example['status'] = row[finalRatingStatusKey]
        analysis['examples'].append(example)

    return analysis


def analyze_user_ratings(
    users_subset: pd.DataFrame,
    ratings: pd.DataFrame,
    notes_full: pd.DataFrame,
    label: str,
    n_examples: int = 20,
) -> Dict:
    """Analyze what users at one extreme rate as helpful."""

    user_ids = set(users_subset[raterParticipantIdKey])

    # Get ratings from these users
    user_ratings = ratings[ratings[raterParticipantIdKey].isin(user_ids)].copy()

    # Convert helpfulness
    if helpfulnessLevelKey in user_ratings.columns:
        user_ratings['helpful_num'] = user_ratings[helpfulnessLevelKey].map(HELPFULNESS_LEVEL_MAP)
    elif 'helpful' in user_ratings.columns:
        user_ratings['helpful_num'] = user_ratings['helpful'].astype(float)

    analysis = {
        'label': label,
        'n_users': len(users_subset),
        'n_ratings': len(user_ratings),
        'mean_helpful_rate': user_ratings['helpful_num'].mean() if 'helpful_num' in user_ratings.columns else None,
    }

    # What notes do they rate as helpful?
    if 'helpful_num' in user_ratings.columns:
        helpful_ratings = user_ratings[user_ratings['helpful_num'] >= 0.5]
        not_helpful_ratings = user_ratings[user_ratings['helpful_num'] == 0]

        # Get note texts
        if summaryKey in notes_full.columns:
            helpful_notes = helpful_ratings.merge(
                notes_full[[noteIdKey, summaryKey, classificationKey] +
                          ([noteTopicKey] if noteTopicKey in notes_full.columns else [])],
                on=noteIdKey,
                how='left'
            )
            not_helpful_notes = not_helpful_ratings.merge(
                notes_full[[noteIdKey, summaryKey, classificationKey] +
                          ([noteTopicKey] if noteTopicKey in notes_full.columns else [])],
                on=noteIdKey,
                how='left'
            )

            # Sample helpful examples
            analysis['helpful_examples'] = []
            if len(helpful_notes) > 0:
                sample = helpful_notes.sample(n=min(n_examples, len(helpful_notes)), random_state=42)
                for _, row in sample.iterrows():
                    analysis['helpful_examples'].append({
                        'noteId': row[noteIdKey],
                        'text': str(row.get(summaryKey, ''))[:300],
                        'classification': row.get(classificationKey, ''),
                    })

            # Sample not-helpful examples
            analysis['not_helpful_examples'] = []
            if len(not_helpful_notes) > 0:
                sample = not_helpful_notes.sample(n=min(n_examples, len(not_helpful_notes)), random_state=42)
                for _, row in sample.iterrows():
                    analysis['not_helpful_examples'].append({
                        'noteId': row[noteIdKey],
                        'text': str(row.get(summaryKey, ''))[:300],
                        'classification': row.get(classificationKey, ''),
                    })

            # Topic preferences
            if noteTopicKey in helpful_notes.columns:
                helpful_topics = helpful_notes[noteTopicKey].value_counts().head(5)
                analysis['helpful_topics'] = helpful_topics.to_dict()

            if noteTopicKey in not_helpful_notes.columns:
                not_helpful_topics = not_helpful_notes[noteTopicKey].value_counts().head(5)
                analysis['not_helpful_topics'] = not_helpful_topics.to_dict()

    return analysis


def print_note_probe(low_analysis: Dict, high_analysis: Dict, factor_name: str):
    """Print note probe results."""

    print(f"\n{'='*80}")
    print(f"NOTE PROBE: {factor_name}")
    print(f"{'='*80}")

    # Low end
    print(f"\n--- LOW END (bottom {low_analysis['count']} notes) ---")

    if low_analysis['topics']:
        print(f"\nTop topics:")
        for topic, count in list(low_analysis['topics'].items())[:5]:
            print(f"  {topic}: {count}")

    if low_analysis['classifications']:
        print(f"\nClassifications:")
        for cls, count in low_analysis['classifications'].items():
            print(f"  {cls}: {count}")

    if low_analysis['statuses']:
        print(f"\nStatuses:")
        for status, count in low_analysis['statuses'].items():
            print(f"  {status}: {count}")

    print(f"\nExample notes:")
    for i, ex in enumerate(low_analysis['examples'][:5], 1):
        print(f"\n  [{i}] {ex.get('classification', '')} | {ex.get('topic', '')} | {ex.get('status', '')}")
        print(f"      {ex['text'][:200]}...")

    # High end
    print(f"\n--- HIGH END (top {high_analysis['count']} notes) ---")

    if high_analysis['topics']:
        print(f"\nTop topics:")
        for topic, count in list(high_analysis['topics'].items())[:5]:
            print(f"  {topic}: {count}")

    if high_analysis['classifications']:
        print(f"\nClassifications:")
        for cls, count in high_analysis['classifications'].items():
            print(f"  {cls}: {count}")

    if high_analysis['statuses']:
        print(f"\nStatuses:")
        for status, count in high_analysis['statuses'].items():
            print(f"  {status}: {count}")

    print(f"\nExample notes:")
    for i, ex in enumerate(high_analysis['examples'][:5], 1):
        print(f"\n  [{i}] {ex.get('classification', '')} | {ex.get('topic', '')} | {ex.get('status', '')}")
        print(f"      {ex['text'][:200]}...")


def print_user_probe(low_analysis: Dict, high_analysis: Dict, factor_name: str):
    """Print user probe results."""

    print(f"\n{'='*80}")
    print(f"USER PROBE: {factor_name}")
    print(f"{'='*80}")

    # Low end users
    print(f"\n--- LOW END USERS ({low_analysis['n_users']} users, {low_analysis['n_ratings']} ratings) ---")
    print(f"Mean helpful rate: {low_analysis['mean_helpful_rate']:.3f}" if low_analysis['mean_helpful_rate'] else "")

    if low_analysis.get('helpful_topics'):
        print(f"\nTopics they rate HELPFUL:")
        for topic, count in list(low_analysis['helpful_topics'].items())[:5]:
            print(f"  {topic}: {count}")

    if low_analysis.get('not_helpful_topics'):
        print(f"\nTopics they rate NOT HELPFUL:")
        for topic, count in list(low_analysis['not_helpful_topics'].items())[:5]:
            print(f"  {topic}: {count}")

    print(f"\nNotes they find HELPFUL:")
    for i, ex in enumerate(low_analysis.get('helpful_examples', [])[:3], 1):
        print(f"  [{i}] {ex.get('classification', '')}")
        print(f"      {ex['text'][:150]}...")

    print(f"\nNotes they find NOT HELPFUL:")
    for i, ex in enumerate(low_analysis.get('not_helpful_examples', [])[:3], 1):
        print(f"  [{i}] {ex.get('classification', '')}")
        print(f"      {ex['text'][:150]}...")

    # High end users
    print(f"\n--- HIGH END USERS ({high_analysis['n_users']} users, {high_analysis['n_ratings']} ratings) ---")
    print(f"Mean helpful rate: {high_analysis['mean_helpful_rate']:.3f}" if high_analysis['mean_helpful_rate'] else "")

    if high_analysis.get('helpful_topics'):
        print(f"\nTopics they rate HELPFUL:")
        for topic, count in list(high_analysis['helpful_topics'].items())[:5]:
            print(f"  {topic}: {count}")

    if high_analysis.get('not_helpful_topics'):
        print(f"\nTopics they rate NOT HELPFUL:")
        for topic, count in list(high_analysis['not_helpful_topics'].items())[:5]:
            print(f"  {topic}: {count}")

    print(f"\nNotes they find HELPFUL:")
    for i, ex in enumerate(high_analysis.get('helpful_examples', [])[:3], 1):
        print(f"  [{i}] {ex.get('classification', '')}")
        print(f"      {ex['text'][:150]}...")

    print(f"\nNotes they find NOT HELPFUL:")
    for i, ex in enumerate(high_analysis.get('not_helpful_examples', [])[:3], 1):
        print(f"  [{i}] {ex.get('classification', '')}")
        print(f"      {ex['text'][:150]}...")


def summarize_axis(
    note_low: Dict, note_high: Dict,
    user_low: Dict, user_high: Dict,
    factor_name: str
) -> str:
    """Generate a summary interpretation of what the axis captures."""

    print(f"\n{'='*80}")
    print(f"AXIS INTERPRETATION SUMMARY: {factor_name}")
    print(f"{'='*80}")

    # Compare note topics
    low_topics = set(note_low.get('topics', {}).keys())
    high_topics = set(note_high.get('topics', {}).keys())

    if low_topics and high_topics:
        only_low = low_topics - high_topics
        only_high = high_topics - low_topics

        if only_low:
            print(f"\nTopics concentrated at LOW end: {', '.join(list(only_low)[:3])}")
        if only_high:
            print(f"Topics concentrated at HIGH end: {', '.join(list(only_high)[:3])}")

    # Compare classifications
    low_class = note_low.get('classifications', {})
    high_class = note_high.get('classifications', {})

    if low_class and high_class:
        low_misinfo = low_class.get('MISINFORMED_OR_POTENTIALLY_MISLEADING', 0)
        low_not = low_class.get('NOT_MISLEADING', 0)
        high_misinfo = high_class.get('MISINFORMED_OR_POTENTIALLY_MISLEADING', 0)
        high_not = high_class.get('NOT_MISLEADING', 0)

        low_ratio = low_misinfo / (low_not + 1) if low_not else float('inf')
        high_ratio = high_misinfo / (high_not + 1) if high_not else float('inf')

        if low_ratio > high_ratio * 1.5:
            print(f"\nLOW end has MORE 'misleading' classifications")
        elif high_ratio > low_ratio * 1.5:
            print(f"HIGH end has MORE 'misleading' classifications")

    # Compare user rating patterns
    if user_low.get('mean_helpful_rate') and user_high.get('mean_helpful_rate'):
        diff = user_high['mean_helpful_rate'] - user_low['mean_helpful_rate']
        if abs(diff) > 0.05:
            if diff > 0:
                print(f"\nHIGH end users are MORE generous raters (+{diff:.2f} helpful rate)")
            else:
                print(f"LOW end users are MORE generous raters ({diff:.2f} helpful rate)")

    print(f"\n" + "-"*40)
    print("SUGGESTED LABEL (based on patterns above):")
    print("-"*40)
    print("  [Manual inspection needed - check the examples above]")
    print("  Possible interpretations:")
    print("    - Political: left vs right")
    print("    - Trust: institutional vs skeptical")
    print("    - Tone: formal vs informal")
    print("    - Domain: specific topic expertise")
    print("    - Quality: high vs low effort notes")


def run_factor_probes(
    notes_path: str,
    scored_notes_path: str,
    users_path: str,
    ratings_dir: str,
    factors_to_probe: Optional[List[int]] = None,
    percentile: float = 2.0,
    sample_ratings: Optional[int] = 1000000,
):
    """Run factor probes for all axes."""

    print("Loading data...")

    # Load notes (for text)
    notes_full = pd.read_csv(notes_path, sep='\t')
    print(f"  Loaded {len(notes_full):,} notes with text")

    # Load scored notes (for factors)
    scored_notes = pd.read_csv(scored_notes_path, sep='\t')
    print(f"  Loaded {len(scored_notes):,} scored notes")

    # Load users
    users = pd.read_csv(users_path, sep='\t')
    print(f"  Loaded {len(users):,} users")

    # Load ratings
    ratings_path = Path(ratings_dir)
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path, sep='\t')
    else:
        rating_files = list(ratings_path.glob("ratings-*.tsv"))
        if not rating_files:
            rating_files = list(ratings_path.glob("ratings*.tsv"))
        ratings_list = [pd.read_csv(f, sep='\t') for f in sorted(rating_files)[:1]]  # Just first file
        ratings = pd.concat(ratings_list, ignore_index=True)

    if sample_ratings and len(ratings) > sample_ratings:
        ratings = ratings.sample(n=sample_ratings, random_state=42)
    print(f"  Loaded {len(ratings):,} ratings")

    # Find factor columns
    note_factor_cols = find_factor_columns(scored_notes, 'note')
    user_factor_cols = find_factor_columns(users, 'user')

    print(f"  Found {len(note_factor_cols)} note factors, {len(user_factor_cols)} user factors")

    # Determine which factors to probe
    if factors_to_probe is None:
        factors_to_probe = list(range(1, len(note_factor_cols) + 1))

    results = {}

    for factor_idx in factors_to_probe:
        if factor_idx > len(note_factor_cols):
            continue

        note_col = note_factor_cols[factor_idx - 1]
        user_col = user_factor_cols[factor_idx - 1] if factor_idx <= len(user_factor_cols) else None

        print(f"\n\n{'#'*80}")
        print(f"# PROBING FACTOR {factor_idx}: {note_col}")
        print(f"{'#'*80}")

        # Note probes
        low_notes, high_notes = get_extreme_notes(scored_notes, note_col, percentile)

        low_note_analysis = analyze_note_content(low_notes, notes_full, f"LOW {note_col}")
        high_note_analysis = analyze_note_content(high_notes, notes_full, f"HIGH {note_col}")

        print_note_probe(low_note_analysis, high_note_analysis, note_col)

        # User probes
        if user_col:
            low_users, high_users = get_extreme_users(users, user_col, percentile)

            low_user_analysis = analyze_user_ratings(low_users, ratings, notes_full, f"LOW {user_col}")
            high_user_analysis = analyze_user_ratings(high_users, ratings, notes_full, f"HIGH {user_col}")

            print_user_probe(low_user_analysis, high_user_analysis, user_col)

            # Summary
            summarize_axis(
                low_note_analysis, high_note_analysis,
                low_user_analysis, high_user_analysis,
                f"Factor {factor_idx}"
            )

        results[factor_idx] = {
            'note_low': low_note_analysis,
            'note_high': high_note_analysis,
            'user_low': low_user_analysis if user_col else None,
            'user_high': high_user_analysis if user_col else None,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Factor probes to label axes')
    parser.add_argument('--notes', required=True, help='Path to notes TSV (with text)')
    parser.add_argument('--scored-notes', required=True, help='Path to scored notes TSV')
    parser.add_argument('--users', required=True, help='Path to helpfulness scores TSV')
    parser.add_argument('--ratings', required=True, help='Path to ratings directory')
    parser.add_argument('--factors', type=str, default=None,
                        help='Comma-separated factor indices to probe (default: all)')
    parser.add_argument('--percentile', type=float, default=2.0,
                        help='Percentile for extremes (default: 2.0 = top/bottom 2%%)')
    args = parser.parse_args()

    factors = None
    if args.factors:
        factors = [int(x) for x in args.factors.split(',')]

    run_factor_probes(
        args.notes,
        args.scored_notes,
        args.users,
        args.ratings,
        factors_to_probe=factors,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
