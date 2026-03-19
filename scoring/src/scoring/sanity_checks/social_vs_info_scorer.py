#!/usr/bin/env python3
"""
Social vs Informational Note Scorer

Goal: Identify CRH notes that are "jokes / high social utility, low info utility"

Pipeline:
1. Restrict to CRH notes only
2. Compute Info Score: sources, numbers, factual claims, external links, named entities
3. Compute Social Score: punchlines, sarcasm, rhetorical questions, clapback structure
4. Use factor loadings to learn what separates social vs info notes
5. Rank "high social / low info" notes

Usage:
    python social_vs_info_scorer.py \
        --notes data/notes-00000.tsv \
        --scored-notes data/scored_notes.tsv \
        --outdir data/social_info_analysis
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    summaryKey,
    finalRatingStatusKey,
)


# ============================================================================
# INFO SCORE FEATURES
# ============================================================================

def count_urls(text: str) -> int:
    """Count URLs/links in text."""
    if pd.isna(text):
        return 0
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return len(re.findall(url_pattern, str(text)))


def count_numbers(text: str) -> int:
    """Count numeric values (dates, percentages, statistics)."""
    if pd.isna(text):
        return 0
    # Match numbers with optional units/percentages
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|M|B|K|million|billion|thousand)?\b'
    return len(re.findall(number_pattern, str(text), re.IGNORECASE))


def has_citation_markers(text: str) -> bool:
    """Check for citation-like patterns."""
    if pd.isna(text):
        return False
    text = str(text)
    patterns = [
        r'according to',
        r'source[s]?:',
        r'reference[s]?:',
        r'cited',
        r'study|research|report',
        r'published',
        r'\([12][0-9]{3}\)',  # Year in parentheses
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def count_hedging_words(text: str) -> int:
    """Count hedging/uncertainty markers (indicates careful factual claims)."""
    if pd.isna(text):
        return 0
    hedges = [
        'may', 'might', 'could', 'possibly', 'potentially',
        'appears', 'seems', 'suggests', 'indicates',
        'allegedly', 'reportedly', 'claimed', 'purported',
    ]
    text_lower = str(text).lower()
    return sum(1 for h in hedges if h in text_lower)


def compute_info_score(text: str) -> Dict[str, float]:
    """Compute information utility score features."""
    features = {
        'n_urls': count_urls(text),
        'n_numbers': count_numbers(text),
        'has_citations': 1.0 if has_citation_markers(text) else 0.0,
        'n_hedging': count_hedging_words(text),
        'text_length': len(str(text)) if not pd.isna(text) else 0,
    }

    # Composite info score (normalized)
    features['info_score'] = (
        features['n_urls'] * 2.0 +  # URLs are strong info signal
        features['n_numbers'] * 0.5 +
        features['has_citations'] * 1.5 +
        features['n_hedging'] * 0.3
    )

    return features


# ============================================================================
# SOCIAL SCORE FEATURES
# ============================================================================

def count_rhetorical_questions(text: str) -> int:
    """Count rhetorical questions."""
    if pd.isna(text):
        return 0
    # Questions that aren't seeking info
    return len(re.findall(r'\?', str(text)))


def has_sarcasm_markers(text: str) -> bool:
    """Check for sarcasm/irony markers."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    markers = [
        'clearly', 'obviously', 'definitely', 'surely',
        '/s', '(sarcasm)', 'shocking', 'surprise',
        'imagine', 'apparently',
    ]
    return any(m in text for m in markers)


def has_punchline_structure(text: str) -> bool:
    """Check for setup-punchline structure."""
    if pd.isna(text):
        return False
    text = str(text)
    # Short sentence followed by short sentence
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 2:
        # Last sentence much shorter than previous = punchline
        if len(sentences) > 1:
            last_len = len(sentences[-1])
            prev_len = len(sentences[-2])
            if last_len < 50 and prev_len > last_len * 1.5:
                return True
    return False


def has_clapback_structure(text: str) -> bool:
    """Check for clapback/burn structure."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    clapback_markers = [
        'actually', 'in fact', 'wrong', 'false', 'incorrect',
        'no,', 'nope', 'not true', 'fake',
        'this is not', 'this isn\'t', 'that\'s not',
    ]
    return any(m in text for m in clapback_markers)


def has_second_person(text: str) -> bool:
    """Check for second-person address (engaging the reader)."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    return bool(re.search(r'\byou\b|\byour\b|\byou\'re\b', text))


def has_emoji(text: str) -> bool:
    """Check for emoji usage."""
    if pd.isna(text):
        return False
    # Basic emoji detection
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(str(text)))


def has_memey_phrases(text: str) -> bool:
    """Check for meme-like language."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    memes = [
        'lol', 'lmao', 'bruh', 'literally', 'iconic',
        'based', 'cringe', 'cope', 'seethe', 'ratio',
        'trust me bro', 'source: trust me',
        'well actually', 'um actually',
    ]
    return any(m in text for m in memes)


def compute_social_score(text: str) -> Dict[str, float]:
    """Compute social utility score features."""
    features = {
        'n_questions': count_rhetorical_questions(text),
        'has_sarcasm': 1.0 if has_sarcasm_markers(text) else 0.0,
        'has_punchline': 1.0 if has_punchline_structure(text) else 0.0,
        'has_clapback': 1.0 if has_clapback_structure(text) else 0.0,
        'has_second_person': 1.0 if has_second_person(text) else 0.0,
        'has_emoji': 1.0 if has_emoji(text) else 0.0,
        'has_meme': 1.0 if has_memey_phrases(text) else 0.0,
    }

    # Composite social score
    features['social_score'] = (
        features['n_questions'] * 0.5 +
        features['has_sarcasm'] * 2.0 +
        features['has_punchline'] * 2.0 +
        features['has_clapback'] * 1.5 +
        features['has_second_person'] * 0.5 +
        features['has_emoji'] * 1.0 +
        features['has_meme'] * 2.0
    )

    return features


# ============================================================================
# FACTOR-BASED ANALYSIS
# ============================================================================

def find_note_factor_columns(df: pd.DataFrame, num_factors: int = 6) -> List[str]:
    """Find note factor columns."""
    factor_cols = []
    for i in range(1, num_factors + 1):
        candidates = [f"internalNoteFactor{i}", f"coreNoteFactor{i}"]
        for col in candidates:
            if col in df.columns:
                factor_cols.append(col)
                break
    return factor_cols


def analyze_factors_vs_scores(
    df: pd.DataFrame,
    factor_cols: List[str],
) -> Dict:
    """Correlate factors with social/info scores."""
    correlations = {}

    for factor_col in factor_cols:
        valid = df[df[factor_col].notna()]
        if len(valid) < 100:
            continue

        correlations[factor_col] = {
            'info_corr': valid['info_score'].corr(valid[factor_col]),
            'social_corr': valid['social_score'].corr(valid[factor_col]),
            'diff_corr': (valid['social_score'] - valid['info_score']).corr(valid[factor_col]),
        }

    return correlations


# ============================================================================
# HUMOR / LOW-EVIDENCE HELPFUL CANDIDATES
# ============================================================================

def find_joke_notes(
    notes: pd.DataFrame,
    scored_notes: pd.DataFrame,
    include_not_helpful: bool = True,
) -> pd.DataFrame:
    """
    Find joke/sarcastic/troll notes - short punchy one-liners.

    These are notes like:
    - "The world did not end."
    - "No.    https://en.m.wikipedia.org/wiki/No"
    - "Joke   https://en.wikipedia.org/wiki/Joke"

    Characteristics:
    1. Very short text (under 100 chars without URL)
    2. May include ironic Wikipedia links
    3. One-liner structure
    4. Can be CRH or not-helpful (jokes can fail to get consensus)

    Args:
        notes: Notes with text
        scored_notes: Notes with scoring output
        include_not_helpful: Whether to include non-CRH notes

    Returns:
        DataFrame of joke note candidates
    """
    # Merge notes with scored notes
    merged = scored_notes.merge(notes[[noteIdKey, summaryKey]], on=noteIdKey, how='left')

    # Filter by status if not including all
    if not include_not_helpful:
        crh_status = 'CURRENTLY_RATED_HELPFUL'
        if finalRatingStatusKey in merged.columns:
            merged = merged[merged[finalRatingStatusKey] == crh_status]

    # Compute text length without URLs
    def text_without_urls(text):
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove URLs
        text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]+', '', text)
        return text.strip()

    merged['text_no_url'] = merged[summaryKey].apply(text_without_urls)
    merged['text_length_no_url'] = merged['text_no_url'].str.len()

    # Detect joke Wikipedia links
    joke_wiki_patterns = [
        r'wikipedia\.org/wiki/(Joke|No|Yes|Lie|Truth|Incel|Tool|Video|Sarcasm)',
        r'youtube\.com.*dQw4w9WgXcQ',  # Rickroll
        r'youtu\.be.*dQw4w9WgXcQ',
    ]

    def has_joke_link(text):
        if pd.isna(text):
            return False
        text = str(text)
        return any(re.search(p, text, re.IGNORECASE) for p in joke_wiki_patterns)

    merged['has_joke_link'] = merged[summaryKey].apply(has_joke_link)

    # Detect profanity/informal markers
    informal_words = ['fuck', 'shit', 'damn', 'ass', 'cuck', 'virgin', 'loser', 'retard', 'lmao', 'lol', 'cooked']

    def has_informal_language(text):
        if pd.isna(text):
            return False
        text = str(text).lower()
        return any(w in text for w in informal_words)

    merged['has_informal'] = merged[summaryKey].apply(has_informal_language)

    # Detect very short punchy structure
    def is_punchy(text):
        if pd.isna(text):
            return False
        text_clean = text_without_urls(text)
        # Short, single sentence, possibly with emoji
        return len(text_clean) < 80 and text_clean.count('.') <= 2

    merged['is_punchy'] = merged[summaryKey].apply(is_punchy)

    # Filter: (very short OR has joke link OR has informal language) AND punchy structure
    joke_mask = (
        (merged['text_length_no_url'] < 100) &
        (
            (merged['has_joke_link']) |
            (merged['has_informal']) |
            (merged['text_length_no_url'] < 50)  # Very short = likely joke/comment
        )
    )

    candidates = merged[joke_mask].copy()

    # Score them
    candidates['joke_score'] = (
        (100 - candidates['text_length_no_url'].clip(0, 100)) / 100 * 2 +  # Shorter = more joke-like
        candidates['has_joke_link'].astype(float) * 3 +  # Joke links
        candidates['has_informal'].astype(float) * 1 +  # Informal language
        candidates['is_punchy'].astype(float) * 1
    )

    return candidates.sort_values('joke_score', ascending=False)


def find_humor_candidates(
    notes: pd.DataFrame,
    scored_notes: pd.DataFrame,
    ratings: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Find humor/social-only helpfulness candidates.

    Better definition than "clapback fact-checks": Low-Evidence Helpful Notes.

    Criteria:
    1. CRH notes (must be helpful)
    2. Low info score (no URL, few numbers, no citations)
    3. Not extreme on F6 (avoid politics dominating)
    4. Optional: High bimodal response within same F6 side

    This identifies notes that achieved CRH status despite having little
    "traditional" informational content - suggesting they provide some
    other form of utility (humor, perspective shift, etc.)

    Args:
        notes: Notes with text
        scored_notes: Notes with scoring output
        ratings: Optional ratings for bimodal analysis

    Returns:
        DataFrame of humor candidates with scores
    """
    # Merge notes with text
    crh_status = 'CURRENTLY_RATED_HELPFUL'
    if finalRatingStatusKey in scored_notes.columns:
        crh = scored_notes[scored_notes[finalRatingStatusKey] == crh_status].copy()
    else:
        status_cols = [c for c in scored_notes.columns if 'status' in c.lower()]
        if status_cols:
            crh = scored_notes[scored_notes[status_cols[0]] == crh_status].copy()
        else:
            crh = scored_notes.copy()

    # Merge with text
    merged = crh.merge(notes[[noteIdKey, summaryKey]], on=noteIdKey, how='left')

    # Compute info features for filtering
    info_features = merged[summaryKey].apply(compute_info_score)
    merged['info_n_urls'] = info_features.apply(lambda x: x['n_urls'])
    merged['info_n_numbers'] = info_features.apply(lambda x: x['n_numbers'])
    merged['info_has_citations'] = info_features.apply(lambda x: x['has_citations'])
    merged['info_score'] = info_features.apply(lambda x: x['info_score'])

    # Compute social features
    social_features = merged[summaryKey].apply(compute_social_score)
    merged['social_score'] = social_features.apply(lambda x: x['social_score'])

    # Filter 1: Low info (no URL, <2 numbers, no citations)
    low_info_mask = (
        (merged['info_n_urls'] == 0) &
        (merged['info_n_numbers'] < 2) &
        (merged['info_has_citations'] == 0)
    )

    # Filter 2: Not political extreme (if F6 is available)
    f6_col = None
    for col in merged.columns:
        if 'Factor6' in col or 'factor6' in col.lower():
            f6_col = col
            break

    if f6_col and f6_col in merged.columns and merged[f6_col].notna().sum() > 0:
        f6_std = merged[f6_col].std()
        not_political_mask = merged[f6_col].abs() < f6_std
    else:
        not_political_mask = pd.Series(True, index=merged.index)

    # Combine filters
    candidates = merged[low_info_mask & not_political_mask].copy()

    # Score candidates: higher social score + lower info score = more interesting
    candidates['humor_score'] = candidates['social_score'] - candidates['info_score']

    # Sort by humor_score
    candidates = candidates.sort_values('humor_score', ascending=False)

    return candidates


def compute_bimodal_response(
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
    user_f6_col: str,
) -> pd.DataFrame:
    """
    Compute bimodal response metric: high variance of ratings within same F6 side.

    This identifies notes where even users who agree ideologically (same F6 sign)
    still have high variance in their ratings - suggesting the note is polarizing
    in a non-political way (e.g., some find it funny, others don't).

    Args:
        ratings: Ratings with user info
        notes: Notes with F6 values
        user_f6_col: Column name for user F6

    Returns:
        DataFrame with bimodal response scores per note
    """
    if user_f6_col not in ratings.columns:
        return pd.DataFrame()

    # Join ratings with user F6
    ratings_with_f6 = ratings.copy()

    # Determine which side of F6 each user is on
    ratings_with_f6['f6_side'] = (ratings_with_f6[user_f6_col] > 0).astype(int)

    # Compute within-side variance for each note
    from constants import helpfulNumKey, helpfulnessLevelKey

    if helpfulNumKey in ratings_with_f6.columns:
        helpful_col = helpfulNumKey
    elif helpfulnessLevelKey in ratings_with_f6.columns:
        # Map to numeric
        HELPFULNESS_MAP = {'NOT_HELPFUL': 0.0, 'SOMEWHAT_HELPFUL': 0.5, 'HELPFUL': 1.0}
        ratings_with_f6['helpful_num'] = ratings_with_f6[helpfulnessLevelKey].map(HELPFULNESS_MAP)
        helpful_col = 'helpful_num'
    else:
        return pd.DataFrame()

    # Group by note and F6 side, compute variance
    within_side_var = ratings_with_f6.groupby([noteIdKey, 'f6_side'])[helpful_col].var()
    within_side_var = within_side_var.reset_index()
    within_side_var.columns = [noteIdKey, 'f6_side', 'within_side_var']

    # Average within-side variance per note
    bimodal_score = within_side_var.groupby(noteIdKey)['within_side_var'].mean()
    bimodal_score = bimodal_score.reset_index()
    bimodal_score.columns = [noteIdKey, 'bimodal_score']

    return bimodal_score


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_analysis(
    notes_path: str,
    scored_notes_path: str,
    outdir: str,
):
    """Run the social vs info analysis."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Load notes with text
    notes = pd.read_csv(notes_path, sep='\t', low_memory=False)
    print(f"  Notes with text: {len(notes):,}")

    # Load scored notes with factors
    scored_notes = pd.read_csv(scored_notes_path, sep='\t')
    print(f"  Scored notes: {len(scored_notes):,}")

    # Find CRH notes
    crh_status = 'CURRENTLY_RATED_HELPFUL'
    if finalRatingStatusKey in scored_notes.columns:
        crh_notes = scored_notes[scored_notes[finalRatingStatusKey] == crh_status]
    else:
        # Try alternate column names
        status_cols = [c for c in scored_notes.columns if 'status' in c.lower()]
        if status_cols:
            crh_notes = scored_notes[scored_notes[status_cols[0]] == crh_status]
        else:
            print("  Warning: No status column found, using all notes")
            crh_notes = scored_notes

    print(f"  CRH notes: {len(crh_notes):,}")

    # Merge to get text
    merged = crh_notes.merge(notes[[noteIdKey, summaryKey]], on=noteIdKey, how='left')
    print(f"  Merged notes with text: {merged[summaryKey].notna().sum():,}")

    # Compute scores
    print("\nComputing info and social scores...")

    info_features = merged[summaryKey].apply(compute_info_score)
    social_features = merged[summaryKey].apply(compute_social_score)

    # Convert to DataFrame columns
    for key in info_features.iloc[0].keys():
        merged[f'info_{key}'] = info_features.apply(lambda x: x[key])
    for key in social_features.iloc[0].keys():
        merged[f'social_{key}'] = social_features.apply(lambda x: x[key])

    # Rename for clarity
    merged['info_score'] = merged['info_info_score']
    merged['social_score'] = merged['social_social_score']

    # Compute ratio: high social / low info
    merged['social_minus_info'] = merged['social_score'] - merged['info_score']
    merged['social_info_ratio'] = merged['social_score'] / (merged['info_score'] + 0.1)

    # ========================================================================
    # ANALYSIS: Factor correlations
    # ========================================================================
    print("\nAnalyzing factor correlations...")

    factor_cols = find_note_factor_columns(merged)
    print(f"  Found {len(factor_cols)} factor columns")

    if factor_cols:
        correlations = analyze_factors_vs_scores(merged, factor_cols)

        print("\nFactor correlations with scores:")
        print("-" * 60)
        print(f"{'Factor':<25} {'Info r':<12} {'Social r':<12} {'Diff r':<12}")
        print("-" * 60)
        for factor, corrs in correlations.items():
            print(f"{factor:<25} {corrs['info_corr']:>10.3f}  {corrs['social_corr']:>10.3f}  {corrs['diff_corr']:>10.3f}")

    # ========================================================================
    # FIND HUMOR CANDIDATES (Low-Evidence Helpful Notes)
    # ========================================================================
    print("\n" + "=" * 80)
    print("HUMOR CANDIDATES: Low-Evidence Helpful Notes")
    print("=" * 80)

    print("\nFinding CRH notes that achieved helpful status without traditional info content...")
    print("(No URLs, few numbers, no citations, not politically extreme)")

    humor_candidates = find_humor_candidates(notes, crh_notes, ratings=None)

    print(f"\nHumor candidates found: {len(humor_candidates):,}")
    print("\nTop 20 humor candidates (highest humor_score = social - info):")

    for i, (_, row) in enumerate(humor_candidates.head(20).iterrows(), 1):
        text = str(row.get(summaryKey, ''))[:300]
        print(f"\n[{i}] Humor={row['humor_score']:.1f}, Social={row['social_score']:.1f}, Info={row['info_score']:.1f}")
        print(f"    {text}...")

    # ========================================================================
    # FIND TOP SOCIAL / LOW INFO NOTES (original method for comparison)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOP 'HIGH SOCIAL / LOW INFO' CRH NOTES (original clapback detector)")
    print("=" * 80)

    # Filter to notes with some social signal
    social_notes = merged[merged['social_score'] > 0].copy()
    social_notes = social_notes.sort_values('social_minus_info', ascending=False)

    print(f"\nNotes with social_score > 0: {len(social_notes):,}")
    print("\nTop 10 high-social/low-info notes:")

    for i, (_, row) in enumerate(social_notes.head(10).iterrows(), 1):
        text = str(row.get(summaryKey, ''))[:300]
        print(f"\n[{i}] Social={row['social_score']:.1f}, Info={row['info_score']:.1f}, Diff={row['social_minus_info']:.1f}")
        print(f"    {text}...")

    # ========================================================================
    # FIND TOP INFO / LOW SOCIAL NOTES (for comparison)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOP 'HIGH INFO / LOW SOCIAL' CRH NOTES (for comparison)")
    print("=" * 80)

    info_notes = merged[merged['info_score'] > 3].copy()  # At least 3 info signals
    info_notes = info_notes.sort_values('social_minus_info', ascending=True)  # Low social

    print(f"\nNotes with info_score > 3: {len(info_notes):,}")
    print("\nTop 10 high-info/low-social notes:")

    for i, (_, row) in enumerate(info_notes.head(10).iterrows(), 1):
        text = str(row.get(summaryKey, ''))[:300]
        print(f"\n[{i}] Social={row['social_score']:.1f}, Info={row['info_score']:.1f}, Diff={row['social_minus_info']:.1f}")
        print(f"    {text}...")

    # ========================================================================
    # STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTIONS")
    print("=" * 80)

    print(f"\nInfo Score:   mean={merged['info_score'].mean():.2f}, std={merged['info_score'].std():.2f}, median={merged['info_score'].median():.2f}")
    print(f"Social Score: mean={merged['social_score'].mean():.2f}, std={merged['social_score'].std():.2f}, median={merged['social_score'].median():.2f}")

    # Percentile breakdown
    print("\nInfo Score percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(merged['info_score'], p):.2f}")

    print("\nSocial Score percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(merged['social_score'], p):.2f}")

    # Feature breakdown
    print("\nFeature prevalence in CRH notes:")
    print(f"  Has URLs: {(merged['info_n_urls'] > 0).mean():.1%}")
    print(f"  Has numbers: {(merged['info_n_numbers'] > 0).mean():.1%}")
    print(f"  Has citations: {merged['info_has_citations'].mean():.1%}")
    print(f"  Has questions: {(merged['social_n_questions'] > 0).mean():.1%}")
    print(f"  Has sarcasm: {merged['social_has_sarcasm'].mean():.1%}")
    print(f"  Has clapback: {merged['social_has_clapback'].mean():.1%}")
    print(f"  Has emoji: {merged['social_has_emoji'].mean():.1%}")
    print(f"  Has meme phrases: {merged['social_has_meme'].mean():.1%}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print(f"\nSaving results to {outdir}...")

    # Save all CRH notes with scores
    output_cols = [noteIdKey, summaryKey, 'info_score', 'social_score',
                   'social_minus_info', 'social_info_ratio'] + factor_cols
    output_cols = [c for c in output_cols if c in merged.columns]

    merged[output_cols].to_csv(outdir / 'crh_notes_scored.tsv', sep='\t', index=False)

    # Save top social notes (original method)
    social_notes.head(100)[output_cols].to_csv(
        outdir / 'top_social_notes.tsv', sep='\t', index=False
    )

    # Save humor candidates (new method)
    humor_output_cols = [noteIdKey, summaryKey, 'info_score', 'social_score', 'humor_score',
                         'info_n_urls', 'info_n_numbers', 'info_has_citations']
    humor_output_cols = [c for c in humor_output_cols if c in humor_candidates.columns]

    if len(humor_candidates) > 0:
        # Save top 200 for manual review
        humor_candidates.head(200)[humor_output_cols].to_csv(
            outdir / 'humor_candidates_for_review.tsv', sep='\t', index=False
        )
        print(f"  Saved {min(200, len(humor_candidates))} humor candidates for manual review")

    # Save summary statistics
    summary = {
        'total_crh_notes': len(merged),
        'notes_with_social_signal': len(social_notes),
        'humor_candidates': len(humor_candidates),
        'info_score_mean': float(merged['info_score'].mean()),
        'info_score_median': float(merged['info_score'].median()),
        'social_score_mean': float(merged['social_score'].mean()),
        'social_score_median': float(merged['social_score'].median()),
        'has_url_pct': float((merged['info_n_urls'] > 0).mean()),
        'has_citation_pct': float(merged['info_has_citations'].mean()),
    }

    import json
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Done!")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Social vs Info Note Scorer')
    parser.add_argument('--notes', required=True, help='Path to notes TSV')
    parser.add_argument('--scored-notes', required=True, help='Path to scored notes TSV')
    parser.add_argument('--outdir', default='data/social_info_analysis', help='Output directory')
    args = parser.parse_args()

    run_analysis(args.notes, args.scored_notes, args.outdir)


if __name__ == "__main__":
    main()
