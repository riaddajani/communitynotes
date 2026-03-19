#!/usr/bin/env python3
"""
Factor Rotation and Interpretation Script

Rotates the 6-factor embedding space into interpretable axes:
- Common-ground/helpfulness axis: where users agree (low disagreement)
- Polarization axes: where users disagree (high disagreement)

The key insight: raw MF factors are rotation-ambiguous, but we can find
a meaningful basis by measuring disagreement along each axis.

Usage:
    python rotate_factors.py \
        --notes data/scored_notes.tsv \
        --users data/helpfulness_scores.tsv \
        --ratings data/
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orthogonal_procrustes

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    createdAtMillisKey,
    helpfulNumKey,
    helpfulnessLevelKey,
    note_factor_key,
    rater_factor_key,
    internalNoteInterceptKey,
    internalRaterInterceptKey,
)

# Helpfulness level mappings
HELPFULNESS_LEVEL_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


def get_helpful_numeric(df: pd.DataFrame) -> pd.Series:
    """
    Get numeric helpfulness from ratings dataframe.
    Converts helpfulnessLevel to numeric if needed.
    """
    # Try helpfulNum first (already numeric)
    if helpfulNumKey in df.columns:
        return df[helpfulNumKey].astype(float)

    # Try helpfulnessLevel (categorical)
    if helpfulnessLevelKey in df.columns:
        return df[helpfulnessLevelKey].map(HELPFULNESS_LEVEL_MAP).astype(float)

    # Fall back to helpful column (binary)
    if 'helpful' in df.columns:
        return df['helpful'].astype(float)

    raise ValueError(f"No helpful column found. Available: {list(df.columns)[:20]}")


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


def compute_disagreement_score_by_axis(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    user_factor_cols: List[str],
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute disagreement score for each factor axis.

    For each axis k:
    1. Bin users by their position along axis k
    2. Within each bin, compute variance of their ratings (after removing note effects)
    3. Disagreement = mean variance across bins

    Low disagreement = users at similar positions rate similarly = common ground
    High disagreement = users at similar positions still disagree = polarization
    """
    # Merge ratings with user factors
    user_cols = [raterParticipantIdKey] + user_factor_cols
    ratings_with_factors = ratings.merge(
        users[user_cols],
        on=raterParticipantIdKey,
        how='inner'
    )

    # Get helpful values as numeric
    ratings_with_factors['helpful_num'] = get_helpful_numeric(ratings_with_factors)

    # Compute note-level mean rating (to remove note effects)
    note_means = ratings_with_factors.groupby(noteIdKey)['helpful_num'].mean()
    ratings_with_factors = ratings_with_factors.merge(
        note_means.rename('note_mean'),
        on=noteIdKey,
        how='left'
    )
    ratings_with_factors['residual'] = ratings_with_factors['helpful_num'] - ratings_with_factors['note_mean']

    disagreement_scores = {}

    for factor_col in user_factor_cols:
        # Bin users by their factor value
        factor_values = ratings_with_factors[factor_col].dropna()
        if len(factor_values) == 0:
            disagreement_scores[factor_col] = np.nan
            continue

        # Create bins
        try:
            ratings_with_factors['bin'] = pd.qcut(
                ratings_with_factors[factor_col],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # Not enough unique values for qcut
            ratings_with_factors['bin'] = pd.cut(
                ratings_with_factors[factor_col],
                bins=n_bins,
                labels=False
            )

        # Compute variance of residuals within each bin
        bin_variances = ratings_with_factors.groupby('bin')['residual'].var()

        # Disagreement score = mean variance across bins
        # Weight by bin size for more stable estimate
        bin_counts = ratings_with_factors.groupby('bin')['residual'].count()
        weighted_var = (bin_variances * bin_counts).sum() / bin_counts.sum()

        disagreement_scores[factor_col] = weighted_var

    return disagreement_scores


def compute_explained_variance_by_axis(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    notes: pd.DataFrame,
    user_factor_cols: List[str],
    note_factor_cols: List[str],
) -> Dict[str, float]:
    """
    Compute how much variance each factor axis explains in ratings.

    For each axis k:
    - Compute user_factor_k * note_factor_k contribution
    - Measure correlation with rating residuals

    Higher explained variance = more important axis for predictions.
    """
    # Merge all data
    user_cols = [raterParticipantIdKey] + user_factor_cols
    note_cols = [noteIdKey] + note_factor_cols

    merged = ratings.merge(users[user_cols], on=raterParticipantIdKey, how='inner')
    merged = merged.merge(notes[note_cols], on=noteIdKey, how='inner')

    # Get helpful values as numeric
    merged['helpful_num'] = get_helpful_numeric(merged)

    explained_variance = {}

    for uf, nf in zip(user_factor_cols, note_factor_cols):
        # Factor contribution: user_factor * note_factor
        contribution = merged[uf] * merged[nf]

        # Correlation with actual rating
        valid = pd.DataFrame({'contribution': contribution, 'rating': merged['helpful_num']}).dropna()
        if len(valid) > 100:
            try:
                corr, _ = stats.pearsonr(valid['contribution'], valid['rating'])
                # R-squared = proportion of variance explained
                explained_variance[uf] = corr ** 2 if not np.isnan(corr) else np.nan
            except:
                explained_variance[uf] = np.nan
        else:
            explained_variance[uf] = np.nan

    return explained_variance


def compute_cross_cutting_score(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    notes: pd.DataFrame,
    user_factor_cols: List[str],
    note_factor_cols: List[str],
) -> Dict[str, float]:
    """
    Compute cross-cutting score: how much does position on this axis
    predict DIFFERENT ratings for the same note?

    High cross-cutting = users on opposite ends rate same notes differently = polarization
    Low cross-cutting = user position doesn't predict rating differences = common ground
    """
    # Merge all data
    user_cols = [raterParticipantIdKey] + user_factor_cols
    note_cols = [noteIdKey] + note_factor_cols

    merged = ratings.merge(users[user_cols], on=raterParticipantIdKey, how='inner')
    merged = merged.merge(notes[note_cols], on=noteIdKey, how='inner')

    # Get helpful values as numeric
    merged['helpful_num'] = get_helpful_numeric(merged)

    cross_cutting_scores = {}

    for uf, nf in zip(user_factor_cols, note_factor_cols):
        # For each note, compute correlation between user factor and their rating
        # Then average the absolute correlations

        note_correlations = []

        # Sample notes for efficiency - prefer notes with many ratings
        note_rating_counts = merged.groupby(noteIdKey).size()
        notes_with_enough = note_rating_counts[note_rating_counts >= 10].index
        sample_notes = np.random.choice(
            notes_with_enough,
            min(5000, len(notes_with_enough)),
            replace=False
        )

        for note_id in sample_notes:
            note_ratings = merged[merged[noteIdKey] == note_id]
            if len(note_ratings) >= 10:  # Need enough ratings
                # Check for variance in both variables
                factor_std = note_ratings[uf].std()
                rating_std = note_ratings['helpful_num'].std()
                if factor_std > 0.01 and rating_std > 0.01:  # Avoid constant inputs
                    try:
                        corr, _ = stats.pearsonr(note_ratings[uf], note_ratings['helpful_num'])
                        if not np.isnan(corr):
                            note_correlations.append(abs(corr))
                    except:
                        pass

        if note_correlations:
            # Average absolute correlation = cross-cutting score
            cross_cutting_scores[uf] = np.mean(note_correlations)
        else:
            cross_cutting_scores[uf] = np.nan

    return cross_cutting_scores


def compute_polarization_score(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    notes: pd.DataFrame,
    user_factor_cols: List[str],
    note_factor_cols: List[str],
) -> Dict[str, float]:
    """
    Alternative polarization measure: correlation between user-note factor
    alignment and rating deviation from mean.

    High correlation = axis captures systematic disagreement = polarization
    Low correlation = axis doesn't predict disagreement = common ground candidate
    """
    # Merge all data
    user_cols = [raterParticipantIdKey] + user_factor_cols
    note_cols = [noteIdKey] + note_factor_cols

    merged = ratings.merge(users[user_cols], on=raterParticipantIdKey, how='inner')
    merged = merged.merge(notes[note_cols], on=noteIdKey, how='inner')

    # Get helpful values as numeric
    merged['helpful_num'] = get_helpful_numeric(merged)

    # Compute note mean
    note_means = merged.groupby(noteIdKey)['helpful_num'].transform('mean')
    merged['deviation'] = merged['helpful_num'] - note_means

    polarization_scores = {}

    for uf, nf in zip(user_factor_cols, note_factor_cols):
        # Compute alignment: user_factor * note_factor
        merged['alignment'] = merged[uf] * merged[nf]

        # Correlation between alignment and deviation
        valid = merged[['alignment', 'deviation']].dropna()
        if len(valid) > 100:
            try:
                corr, _ = stats.pearsonr(valid['alignment'], valid['deviation'])
                polarization_scores[uf] = abs(corr) if not np.isnan(corr) else np.nan
            except:
                polarization_scores[uf] = np.nan
        else:
            polarization_scores[uf] = np.nan

    return polarization_scores


def identify_axis_types(
    cross_cutting_scores: Dict[str, float],
    explained_variance: Dict[str, float],
    polarization_scores: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Identify which axes are common-ground vs polarization.

    Uses polarization score as primary metric (fallback when cross-cutting is NaN):
    - Low polarization = common ground (user-note alignment doesn't predict deviation)
    - High polarization = polarization (alignment predicts disagreement)

    Returns dict mapping factor name to:
    - type: 'common_ground' or 'polarization'
    - cross_cutting: how much position predicts disagreement
    - explained_var: how much variance this axis explains
    - rank: 1 = most common-ground, 6 = most polarization
    """
    results = {}

    # Check if cross-cutting scores are available
    cc_available = any(
        not np.isnan(v) for v in cross_cutting_scores.values() if v is not None
    )

    if cc_available:
        # Use cross-cutting as primary (lower = more common-ground)
        sorted_factors = sorted(
            cross_cutting_scores.keys(),
            key=lambda f: cross_cutting_scores.get(f, 0) if not np.isnan(cross_cutting_scores.get(f, 0)) else 0
        )
    else:
        # Fall back to polarization score (lower = more common-ground)
        sorted_factors = sorted(
            polarization_scores.keys(),
            key=lambda f: polarization_scores.get(f, 0) if not np.isnan(polarization_scores.get(f, 0)) else 0
        )

    for rank, factor in enumerate(sorted_factors, 1):
        cc = cross_cutting_scores.get(factor, np.nan)
        ev = explained_variance.get(factor, np.nan)
        pol = polarization_scores.get(factor, np.nan)

        results[factor] = {
            'cross_cutting': cc,
            'explained_variance': ev,
            'polarization': pol,
            'rank': rank,
            'type': 'common_ground' if rank == 1 else 'polarization'
        }

    return results


def compute_rotation_matrix(
    users: pd.DataFrame,
    user_factor_cols: List[str],
    axis_types: Dict[str, Dict]
) -> np.ndarray:
    """
    Compute an orthogonal rotation matrix that:
    1. Puts the common-ground axis first
    2. Orders remaining axes by polarization (most to least)

    Uses the natural ordering from axis_types.
    """
    n_factors = len(user_factor_cols)

    # Get user factor matrix
    factor_matrix = users[user_factor_cols].dropna().values

    # Sort columns by rank (common-ground first)
    sorted_cols = sorted(user_factor_cols, key=lambda f: axis_types[f]['rank'])
    col_indices = [user_factor_cols.index(c) for c in sorted_cols]

    # Create permutation matrix
    P = np.zeros((n_factors, n_factors))
    for new_idx, old_idx in enumerate(col_indices):
        P[new_idx, old_idx] = 1.0

    return P, sorted_cols


def select_canonical_basis(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    notes: pd.DataFrame,
    user_factor_cols: List[str],
    note_factor_cols: List[str],
) -> Tuple[Dict, Dict[str, float]]:
    """
    Defensible procedure for axis selection.

    This implements a data-driven approach to selecting a canonical basis
    that is objective and reproducible:

    1. Compute polarization score for each axis
    2. common-ground axis = argmin(polarization)
    3. polarization axes = sorted by polarization (ascending)
    4. Freeze as canonical ordering

    Returns:
        canonical_order: Dict with axis assignments
        polarization_scores: Dict mapping factor name to polarization score
    """
    # Compute polarization scores using the existing function
    polarization_scores = compute_polarization_score(
        ratings, users, notes, user_factor_cols, note_factor_cols
    )

    # Filter out NaN values and sort by polarization (ascending)
    valid_scores = {k: v for k, v in polarization_scores.items()
                    if v is not None and not np.isnan(v)}

    if not valid_scores:
        raise ValueError("No valid polarization scores computed")

    ranked = sorted(valid_scores.items(), key=lambda x: x[1])

    # Map factor names to indices
    factor_indices = {col: user_factor_cols.index(col) + 1 for col in user_factor_cols}

    canonical_order = {
        'common_ground_axis': factor_indices.get(ranked[0][0], 1),
        'common_ground_factor': ranked[0][0],
        'main_polarization_axis': factor_indices.get(ranked[-1][0], len(user_factor_cols)),
        'main_polarization_factor': ranked[-1][0],
        'axis_order': [factor_indices.get(r[0], i+1) for i, r in enumerate(ranked)],
        'factor_order': [r[0] for r in ranked],
        'polarization_ranking': {r[0]: i+1 for i, r in enumerate(ranked)},
    }

    return canonical_order, polarization_scores


def compute_data_hash(ratings: pd.DataFrame, users: pd.DataFrame, notes: pd.DataFrame) -> str:
    """Compute a hash of the input data for reproducibility tracking."""
    hash_input = f"{len(ratings)}_{len(users)}_{len(notes)}"

    # Add some data statistics
    if noteIdKey in ratings.columns:
        hash_input += f"_{ratings[noteIdKey].nunique()}"
    if raterParticipantIdKey in ratings.columns:
        hash_input += f"_{ratings[raterParticipantIdKey].nunique()}"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def save_canonical_basis(
    canonical_order: Dict,
    polarization_scores: Dict[str, float],
    output_path: str,
    data_hash: str,
):
    """
    Save the canonical basis to a JSON file for reproducibility.

    This frozen file serves as the authoritative axis ordering for
    downstream analysis and ensures results are reproducible.
    """
    output = {
        'canonical_order': canonical_order,
        'polarization_scores': {k: float(v) if v is not None and not np.isnan(v) else None
                                for k, v in polarization_scores.items()},
        'metadata': {
            'computed_on': datetime.now().isoformat(),
            'data_hash': data_hash,
            'version': '1.0',
            'description': 'Canonical factor basis for 6-factor model. '
                          'Axes ordered by polarization score (low=common ground, high=polarization).',
        },
        'interpretation_guide': {
            'common_ground_axis': 'Lowest polarization - user-note alignment has LEAST effect on rating deviation',
            'main_polarization_axis': 'Highest polarization (F6) - users at opposite ends systematically disagree',
            'usage': 'Use this ordering for consistent factor interpretation across analyses',
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved canonical basis to: {output_path}")
    return output


def load_canonical_basis(input_path: str) -> Dict:
    """Load a previously saved canonical basis."""
    with open(input_path, 'r') as f:
        return json.load(f)


def apply_varimax_rotation(
    factor_matrix: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply varimax rotation to maximize simple structure.

    Returns:
        rotated_matrix: Rotated factor matrix
        rotation_matrix: The orthogonal rotation matrix R
    """
    n_samples, n_factors = factor_matrix.shape

    # Initialize rotation matrix as identity
    R = np.eye(n_factors)

    # Normalize columns
    h = np.sqrt(np.sum(factor_matrix ** 2, axis=1, keepdims=True))
    h[h == 0] = 1  # Avoid division by zero
    A = factor_matrix / h

    for iteration in range(max_iter):
        # Rotated loadings
        L = A @ R

        # Varimax criterion gradient
        L2 = L ** 2
        L3 = L ** 3

        # Compute the SVD of the gradient
        u = L3 - (gamma / n_samples) * L @ np.diag(np.sum(L2, axis=0))
        U, S, Vt = np.linalg.svd(A.T @ u)

        # New rotation matrix
        R_new = U @ Vt

        # Check convergence
        if np.max(np.abs(R_new - R)) < tol:
            break

        R = R_new

    rotated = factor_matrix @ R
    return rotated, R


def print_interpretation_report(
    axis_types: Dict[str, Dict],
    cross_cutting_scores: Dict[str, float],
    explained_variance: Dict[str, float],
    polarization_scores: Dict[str, float],
    factor_variances: Dict[str, float]
):
    """Print a human-readable interpretation report."""

    print("\n" + "=" * 80)
    print("FACTOR INTERPRETATION REPORT")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("AXIS CLASSIFICATION (sorted by cross-cutting score: low=consensus, high=polarization)")
    print("-" * 80)
    print(f"{'Factor':<25} {'Type':<14} {'CrossCut':<10} {'ExplVar':<10} {'Polariz':<10}")
    print("-" * 80)

    sorted_factors = sorted(axis_types.keys(), key=lambda f: axis_types[f]['rank'])

    for factor in sorted_factors:
        info = axis_types[factor]
        axis_type = "COMMON GND" if info['type'] == 'common_ground' else "POLARIZED"
        cc = f"{info['cross_cutting']:.4f}" if info['cross_cutting'] and not np.isnan(info['cross_cutting']) else "N/A"
        ev = f"{info['explained_variance']:.4f}" if info['explained_variance'] and not np.isnan(info['explained_variance']) else "N/A"
        pol = f"{info['polarization']:.4f}" if info['polarization'] and not np.isnan(info['polarization']) else "N/A"
        print(f"{factor:<25} {axis_type:<14} {cc:<10} {ev:<10} {pol:<10}")

    print("\n" + "-" * 80)
    print("METRIC DEFINITIONS")
    print("-" * 80)
    print("  CrossCut:  Avg |corr(user_factor, rating)| within notes. Higher = more polarizing.")
    print("  ExplVar:   R² of (user_factor × note_factor) vs rating. Higher = more predictive.")
    print("  Polariz:   |corr(user×note alignment, rating deviation)|. Higher = more divisive.")

    print("\n" + "-" * 80)
    print("INTERPRETATION GUIDE")
    print("-" * 80)

    common_ground = [f for f, info in axis_types.items() if info['type'] == 'common_ground']
    polarization = [f for f, info in axis_types.items() if info['type'] == 'polarization']

    if common_ground:
        cg = common_ground[0]
        cg_info = axis_types[cg]
        pol = cg_info['polarization']
        ev = cg_info['explained_variance']
        pol_str = f"{pol:.4f}" if pol and not np.isnan(pol) else "N/A"
        ev_str = f"{ev:.4f}" if ev and not np.isnan(ev) else "N/A"
        print(f"\n★ COMMON-GROUND AXIS: {cg}")
        print(f"    Polarization score: {pol_str} (lowest)")
        print(f"    Explained variance: {ev_str}")
        print("    - User-note alignment on this axis has LEAST effect on rating deviation")
        print("    - This is more about individual quality assessment than group disagreement")

    if polarization:
        print(f"\n⚡ POLARIZATION AXES (ordered by polarization score, highest first):")
        # Sort polarization axes by their polarization score descending
        pol_sorted = sorted(polarization, key=lambda f: axis_types[f]['polarization'] or 0, reverse=True)
        for i, f in enumerate(pol_sorted, 1):
            f_info = axis_types[f]
            pol = f_info['polarization']
            ev = f_info['explained_variance']
            pol_str = f"{pol:.4f}" if pol and not np.isnan(pol) else "N/A"
            ev_str = f"{ev:.4f}" if ev and not np.isnan(ev) else "N/A"
            print(f"    {i}. {f} (polariz: {pol_str}, explVar: {ev_str})")
        print("    - Users at opposite ends systematically disagree on same notes")
        print("    - These capture ideological, topical, or perspective divides")
        print("    - Notes need cross-cutting support across these dimensions to be CRH")

    print("\n" + "-" * 80)
    print("RECOMMENDED ROTATION (reorder for interpretability)")
    print("-" * 80)
    print("\nNew basis: Factor 1 = least polarizing, Factor 6 = most polarizing:")

    # Re-sort by polarization ascending (common ground first, most polarizing last)
    sorted_by_pol = sorted(
        sorted_factors,
        key=lambda f: axis_types[f]['polarization'] if axis_types[f]['polarization'] and not np.isnan(axis_types[f]['polarization']) else 0
    )

    for i, factor in enumerate(sorted_by_pol, 1):
        info = axis_types[factor]
        pol = info['polarization']
        ev = info['explained_variance']
        if i == 1:
            label = "Common Ground / Quality"
        elif pol and pol > 0.2:
            label = "MAJOR Polarization"
        elif pol and pol > 0.1:
            label = "Moderate Polarization"
        else:
            label = "Minor Polarization"
        pol_str = f"(pol={pol:.4f})" if pol and not np.isnan(pol) else ""
        print(f"  Rotated Factor {i} <- {factor:<25} [{label}] {pol_str}")

    print("\n" + "=" * 80)


def run_factor_interpretation(
    notes_path: str,
    users_path: str,
    ratings_dir: str,
    sample_ratings: Optional[float] = None
) -> Dict:
    """Run full factor interpretation analysis."""

    # Load data
    print("Loading data...")
    notes = pd.read_csv(notes_path, sep='\t')
    users = pd.read_csv(users_path, sep='\t')
    print(f"  Loaded {len(notes):,} notes, {len(users):,} users")

    # Load ratings
    print("Loading ratings...")
    ratings_path = Path(ratings_dir)
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path, sep='\t')
    else:
        rating_files = list(ratings_path.glob("ratings-*.tsv"))
        if not rating_files:
            rating_files = list(ratings_path.glob("ratings*.tsv"))
        ratings_list = [pd.read_csv(f, sep='\t') for f in sorted(rating_files)]
        ratings = pd.concat(ratings_list, ignore_index=True)
    print(f"  Loaded {len(ratings):,} ratings")

    # Sample if requested (for faster iteration)
    if sample_ratings and sample_ratings < 1.0:
        ratings = ratings.sample(frac=sample_ratings, random_state=42)
        print(f"  Sampled to {len(ratings):,} ratings")

    # Find factor columns
    note_factor_cols = find_factor_columns(notes, 'note')
    user_factor_cols = find_factor_columns(users, 'user')
    print(f"  Found {len(note_factor_cols)} note factors, {len(user_factor_cols)} user factors")

    if len(user_factor_cols) < 2:
        raise ValueError("Need at least 2 factors for interpretation analysis")

    # Compute cross-cutting scores (main metric)
    print("\nComputing cross-cutting scores (this may take a while)...")
    cross_cutting_scores = compute_cross_cutting_score(
        ratings, users, notes, user_factor_cols, note_factor_cols
    )

    # Compute explained variance
    print("Computing explained variance by axis...")
    explained_variance = compute_explained_variance_by_axis(
        ratings, users, notes, user_factor_cols, note_factor_cols
    )

    # Compute polarization scores (secondary metric)
    print("Computing polarization scores...")
    polarization_scores = compute_polarization_score(
        ratings, users, notes, user_factor_cols, note_factor_cols
    )

    # Identify axis types using cross-cutting as primary metric
    axis_types = identify_axis_types(cross_cutting_scores, explained_variance, polarization_scores)

    # Compute factor variances for reference
    factor_variances = {col: users[col].var() for col in user_factor_cols}

    # Print report
    print_interpretation_report(
        axis_types, cross_cutting_scores, explained_variance, polarization_scores, factor_variances
    )

    return {
        'axis_types': axis_types,
        'cross_cutting_scores': cross_cutting_scores,
        'explained_variance': explained_variance,
        'polarization_scores': polarization_scores,
        'factor_variances': factor_variances,
        'user_factor_cols': user_factor_cols,
        'note_factor_cols': note_factor_cols,
    }


def apply_rotation(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
    rotation_order: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply rotation by reordering factors according to polarization ranking.

    Returns copies of notes and users with new rotated factor columns.
    """
    notes_rotated = notes.copy()
    users_rotated = users.copy()

    # Create mapping from user factor index to note factor
    # Both lists should be in the same order (factor 1, 2, 3, etc.)
    user_to_note = dict(zip(user_factor_cols, note_factor_cols))

    print(f"  Rotation mapping:")
    for new_idx, old_user_col in enumerate(rotation_order, 1):
        old_note_col = user_to_note.get(old_user_col)
        print(f"    Factor {new_idx}: user={old_user_col}, note={old_note_col}")

        # Create new column names
        new_user_col = f'rotatedRaterFactor{new_idx}'
        new_note_col = f'rotatedNoteFactor{new_idx}'

        # Copy values
        if old_user_col in users.columns:
            users_rotated[new_user_col] = users[old_user_col].copy()
        else:
            print(f"    WARNING: {old_user_col} not found in users")

        if old_note_col and old_note_col in notes.columns:
            notes_rotated[new_note_col] = notes[old_note_col].copy()
        else:
            print(f"    WARNING: {old_note_col} not found in notes")

    return notes_rotated, users_rotated


def analyze_major_polarization_axis(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    major_axis_col: str,
    note_factor_col: str,
) -> Dict:
    """
    Deep analysis of the major polarization axis (Factor 6).

    Investigates what this axis captures by correlating with:
    - Note topics
    - Note classification (misleading vs not)
    - Rating patterns
    - User characteristics
    """
    results = {
        'axis': major_axis_col,
        'note_axis': note_factor_col,
    }

    print("\n" + "=" * 80)
    print(f"DEEP ANALYSIS: {major_axis_col} (Major Polarization Axis)")
    print("=" * 80)

    # 1. Distribution analysis
    print("\n" + "-" * 80)
    print("1. DISTRIBUTION ANALYSIS")
    print("-" * 80)

    user_values = users[major_axis_col].dropna()
    note_values = notes[note_factor_col].dropna()

    print(f"\nUser factor distribution:")
    print(f"  Mean: {user_values.mean():.4f}")
    print(f"  Std:  {user_values.std():.4f}")
    print(f"  Min:  {user_values.min():.4f}")
    print(f"  Max:  {user_values.max():.4f}")
    print(f"  Skew: {user_values.skew():.4f}")

    # Percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pct_values = np.percentile(user_values, percentiles)
    print(f"  Percentiles: {dict(zip(percentiles, [f'{v:.3f}' for v in pct_values]))}")

    print(f"\nNote factor distribution:")
    print(f"  Mean: {note_values.mean():.4f}")
    print(f"  Std:  {note_values.std():.4f}")
    print(f"  Min:  {note_values.min():.4f}")
    print(f"  Max:  {note_values.max():.4f}")

    results['user_mean'] = user_values.mean()
    results['user_std'] = user_values.std()
    results['note_mean'] = note_values.mean()
    results['note_std'] = note_values.std()

    # 2. Correlation with note topics
    print("\n" + "-" * 80)
    print("2. CORRELATION WITH NOTE TOPICS")
    print("-" * 80)

    if 'noteTopic' in notes.columns:
        topic_means = notes.groupby('noteTopic')[note_factor_col].agg(['mean', 'std', 'count'])
        topic_means = topic_means.sort_values('mean')

        print(f"\nNote factor by topic (sorted by mean):")
        print(f"{'Topic':<40} {'Mean':>10} {'Std':>10} {'Count':>10}")
        print("-" * 70)
        for topic, row in topic_means.iterrows():
            if row['count'] >= 100:  # Only show topics with enough notes
                print(f"{str(topic):<40} {row['mean']:>10.4f} {row['std']:>10.4f} {row['count']:>10.0f}")

        results['topic_analysis'] = topic_means.to_dict()
    else:
        print("  No noteTopic column found")

    # 3. Correlation with note classification
    print("\n" + "-" * 80)
    print("3. CORRELATION WITH NOTE CLASSIFICATION")
    print("-" * 80)

    if 'classification' in notes.columns:
        class_means = notes.groupby('classification')[note_factor_col].agg(['mean', 'std', 'count'])
        print(f"\nNote factor by classification:")
        for cls, row in class_means.iterrows():
            print(f"  {cls}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={row['count']:.0f}")
        results['classification_analysis'] = class_means.to_dict()

    # 4. Correlation with note status
    print("\n" + "-" * 80)
    print("4. CORRELATION WITH NOTE STATUS")
    print("-" * 80)

    status_cols = ['coreRatingStatus', 'finalRatingStatus', 'currentLabelKey']
    for status_col in status_cols:
        if status_col in notes.columns:
            status_means = notes.groupby(status_col)[note_factor_col].agg(['mean', 'std', 'count'])
            print(f"\nNote factor by {status_col}:")
            for status, row in status_means.iterrows():
                if row['count'] >= 50:
                    print(f"  {status}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={row['count']:.0f}")

    # 5. Extreme users analysis
    print("\n" + "-" * 80)
    print("5. EXTREME USERS ANALYSIS")
    print("-" * 80)

    # Get users at extremes
    low_threshold = np.percentile(user_values, 5)
    high_threshold = np.percentile(user_values, 95)

    low_users = users[users[major_axis_col] < low_threshold]
    high_users = users[users[major_axis_col] > high_threshold]

    print(f"\nUsers at extremes (bottom 5% vs top 5%):")
    print(f"  Low end (<{low_threshold:.3f}): {len(low_users):,} users")
    print(f"  High end (>{high_threshold:.3f}): {len(high_users):,} users")

    # Compare other characteristics
    other_user_cols = [c for c in users.columns if 'Factor' in c and c != major_axis_col]
    if other_user_cols:
        print(f"\n  Other factor means for extreme users:")
        print(f"  {'Factor':<30} {'Low End':>12} {'High End':>12} {'Diff':>12}")
        print("  " + "-" * 66)
        for col in other_user_cols[:5]:  # Limit to 5
            low_mean = low_users[col].mean()
            high_mean = high_users[col].mean()
            diff = high_mean - low_mean
            print(f"  {col:<30} {low_mean:>12.4f} {high_mean:>12.4f} {diff:>+12.4f}")

    # 6. Rating behavior analysis
    print("\n" + "-" * 80)
    print("6. RATING BEHAVIOR BY AXIS POSITION")
    print("-" * 80)

    # Merge ratings with user factors
    ratings_with_users = ratings.merge(
        users[[raterParticipantIdKey, major_axis_col]],
        on=raterParticipantIdKey,
        how='inner'
    )

    # Get helpful values
    ratings_with_users['helpful_num'] = get_helpful_numeric(ratings_with_users)

    # Bin users by factor position
    ratings_with_users['user_bin'] = pd.qcut(
        ratings_with_users[major_axis_col],
        q=5,
        labels=['Very Low', 'Low', 'Middle', 'High', 'Very High']
    )

    bin_stats = ratings_with_users.groupby('user_bin')['helpful_num'].agg(['mean', 'std', 'count'])
    print(f"\nAverage helpfulness rating by user position on axis:")
    print(f"  {'Position':<15} {'Mean Rating':>12} {'Std':>10} {'N Ratings':>12}")
    print("  " + "-" * 50)
    for pos, row in bin_stats.iterrows():
        print(f"  {pos:<15} {row['mean']:>12.4f} {row['std']:>10.4f} {row['count']:>12,.0f}")

    results['rating_by_position'] = bin_stats.to_dict()

    # 7. Disagreement analysis
    print("\n" + "-" * 80)
    print("7. DISAGREEMENT PATTERN")
    print("-" * 80)

    # For notes, compare ratings from low vs high users
    # Merge with note factors
    ratings_with_notes = ratings_with_users.merge(
        notes[[noteIdKey, note_factor_col]],
        on=noteIdKey,
        how='inner'
    )

    # Compute alignment
    ratings_with_notes['alignment'] = (
        ratings_with_notes[major_axis_col] * ratings_with_notes[note_factor_col]
    )

    # Bin by alignment
    ratings_with_notes['alignment_bin'] = pd.qcut(
        ratings_with_notes['alignment'],
        q=5,
        labels=['Strong Oppose', 'Oppose', 'Neutral', 'Align', 'Strong Align'],
        duplicates='drop'
    )

    alignment_stats = ratings_with_notes.groupby('alignment_bin')['helpful_num'].agg(['mean', 'count'])
    print(f"\nRating by user-note alignment on this axis:")
    print(f"  {'Alignment':<15} {'Mean Rating':>12} {'N':>12}")
    print("  " + "-" * 40)
    for alignment, row in alignment_stats.iterrows():
        print(f"  {alignment:<15} {row['mean']:>12.4f} {row['count']:>12,.0f}")

    alignment_spread = alignment_stats['mean'].max() - alignment_stats['mean'].min()
    print(f"\n  Rating spread (aligned vs opposed): {alignment_spread:.4f}")
    print(f"  This measures how much this axis drives disagreement")

    results['alignment_spread'] = alignment_spread

    print("\n" + "=" * 80)

    return results


def save_rotated_factors(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    output_dir: str,
    rotation_order: List[str],
    note_factor_cols: List[str],
    user_factor_cols: List[str],
):
    """Save rotated factors to TSV files (only rows with non-null factors)."""

    notes_rotated, users_rotated = apply_rotation(
        notes, users, note_factor_cols, user_factor_cols, rotation_order
    )

    # Get just the rotated columns plus IDs
    rotated_note_cols = [noteIdKey] + [c for c in notes_rotated.columns if 'rotated' in c.lower()]
    rotated_user_cols = [raterParticipantIdKey] + [c for c in users_rotated.columns if 'rotated' in c.lower()]

    notes_output = notes_rotated[rotated_note_cols]
    users_output = users_rotated[rotated_user_cols]

    # Filter to only rows with at least one non-null rotated factor
    factor_note_cols = [c for c in rotated_note_cols if 'rotated' in c.lower()]
    factor_user_cols = [c for c in rotated_user_cols if 'rotated' in c.lower()]

    notes_output = notes_output[notes_output[factor_note_cols].notna().any(axis=1)]
    users_output = users_output[users_output[factor_user_cols].notna().any(axis=1)]

    notes_path = Path(output_dir) / 'rotated_note_factors.tsv'
    users_path = Path(output_dir) / 'rotated_user_factors.tsv'

    notes_output.to_csv(notes_path, sep='\t', index=False)
    users_output.to_csv(users_path, sep='\t', index=False)

    print(f"\nSaved rotated factors to:")
    print(f"  {notes_path} ({len(notes_output):,} notes with factors)")
    print(f"  {users_path} ({len(users_output):,} users with factors)")

    return notes_path, users_path


def main():
    parser = argparse.ArgumentParser(
        description='Rotate factors into interpretable common-ground vs polarization axes'
    )
    parser.add_argument('--notes', required=True, help='Path to scored notes TSV')
    parser.add_argument('--users', required=True, help='Path to helpfulness scores TSV')
    parser.add_argument('--ratings', required=True, help='Path to ratings directory or file')
    parser.add_argument('--sample', type=float, default=None,
                        help='Sample fraction of ratings (0-1) for faster iteration')
    parser.add_argument('--analyze-major', action='store_true',
                        help='Deep analysis of the major polarization axis')
    parser.add_argument('--save-rotated', type=str, default=None,
                        help='Output directory to save rotated factors')
    parser.add_argument('--save-canonical-basis', type=str, default=None,
                        help='Path to save canonical basis JSON (e.g., data/canonical_factor_basis.json)')
    args = parser.parse_args()

    results = run_factor_interpretation(
        args.notes, args.users, args.ratings, args.sample
    )

    # Save canonical basis if requested
    if args.save_canonical_basis:
        print("\n" + "=" * 60)
        print("COMPUTING CANONICAL BASIS")
        print("=" * 60)

        # Load data for canonical basis computation
        notes = pd.read_csv(args.notes, sep='\t')
        users = pd.read_csv(args.users, sep='\t')

        # Load ratings
        ratings_path = Path(args.ratings)
        if ratings_path.is_file():
            ratings = pd.read_csv(ratings_path, sep='\t')
        else:
            rating_files = list(ratings_path.glob("ratings-*.tsv"))
            if not rating_files:
                rating_files = list(ratings_path.glob("ratings*.tsv"))
            ratings_list = [pd.read_csv(f, sep='\t') for f in sorted(rating_files)]
            ratings = pd.concat(ratings_list, ignore_index=True)

        # Sample if needed
        if args.sample and args.sample < 1.0:
            ratings = ratings.sample(frac=args.sample, random_state=42)

        # Compute canonical basis
        canonical_order, polarization_scores = select_canonical_basis(
            ratings, users, notes,
            results['user_factor_cols'],
            results['note_factor_cols']
        )

        # Compute data hash for reproducibility
        data_hash = compute_data_hash(ratings, users, notes)

        # Save to file
        saved_basis = save_canonical_basis(
            canonical_order, polarization_scores,
            args.save_canonical_basis, data_hash
        )

        print("\nCanonical basis summary:")
        print(f"  Common ground axis: Factor {canonical_order['common_ground_axis']} ({canonical_order['common_ground_factor']})")
        print(f"  Main polarization axis: Factor {canonical_order['main_polarization_axis']} ({canonical_order['main_polarization_factor']})")
        print(f"  Axis order (low→high pol): {canonical_order['axis_order']}")

        results['canonical_basis'] = saved_basis

    # Deep analysis of major polarization axis
    if args.analyze_major:
        # Find the most polarizing axis
        axis_types = results['axis_types']
        major_axis = max(axis_types.keys(), key=lambda f: axis_types[f]['polarization'] or 0)

        # Find corresponding note factor
        user_factor_cols = results['user_factor_cols']
        note_factor_cols = results['note_factor_cols']
        major_idx = user_factor_cols.index(major_axis)
        major_note_axis = note_factor_cols[major_idx]

        # Load full data for analysis
        print("\nLoading full data for deep analysis...")
        notes = pd.read_csv(args.notes, sep='\t')
        users = pd.read_csv(args.users, sep='\t')

        # Load ratings (sample for speed)
        ratings_path = Path(args.ratings)
        if ratings_path.is_file():
            ratings = pd.read_csv(ratings_path, sep='\t')
        else:
            rating_files = list(ratings_path.glob("ratings-*.tsv"))
            if not rating_files:
                rating_files = list(ratings_path.glob("ratings*.tsv"))
            ratings_list = [pd.read_csv(f, sep='\t') for f in sorted(rating_files)]
            ratings = pd.concat(ratings_list, ignore_index=True)

        # Sample for analysis speed
        if len(ratings) > 5_000_000:
            ratings = ratings.sample(n=5_000_000, random_state=42)

        analysis_results = analyze_major_polarization_axis(
            notes, users, ratings, major_axis, major_note_axis
        )
        results['major_axis_analysis'] = analysis_results

    # Save rotated factors
    if args.save_rotated:
        # Compute rotation order
        axis_types = results['axis_types']
        rotation_order = sorted(
            axis_types.keys(),
            key=lambda f: axis_types[f]['polarization'] if axis_types[f]['polarization'] and not np.isnan(axis_types[f]['polarization']) else 0
        )

        notes = pd.read_csv(args.notes, sep='\t')
        users = pd.read_csv(args.users, sep='\t')

        save_rotated_factors(
            notes, users, args.save_rotated,
            rotation_order,
            results['note_factor_cols'],
            results['user_factor_cols']
        )


if __name__ == "__main__":
    main()
