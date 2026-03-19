#!/usr/bin/env python3
"""
Sanity Check Script for 6-Factor Matrix Factorization Model

Performs three key sanity checks:
A. Held-out predictive performance (time-based train/test split)
B. Factor usage / per-dimension variance
C. Identifiability / rotation awareness (covariance structure)

Usage:
    python sanity_check_6factor.py \
        --notes <scored_notes.tsv> \
        --users <helpfulness_scores.tsv> \
        --ratings <ratings_dir>
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    helpfulnessLevelKey,
    createdAtMillisKey,
)

HELPFULNESS_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def find_factor_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Find factor columns for notes or users.

    For notes: coreNoteFactor1, internalNoteFactor2, ...
    For users: coreRaterFactor1, internalRaterFactor2, ...
    """
    factor_cols = []

    if prefix == 'note':
        # Check for coreNoteFactor1 first
        if 'coreNoteFactor1' in df.columns:
            factor_cols.append('coreNoteFactor1')
        # Then look for internalNoteFactor columns
        for col in sorted(df.columns):
            if col.startswith('internalNoteFactor'):
                factor_cols.append(col)
    elif prefix == 'user':
        # Check for coreRaterFactor1 first
        if 'coreRaterFactor1' in df.columns:
            factor_cols.append('coreRaterFactor1')
        # Then look for internalRaterFactor columns
        for col in sorted(df.columns):
            if col.startswith('internalRaterFactor'):
                factor_cols.append(col)

    return factor_cols


def load_scored_notes(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load scored notes with factor columns."""
    df = pd.read_csv(path, sep='\t', low_memory=False)
    factor_cols = find_factor_columns(df, 'note')
    return df, factor_cols


def load_helpfulness_scores(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load user helpfulness scores with factor columns."""
    df = pd.read_csv(path, sep='\t', low_memory=False)
    factor_cols = find_factor_columns(df, 'user')
    return df, factor_cols


def load_ratings(ratings_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load ratings from file or directory."""
    path = Path(ratings_path)

    if path.is_dir():
        rating_files = sorted(path.glob("ratings*.tsv"))
        dfs = []
        for rf in rating_files:
            print(f"    Loading {rf.name}...")
            dfs.append(pd.read_csv(rf, sep='\t', low_memory=False))
        ratings = pd.concat(dfs, ignore_index=True)
    else:
        ratings = pd.read_csv(path, sep='\t', low_memory=False)

    if sample_size and len(ratings) > sample_size:
        ratings = ratings.sample(n=sample_size, random_state=42)

    return ratings


# ============================================================================
# CHECK A: HELD-OUT PREDICTIVE PERFORMANCE
# ============================================================================

def time_based_train_test_split(
    ratings: pd.DataFrame,
    test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings by timestamp - no data leakage."""
    if createdAtMillisKey not in ratings.columns:
        # Fall back to random split with warning
        print("    [WARN] No timestamp column, using random split (potential leakage)")
        n_test = int(len(ratings) * test_fraction)
        shuffled = ratings.sample(frac=1, random_state=42)
        return shuffled.iloc[n_test:], shuffled.iloc[:n_test]

    # Sort by time
    sorted_ratings = ratings.sort_values(createdAtMillisKey)
    split_idx = int(len(sorted_ratings) * (1 - test_fraction))

    return sorted_ratings.iloc[:split_idx], sorted_ratings.iloc[split_idx:]


def compute_predictions(
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
) -> np.ndarray:
    """
    Compute model predictions: r_hat = mu + i_u + i_n + f_u · f_n
    """
    # Merge factors
    note_cols = [noteIdKey, 'coreNoteIntercept'] + note_factor_cols
    note_cols = [c for c in note_cols if c in notes.columns]

    user_cols = [raterParticipantIdKey, 'coreRaterIntercept'] + user_factor_cols
    user_cols = [c for c in user_cols if c in users.columns]

    merged = ratings.merge(notes[note_cols], on=noteIdKey, how='inner')
    merged = merged.merge(users[user_cols], on=raterParticipantIdKey, how='inner')

    if len(merged) == 0:
        return np.array([]), np.array([])

    # Compute predictions
    predictions = np.zeros(len(merged))

    # Global intercept (assume 0.5 as default)
    predictions += 0.5

    # User intercept
    if 'coreRaterIntercept' in merged.columns:
        predictions += merged['coreRaterIntercept'].fillna(0).values

    # Note intercept
    if 'coreNoteIntercept' in merged.columns:
        predictions += merged['coreNoteIntercept'].fillna(0).values

    # Factor dot product
    n_factors = min(len(note_factor_cols), len(user_factor_cols))
    for i in range(n_factors):
        note_f = merged[note_factor_cols[i]].fillna(0).values
        user_f = merged[user_factor_cols[i]].fillna(0).values
        predictions += note_f * user_f

    # Get actual values
    if helpfulnessLevelKey in merged.columns:
        actuals = merged[helpfulnessLevelKey].map(HELPFULNESS_MAP).values
    elif 'helpful' in merged.columns:
        actuals = merged['helpful'].values
    else:
        actuals = np.full(len(merged), np.nan)

    return predictions, actuals


def check_predictive_performance(
    ratings: pd.DataFrame,
    notes: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
) -> Dict:
    """
    Check A: Held-out predictive performance.
    """
    results = {
        'check': 'A. Predictive Performance',
        'pass': True,
        'messages': [],
    }

    # Time-based split
    train_ratings, test_ratings = time_based_train_test_split(ratings, test_fraction=0.2)
    results['n_train'] = len(train_ratings)
    results['n_test'] = len(test_ratings)

    # Compute predictions on test set
    predictions, actuals = compute_predictions(
        test_ratings, notes, users, note_factor_cols, user_factor_cols
    )

    # Filter valid predictions
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    results['n_valid_predictions'] = len(predictions)

    if len(predictions) < 100:
        results['pass'] = False
        results['messages'].append(f"Too few valid predictions: {len(predictions)}")
        return results

    # RMSE
    rmse = np.sqrt(((predictions - actuals) ** 2).mean())
    results['rmse'] = rmse

    if rmse < 0.5:
        results['messages'].append(f"[PASS] RMSE: {rmse:.4f} (threshold: < 0.5)")
    else:
        results['pass'] = False
        results['messages'].append(f"[FAIL] RMSE: {rmse:.4f} (threshold: < 0.5)")

    # Log loss for binary helpful
    from sklearn.metrics import log_loss as sklearn_log_loss
    binary_actuals = (actuals > 0.25).astype(int)
    predictions_clipped = np.clip(predictions, 0.01, 0.99)

    try:
        ll = sklearn_log_loss(binary_actuals, predictions_clipped)
        results['log_loss'] = ll

        if ll < 1.0:
            results['messages'].append(f"[PASS] Log Loss: {ll:.4f} (threshold: < 1.0)")
        else:
            results['pass'] = False
            results['messages'].append(f"[FAIL] Log Loss: {ll:.4f} (threshold: < 1.0)")
    except Exception as e:
        results['messages'].append(f"[WARN] Could not compute log loss: {e}")

    # Calibration (simple slope)
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(predictions, actuals)
        results['calibration_slope'] = slope
        results['calibration_intercept'] = intercept

        if 0.7 < slope < 1.3:
            results['messages'].append(f"[PASS] Calibration slope: {slope:.3f} (ideal: 1.0)")
        else:
            results['messages'].append(f"[WARN] Calibration slope: {slope:.3f} (ideal: 1.0)")
    except Exception as e:
        results['messages'].append(f"[WARN] Could not compute calibration: {e}")

    return results


# ============================================================================
# CHECK B: FACTOR USAGE / PER-DIMENSION VARIANCE
# ============================================================================

def check_factor_variance(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
    min_variance_ratio: float = 0.1
) -> Dict:
    """
    Check B: Factor usage - ensure no dimensions collapsed.
    """
    results = {
        'check': 'B. Factor Variance',
        'pass': True,
        'messages': [],
    }

    # Note factor variances
    note_variances = {}
    for col in note_factor_cols:
        if col in notes.columns:
            var = notes[col].dropna().var()
            note_variances[col] = var

    results['note_variances'] = note_variances

    if note_variances:
        max_var = max(note_variances.values())
        min_var = min(note_variances.values())
        ratio = min_var / max_var if max_var > 0 else 0
        results['note_variance_ratio'] = ratio

        collapsed = [k for k, v in note_variances.items() if v < max_var * min_variance_ratio]
        results['note_collapsed_dims'] = collapsed

        if ratio > min_variance_ratio:
            results['messages'].append(f"[PASS] Note variance ratio: {ratio:.3f} (threshold: > {min_variance_ratio})")
        else:
            results['pass'] = False
            results['messages'].append(f"[FAIL] Note variance ratio: {ratio:.3f} (threshold: > {min_variance_ratio})")

        if collapsed:
            results['messages'].append(f"[WARN] Collapsed note dimensions: {collapsed}")

        # Show individual variances
        results['messages'].append("  Note factor variances:")
        for col, var in sorted(note_variances.items()):
            results['messages'].append(f"    {col}: {var:.4f}")

    # User factor variances
    user_variances = {}
    for col in user_factor_cols:
        if col in users.columns:
            var = users[col].dropna().var()
            user_variances[col] = var

    results['user_variances'] = user_variances

    if user_variances:
        max_var = max(user_variances.values())
        min_var = min(user_variances.values())
        ratio = min_var / max_var if max_var > 0 else 0
        results['user_variance_ratio'] = ratio

        collapsed = [k for k, v in user_variances.items() if v < max_var * min_variance_ratio]
        results['user_collapsed_dims'] = collapsed

        if ratio > min_variance_ratio:
            results['messages'].append(f"[PASS] User variance ratio: {ratio:.3f} (threshold: > {min_variance_ratio})")
        else:
            results['pass'] = False
            results['messages'].append(f"[FAIL] User variance ratio: {ratio:.3f} (threshold: > {min_variance_ratio})")

        if collapsed:
            results['messages'].append(f"[WARN] Collapsed user dimensions: {collapsed}")

        # Show individual variances
        results['messages'].append("  User factor variances:")
        for col, var in sorted(user_variances.items()):
            results['messages'].append(f"    {col}: {var:.4f}")

    return results


# ============================================================================
# CHECK C: IDENTIFIABILITY / ROTATION AWARENESS
# ============================================================================

def check_factor_orthogonality(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
    max_correlation: float = 0.3
) -> Dict:
    """
    Check C: Factor covariance structure.
    """
    results = {
        'check': 'C. Identifiability / Covariance',
        'pass': True,
        'messages': [],
    }

    # Note factor covariance
    if note_factor_cols:
        note_factors = notes[note_factor_cols].dropna()
        if len(note_factors) > 100:
            note_cov = note_factors.cov().values
            note_corr = note_factors.corr().values

            # Get off-diagonal max
            n = len(note_factor_cols)
            off_diag_mask = ~np.eye(n, dtype=bool)
            off_diag_corr = np.abs(note_corr[off_diag_mask])
            max_off_diag = off_diag_corr.max() if len(off_diag_corr) > 0 else 0

            results['note_off_diagonal_max'] = max_off_diag
            results['note_correlation_matrix'] = note_corr.tolist()

            if max_off_diag < max_correlation:
                results['messages'].append(f"[PASS] Note off-diagonal max correlation: {max_off_diag:.3f} (threshold: < {max_correlation})")
            else:
                results['messages'].append(f"[WARN] Note off-diagonal max correlation: {max_off_diag:.3f} (threshold: < {max_correlation})")

            # Condition number
            eigenvalues = np.linalg.eigvalsh(note_cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                cond_num = eigenvalues.max() / eigenvalues.min()
                results['note_condition_number'] = cond_num

                if cond_num < 10:
                    results['messages'].append(f"[PASS] Note condition number: {cond_num:.2f} (threshold: < 10)")
                else:
                    results['messages'].append(f"[WARN] Note condition number: {cond_num:.2f} (threshold: < 10)")

            # Eigenvalues
            results['messages'].append(f"  Note covariance eigenvalues: {sorted(eigenvalues, reverse=True)[:6]}")

    # User factor covariance
    if user_factor_cols:
        user_factors = users[user_factor_cols].dropna()
        if len(user_factors) > 100:
            user_cov = user_factors.cov().values
            user_corr = user_factors.corr().values

            n = len(user_factor_cols)
            off_diag_mask = ~np.eye(n, dtype=bool)
            off_diag_corr = np.abs(user_corr[off_diag_mask])
            max_off_diag = off_diag_corr.max() if len(off_diag_corr) > 0 else 0

            results['user_off_diagonal_max'] = max_off_diag
            results['user_correlation_matrix'] = user_corr.tolist()

            if max_off_diag < max_correlation:
                results['messages'].append(f"[PASS] User off-diagonal max correlation: {max_off_diag:.3f} (threshold: < {max_correlation})")
            else:
                results['messages'].append(f"[WARN] User off-diagonal max correlation: {max_off_diag:.3f} (threshold: < {max_correlation})")

            eigenvalues = np.linalg.eigvalsh(user_cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                cond_num = eigenvalues.max() / eigenvalues.min()
                results['user_condition_number'] = cond_num

                if cond_num < 10:
                    results['messages'].append(f"[PASS] User condition number: {cond_num:.2f} (threshold: < 10)")
                else:
                    results['messages'].append(f"[WARN] User condition number: {cond_num:.2f} (threshold: < 10)")

            results['messages'].append(f"  User covariance eigenvalues: {sorted(eigenvalues, reverse=True)[:6]}")

    # Rotation ambiguity warning
    results['messages'].append("")
    results['messages'].append("[INFO] Factors are rotation/sign ambiguous - don't interpret raw dimensions!")
    results['messages'].append("       The same model fit can be achieved with any orthogonal rotation of factors.")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_checks(
    notes_path: str,
    users_path: str,
    ratings_path: str,
    sample_ratings: int = 500000,
) -> Dict:
    """Run all sanity checks."""

    print("=" * 80)
    print("SANITY CHECK: 6-Factor Matrix Factorization Model")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    notes, note_factor_cols = load_scored_notes(notes_path)
    print(f"  Notes: {len(notes):,} with {len(note_factor_cols)} factors")
    print(f"  Note factors: {note_factor_cols}")

    users, user_factor_cols = load_helpfulness_scores(users_path)
    print(f"  Users: {len(users):,} with {len(user_factor_cols)} factors")
    print(f"  User factors: {user_factor_cols}")

    ratings = load_ratings(ratings_path, sample_size=sample_ratings)
    print(f"  Ratings: {len(ratings):,}")

    results = {
        'n_notes': len(notes),
        'n_users': len(users),
        'n_ratings': len(ratings),
        'n_factors': len(note_factor_cols),
        'checks': [],
    }

    # Check A: Predictive Performance
    print("\n" + "-" * 80)
    print("A. HELD-OUT PREDICTIVE PERFORMANCE")
    print("-" * 80)

    check_a = check_predictive_performance(
        ratings, notes, users, note_factor_cols, user_factor_cols
    )
    results['checks'].append(check_a)
    for msg in check_a['messages']:
        print(f"  {msg}")

    # Check B: Factor Variance
    print("\n" + "-" * 80)
    print("B. FACTOR USAGE / VARIANCE")
    print("-" * 80)

    check_b = check_factor_variance(
        notes, users, note_factor_cols, user_factor_cols
    )
    results['checks'].append(check_b)
    for msg in check_b['messages']:
        print(f"  {msg}")

    # Check C: Identifiability
    print("\n" + "-" * 80)
    print("C. IDENTIFIABILITY / COVARIANCE STRUCTURE")
    print("-" * 80)

    check_c = check_factor_orthogonality(
        notes, users, note_factor_cols, user_factor_cols
    )
    results['checks'].append(check_c)
    for msg in check_c['messages']:
        print(f"  {msg}")

    # Summary
    print("\n" + "=" * 80)
    n_pass = sum(1 for c in results['checks'] if c['pass'])
    n_total = len(results['checks'])

    all_pass = all(c['pass'] for c in results['checks'])
    results['overall_pass'] = all_pass

    if all_pass:
        print(f"OVERALL: {n_pass}/{n_total} checks passed ✓")
    else:
        print(f"OVERALL: {n_pass}/{n_total} checks passed - ISSUES FOUND")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='6-Factor Model Sanity Checks')
    parser.add_argument('--notes', required=True, help='Path to scored_notes.tsv')
    parser.add_argument('--users', required=True, help='Path to helpfulness_scores.tsv')
    parser.add_argument('--ratings', required=True, help='Path to ratings file or directory')
    parser.add_argument('--sample-ratings', type=int, default=500000,
                        help='Sample size for ratings (default: 500000)')

    args = parser.parse_args()

    run_all_checks(
        notes_path=args.notes,
        users_path=args.users,
        ratings_path=args.ratings,
        sample_ratings=args.sample_ratings,
    )


if __name__ == "__main__":
    main()
