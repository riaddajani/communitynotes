#!/usr/bin/env python3
"""
Run Remaining Analyses for Research Paper

Completes three outstanding items:
1. Temporal stability across time splits
2. Re-run calibration on CRH status (note-level, not individual votes)
3. Create canonical_factor_basis.json with frozen axis ordering

Usage:
    python run_remaining_analyses.py \
        --notes data/scored_notes.tsv \
        --users data/helpfulness_scores.tsv \
        --ratings data/ratings/ \
        --outdir data/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    createdAtMillisKey,
    helpfulnessLevelKey,
    finalRatingStatusKey,
)

HELPFULNESS_MAP = {
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


# ============================================================================
# TASK 1: TEMPORAL STABILITY
# ============================================================================

def run_temporal_stability(
    ratings: pd.DataFrame,
    n_splits: int = 3,
    n_factors: int = 6,
    n_epochs: int = 50,
    sample_frac: float = 0.3,
) -> Dict:
    """
    Test if the polarization axis is stable across different time periods.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {'error': 'PyTorch not available'}

    if createdAtMillisKey not in ratings.columns:
        return {'error': f'No {createdAtMillisKey} column found'}

    print(f"\n{'='*60}")
    print("TASK 1: TEMPORAL STABILITY ACROSS TIME SPLITS")
    print(f"{'='*60}")

    # Prepare data
    if helpfulnessLevelKey in ratings.columns:
        ratings = ratings.copy()
        ratings['helpful'] = ratings[helpfulnessLevelKey].map(HELPFULNESS_MAP)
    elif 'helpful' not in ratings.columns:
        return {'error': 'No helpfulness column found'}

    ratings = ratings[[noteIdKey, raterParticipantIdKey, createdAtMillisKey, 'helpful']].dropna()

    # Sort by time and split
    ratings = ratings.sort_values(createdAtMillisKey)
    split_size = len(ratings) // n_splits

    print(f"Total ratings: {len(ratings):,}")
    print(f"Splits: {n_splits}, each ~{split_size:,} ratings")

    user_pol_axis_by_split = {}
    polarization_axes = []

    for split_idx in range(n_splits):
        start_idx = split_idx * split_size
        end_idx = (split_idx + 1) * split_size if split_idx < n_splits - 1 else len(ratings)
        split_ratings = ratings.iloc[start_idx:end_idx].copy()

        # Sample for speed
        if sample_frac < 1.0:
            split_ratings = split_ratings.sample(frac=sample_frac, random_state=42)

        print(f"\n  Split {split_idx + 1}: {len(split_ratings):,} ratings")

        # Build index mappings
        unique_users = split_ratings[raterParticipantIdKey].unique()
        unique_notes = split_ratings[noteIdKey].unique()
        user_to_idx = {u: i for i, u in enumerate(unique_users)}
        note_to_idx = {n: i for i, n in enumerate(unique_notes)}

        split_ratings['user_idx'] = split_ratings[raterParticipantIdKey].map(user_to_idx)
        split_ratings['note_idx'] = split_ratings[noteIdKey].map(note_to_idx)

        n_users = len(user_to_idx)
        n_notes = len(note_to_idx)

        # Train model
        torch.manual_seed(42)

        class SimpleMF(nn.Module):
            def __init__(self):
                super().__init__()
                self.user_factors = nn.Embedding(n_users, n_factors)
                self.note_factors = nn.Embedding(n_notes, n_factors)
                self.user_intercepts = nn.Embedding(n_users, 1)
                self.note_intercepts = nn.Embedding(n_notes, 1)
                self.global_intercept = nn.Parameter(torch.tensor(0.5))
                nn.init.normal_(self.user_factors.weight, std=0.1)
                nn.init.normal_(self.note_factors.weight, std=0.1)

            def forward(self, user_ids, note_ids):
                pred = self.global_intercept
                pred = pred + self.user_intercepts(user_ids).squeeze()
                pred = pred + self.note_intercepts(note_ids).squeeze()
                pred = pred + (self.user_factors(user_ids) * self.note_factors(note_ids)).sum(dim=1)
                return pred

        model = SimpleMF()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        user_ids = torch.tensor(split_ratings['user_idx'].values, dtype=torch.long)
        note_ids = torch.tensor(split_ratings['note_idx'].values, dtype=torch.long)
        targets = torch.tensor(split_ratings['helpful'].values, dtype=torch.float32)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            preds = model(user_ids, note_ids)
            loss = ((preds - targets) ** 2).mean()
            loss += 0.01 * (model.user_factors.weight.pow(2).mean() + model.note_factors.weight.pow(2).mean())
            loss.backward()
            optimizer.step()

        # Extract factors
        with torch.no_grad():
            user_factors = model.user_factors.weight.numpy().copy()
            note_factors = model.note_factors.weight.numpy().copy()

        # Find polarization axis (highest variance in factor interactions)
        sample_ratings = split_ratings.sample(min(10000, len(split_ratings)), random_state=42)
        polarization_scores = []
        for factor_idx in range(n_factors):
            user_f = np.array([user_factors[int(sample_ratings['user_idx'].iloc[i]), factor_idx]
                              for i in range(len(sample_ratings))])
            note_f = np.array([note_factors[int(sample_ratings['note_idx'].iloc[i]), factor_idx]
                              for i in range(len(sample_ratings))])
            interaction = user_f * note_f
            polarization_scores.append(np.var(interaction))

        pol_axis = int(np.argmax(polarization_scores))
        polarization_axes.append(pol_axis)
        print(f"    Polarization axis: factor {pol_axis + 1}")

        # Store user polarization values
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        user_pol_dict = {idx_to_user[i]: user_factors[i, pol_axis] for i in range(len(user_factors))}
        user_pol_axis_by_split[split_idx] = user_pol_dict

    # Find common users
    all_users = [set(d.keys()) for d in user_pol_axis_by_split.values()]
    common_users = list(set.intersection(*all_users))

    print(f"\n  Users appearing in all splits: {len(common_users):,}")

    if len(common_users) < 100:
        return {'error': f'Too few common users across splits: {len(common_users)}'}

    # Compute correlations
    temporal_correlations = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            pol_i = np.array([user_pol_axis_by_split[i][u] for u in common_users])
            pol_j = np.array([user_pol_axis_by_split[j][u] for u in common_users])
            corr = np.corrcoef(pol_i, pol_j)[0, 1]
            abs_corr = abs(corr) if not np.isnan(corr) else 0.0
            temporal_correlations.append(abs_corr)
            print(f"  Split {i+1} vs {j+1}: |r| = {abs_corr:.3f}")

    results = {
        'n_splits': n_splits,
        'n_common_users': len(common_users),
        'polarization_axes_found': polarization_axes,
        'temporal_stability': {
            'description': 'Correlation of users on their respective polarization axes across time periods',
            'mean': float(np.mean(temporal_correlations)),
            'min': float(np.min(temporal_correlations)),
            'max': float(np.max(temporal_correlations)),
            'all_correlations': [float(c) for c in temporal_correlations],
        },
        'pass_threshold': 0.5,
        'pass': float(np.min(temporal_correlations)) > 0.5,
    }

    print(f"\n  RESULT: {'PASS' if results['pass'] else 'FAIL'}")
    print(f"  Mean temporal correlation: {results['temporal_stability']['mean']:.3f}")
    print(f"  Min temporal correlation: {results['temporal_stability']['min']:.3f}")

    return results


# ============================================================================
# TASK 2: CRH CALIBRATION (Note-Level)
# ============================================================================

def run_crh_calibration(
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
) -> Dict:
    """
    Compute calibration on CRH status (note-level), not individual votes.

    The model intercept represents note-level consensus, so we should
    check calibration against note-level outcomes (CRH vs not).
    """
    print(f"\n{'='*60}")
    print("TASK 2: CRH CALIBRATION (Note-Level)")
    print(f"{'='*60}")

    # Get note intercepts
    if 'coreNoteIntercept' not in notes.columns:
        return {'error': 'No coreNoteIntercept column found'}

    if finalRatingStatusKey not in notes.columns:
        return {'error': f'No {finalRatingStatusKey} column found'}

    # Filter to notes with intercepts
    valid_notes = notes[notes['coreNoteIntercept'].notna()].copy()
    print(f"Notes with intercepts: {len(valid_notes):,}")

    # Create binary CRH target
    valid_notes['is_crh'] = (valid_notes[finalRatingStatusKey] == 'CURRENTLY_RATED_HELPFUL').astype(float)

    crh_count = valid_notes['is_crh'].sum()
    print(f"CRH notes: {int(crh_count):,} ({crh_count/len(valid_notes):.1%})")

    # Get predictions (note intercept)
    predictions = valid_notes['coreNoteIntercept'].values
    actuals = valid_notes['is_crh'].values

    # Calibration: regress actual on predicted
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(predictions, actuals)

    print(f"\nCalibration Results:")
    print(f"  Slope: {slope:.4f} (ideal: 1.0)")
    print(f"  Intercept: {intercept:.4f} (ideal: 0.0)")
    print(f"  R-squared: {r_value**2:.4f}")

    # AUC for CRH prediction
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    try:
        crh_auc = roc_auc_score(actuals, predictions)
        print(f"  AUC: {crh_auc:.4f}")
    except:
        crh_auc = None

    # PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(actuals, predictions)
        pr_auc = auc(recall, precision)
        print(f"  PR-AUC: {pr_auc:.4f}")
    except:
        pr_auc = None

    # Calibration by bin
    print("\n  Calibration by intercept bin:")
    bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
    bin_labels = ['<0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>0.8']
    valid_notes['intercept_bin'] = pd.cut(valid_notes['coreNoteIntercept'], bins=bins, labels=bin_labels)

    calibration_by_bin = []
    for bin_label in bin_labels:
        bin_data = valid_notes[valid_notes['intercept_bin'] == bin_label]
        if len(bin_data) > 0:
            mean_pred = bin_data['coreNoteIntercept'].mean()
            mean_actual = bin_data['is_crh'].mean()
            count = len(bin_data)
            print(f"    {bin_label}: pred={mean_pred:.3f}, actual={mean_actual:.3f}, n={count:,}")
            calibration_by_bin.append({
                'bin': bin_label,
                'mean_prediction': float(mean_pred),
                'mean_actual': float(mean_actual),
                'count': int(count),
            })

    results = {
        'n_notes': len(valid_notes),
        'n_crh': int(crh_count),
        'crh_rate': float(crh_count / len(valid_notes)),
        'calibration_slope': float(slope),
        'calibration_intercept': float(intercept),
        'calibration_r_squared': float(r_value**2),
        'crh_auc': float(crh_auc) if crh_auc else None,
        'crh_pr_auc': float(pr_auc) if pr_auc else None,
        'calibration_by_bin': calibration_by_bin,
        'pass_threshold': 0.3,
        'pass': slope > 0.3,
    }

    print(f"\n  RESULT: {'PASS' if results['pass'] else 'FAIL'} (slope > 0.3)")

    return results


# ============================================================================
# TASK 3: CANONICAL FACTOR BASIS
# ============================================================================

def create_canonical_factor_basis(
    notes: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    note_factor_cols: List[str],
    user_factor_cols: List[str],
) -> Dict:
    """
    Create canonical factor basis with frozen axis ordering.

    Procedure:
    1. Compute polarization score for each axis
    2. Rank axes by polarization (highest = most divisive)
    3. Save canonical ordering for reproducibility
    """
    print(f"\n{'='*60}")
    print("TASK 3: CANONICAL FACTOR BASIS")
    print(f"{'='*60}")

    # Prepare ratings with helpfulness
    if helpfulnessLevelKey in ratings.columns:
        ratings = ratings.copy()
        ratings['helpful'] = ratings[helpfulnessLevelKey].map(HELPFULNESS_MAP)
    elif 'helpful' not in ratings.columns:
        return {'error': 'No helpfulness column found'}

    # Sample ratings for speed
    sample_ratings = ratings.sample(min(50000, len(ratings)), random_state=42)

    # Merge with factors
    sample_ratings = sample_ratings.merge(
        notes[[noteIdKey] + note_factor_cols].dropna(),
        on=noteIdKey, how='inner'
    )
    sample_ratings = sample_ratings.merge(
        users[[raterParticipantIdKey] + user_factor_cols].dropna(),
        on=raterParticipantIdKey, how='inner'
    )

    print(f"Ratings with factors: {len(sample_ratings):,}")

    # Compute polarization score for each axis
    polarization_scores = {}

    for i, (nf, uf) in enumerate(zip(note_factor_cols, user_factor_cols)):
        # Polarization = variance of (user_f * note_f) weighted by rating disagreement
        user_f = sample_ratings[uf].values
        note_f = sample_ratings[nf].values
        helpful = sample_ratings['helpful'].values

        # Interaction term
        interaction = user_f * note_f

        # Polarization: how much does factor alignment predict helpfulness?
        # High polarization = large variance in interaction AND correlation with rating
        polarization = np.var(interaction)

        # Also check correlation with residuals
        mean_helpful = helpful.mean()
        residual = helpful - mean_helpful
        corr_with_residual = np.corrcoef(interaction, residual)[0, 1]
        if np.isnan(corr_with_residual):
            corr_with_residual = 0

        # Combined score: variance * |correlation|
        combined_score = polarization * (1 + abs(corr_with_residual))

        polarization_scores[f'F{i+1}'] = {
            'variance': float(polarization),
            'correlation_with_residual': float(corr_with_residual),
            'combined_score': float(combined_score),
            'note_col': nf,
            'user_col': uf,
        }

        print(f"  F{i+1}: var={polarization:.4f}, corr={corr_with_residual:.4f}, score={combined_score:.4f}")

    # Rank by combined score (descending)
    ranked = sorted(polarization_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)

    # Identify axes
    main_polarization = ranked[0][0]
    common_ground = ranked[-1][0]

    print(f"\n  Main polarization axis: {main_polarization}")
    print(f"  Common ground axis: {common_ground}")
    print(f"  Ranked order: {[r[0] for r in ranked]}")

    # Compute factor variances for notes
    note_variances = {}
    for col in note_factor_cols:
        var = notes[col].dropna().var()
        factor_num = col.replace('coreNoteFactor', 'F').replace('internalNoteFactor', 'F')
        note_variances[factor_num] = float(var)

    # Compute factor correlation matrix
    note_factors_df = notes[note_factor_cols].dropna()
    corr_matrix = note_factors_df.corr()
    factor_correlations = {}
    for i, col1 in enumerate(note_factor_cols):
        f1 = f'F{i+1}'
        for j, col2 in enumerate(note_factor_cols):
            f2 = f'F{j+1}'
            factor_correlations[f'{f1}-{f2}'] = float(corr_matrix.loc[col1, col2])

    # Get off-diagonal max
    n = len(note_factor_cols)
    off_diag = corr_matrix.values[np.triu_indices(n, k=1)]
    max_off_diag = float(np.abs(off_diag).max())

    import datetime

    results = {
        'main_polarization_axis': main_polarization,
        'common_ground_axis': common_ground,
        'ranked_by_polarization': [r[0] for r in ranked],
        'polarization_scores': polarization_scores,
        'note_variances': note_variances,
        'max_off_diagonal_correlation': max_off_diag,
        'computed_on': datetime.datetime.now().isoformat(),
        'n_ratings_used': len(sample_ratings),
        'note_factor_columns': note_factor_cols,
        'user_factor_columns': user_factor_cols,
    }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run remaining analyses')
    parser.add_argument('--notes', required=True, help='Path to scored_notes.tsv')
    parser.add_argument('--users', required=True, help='Path to helpfulness_scores.tsv')
    parser.add_argument('--ratings', required=True, help='Path to ratings directory or file')
    parser.add_argument('--outdir', default='data/', help='Output directory')
    parser.add_argument('--sample-frac', type=float, default=0.3, help='Sample fraction for temporal stability')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    notes = pd.read_csv(args.notes, sep='\t', low_memory=False)
    print(f"  Notes: {len(notes):,}")

    users = pd.read_csv(args.users, sep='\t', low_memory=False)
    print(f"  Users: {len(users):,}")

    # Load ratings
    ratings_path = Path(args.ratings)
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path, sep='\t', low_memory=False)
    else:
        rating_files = sorted(ratings_path.glob("ratings*.tsv"))
        if not rating_files:
            print(f"No rating files found in {ratings_path}")
            return
        # Load first file only for speed
        print(f"  Loading {rating_files[0].name}...")
        ratings = pd.read_csv(rating_files[0], sep='\t', low_memory=False)
    print(f"  Ratings: {len(ratings):,}")

    # Find factor columns
    note_factor_cols = find_factor_columns(notes, 'note')
    user_factor_cols = find_factor_columns(users, 'user')
    print(f"  Note factors: {note_factor_cols}")
    print(f"  User factors: {user_factor_cols}")

    all_results = {}

    # Task 1: Temporal Stability
    temporal_results = run_temporal_stability(
        ratings, n_splits=3, sample_frac=args.sample_frac
    )
    all_results['temporal_stability'] = temporal_results

    # Save temporal results
    temporal_path = outdir / 'temporal_stability_results.json'
    with open(temporal_path, 'w') as f:
        json.dump(temporal_results, f, indent=2)
    print(f"\nSaved temporal stability results to: {temporal_path}")

    # Task 2: CRH Calibration
    calibration_results = run_crh_calibration(
        notes, ratings, users, note_factor_cols, user_factor_cols
    )
    all_results['crh_calibration'] = calibration_results

    # Save calibration results (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(v) for v in d]
        else:
            return convert_numpy(d)

    calibration_path = outdir / 'crh_calibration_results.json'
    with open(calibration_path, 'w') as f:
        json.dump(convert_dict(calibration_results), f, indent=2)
    print(f"\nSaved CRH calibration results to: {calibration_path}")

    # Task 3: Canonical Factor Basis
    basis_results = create_canonical_factor_basis(
        notes, users, ratings, note_factor_cols, user_factor_cols
    )
    all_results['canonical_basis'] = basis_results

    # Save canonical basis
    basis_path = outdir / 'canonical_factor_basis.json'
    with open(basis_path, 'w') as f:
        json.dump(basis_results, f, indent=2)
    print(f"\nSaved canonical factor basis to: {basis_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'error' not in temporal_results:
        status = "PASS" if temporal_results['pass'] else "FAIL"
        print(f"  Temporal Stability: {status} (mean r={temporal_results['temporal_stability']['mean']:.3f})")
    else:
        print(f"  Temporal Stability: ERROR - {temporal_results['error']}")

    if 'error' not in calibration_results:
        status = "PASS" if calibration_results['pass'] else "FAIL"
        print(f"  CRH Calibration: {status} (slope={calibration_results['calibration_slope']:.3f})")
    else:
        print(f"  CRH Calibration: ERROR - {calibration_results['error']}")

    if 'error' not in basis_results:
        print(f"  Canonical Basis: Main polarization={basis_results['main_polarization_axis']}")
    else:
        print(f"  Canonical Basis: ERROR - {basis_results['error']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
