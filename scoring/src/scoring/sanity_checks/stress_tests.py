#!/usr/bin/env python3
"""
Stress Tests for Multi-Factor Matrix Factorization

Tests whether K>1 factors (especially with SIGReg) improve robustness:
1. Brigading simulation: inject coordinated bloc, check if model detects/resists
2. Topic shift: train on one time window, test on later window
3. Cold start: evaluate notes/users with few ratings

Usage:
    python stress_tests.py \
        --ratings data/ratings-00000.tsv \
        --outdir data/stress_tests
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    helpfulnessLevelKey,
    createdAtMillisKey,
)

# Simple MF model for stress tests (self-contained)
class SimpleMF(torch.nn.Module):
    """Simple matrix factorization for stress testing."""

    def __init__(self, n_users: int, n_notes: int, n_factors: int = 1):
        super().__init__()
        self.n_factors = n_factors
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.note_factors = torch.nn.Embedding(n_notes, n_factors)
        self.user_intercepts = torch.nn.Embedding(n_users, 1)
        self.note_intercepts = torch.nn.Embedding(n_notes, 1)
        self.global_intercept = torch.nn.Parameter(torch.tensor(0.5))

        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.note_factors.weight)
        self.user_intercepts.weight.data.fill_(0.0)
        self.note_intercepts.weight.data.fill_(0.0)

    def forward(self, user_idx, note_idx):
        pred = self.global_intercept
        pred = pred + self.user_intercepts(user_idx).squeeze()
        pred = pred + self.note_intercepts(note_idx).squeeze()
        pred = pred + (self.user_factors(user_idx) * self.note_factors(note_idx)).sum(dim=1)
        return pred


# ============================================================================
# BRIGADING RESISTANCE STRATEGIES
# ============================================================================

def apply_norm_cap(model: SimpleMF, user_reputation: np.ndarray, max_norm: float = 1.0):
    """
    Cap factor norms for low-reputation users.

    This prevents new/suspicious users from having outsized influence
    through high factor norms. Brigade users typically have low reputation
    (new accounts, limited history) and shouldn't have high expressiveness.

    Args:
        model: The MF model to modify
        user_reputation: Array of reputation scores (higher = more trusted)
        max_norm: Maximum allowed norm for low-reputation users
    """
    with torch.no_grad():
        for user_idx, rep in enumerate(user_reputation):
            if rep < 0.1:  # New/low reputation threshold
                norm = model.user_factors.weight[user_idx].norm()
                if norm > max_norm:
                    model.user_factors.weight[user_idx] *= max_norm / norm


def compute_group_robust_penalty(
    model: SimpleMF,
    user_indices: torch.Tensor,
    min_group_size: int = 50,
    penalty_weight: float = 0.1,
) -> torch.Tensor:
    """
    Penalize cases where small cohorts dominate factor gradient.

    This prevents brigades from getting disproportionate influence by
    penalizing factor magnitudes for users in small rating cohorts.

    The key insight: brigades are typically small, coordinated groups.
    By penalizing users who appear in small cohorts, we reduce their
    ability to move note intercepts.

    Args:
        model: The MF model
        user_indices: Indices of users in current batch
        min_group_size: Users with fewer than this many ratings get penalized
        penalty_weight: Weight of the penalty term

    Returns:
        Regularization penalty tensor
    """
    # Count unique users in batch (proxy for rating activity)
    unique_users, counts = torch.unique(user_indices, return_counts=True)

    # Identify small-cohort users (potential brigade members)
    small_cohort_mask = counts < min_group_size

    if small_cohort_mask.sum() == 0:
        return torch.tensor(0.0)

    small_cohort_users = unique_users[small_cohort_mask]

    # Penalize their factor magnitudes
    penalty = model.user_factors.weight[small_cohort_users].pow(2).sum()

    return penalty * penalty_weight


def compute_user_rating_counts(ratings: pd.DataFrame) -> Dict:
    """Compute rating counts per user for cohort analysis."""
    return ratings.groupby(raterParticipantIdKey).size().to_dict()


HAS_MF = True  # We have our simple model


HELPFULNESS_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


def prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Prepare ratings with numeric helpfulness."""
    ratings = ratings.copy()
    if helpfulnessLevelKey in ratings.columns:
        ratings['helpful'] = ratings[helpfulnessLevelKey].map(HELPFULNESS_MAP)
    elif 'helpful' not in ratings.columns:
        raise ValueError("No helpfulness column found")

    # Remove rows with NaN helpful values
    ratings = ratings[ratings['helpful'].notna()].copy()

    return ratings


def build_index_maps(ratings: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Build user/note index mappings."""
    user_ids = ratings[raterParticipantIdKey].unique()
    note_ids = ratings[noteIdKey].unique()

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    note_to_idx = {nid: i for i, nid in enumerate(note_ids)}

    return user_to_idx, note_to_idx


def train_model(
    ratings: pd.DataFrame,
    user_to_idx: Dict,
    note_to_idx: Dict,
    n_factors: int = 1,
    n_epochs: int = 100,
    lr: float = 0.05,
    l2_lambda: float = 0.03,
    seed: int = 42,
    use_norm_cap: bool = False,
    use_group_robust: bool = False,
    user_rating_counts: Optional[Dict] = None,
) -> Tuple[SimpleMF, List[float]]:
    """Train a simple MF model.

    Args:
        ratings: Ratings dataframe
        user_to_idx: User ID to index mapping
        note_to_idx: Note ID to index mapping
        n_factors: Number of factors
        n_epochs: Training epochs
        lr: Learning rate
        l2_lambda: L2 regularization weight
        seed: Random seed
        use_norm_cap: Apply norm cap for low-activity users (brigade resistance)
        use_group_robust: Apply group-robust penalty (brigade resistance)
        user_rating_counts: Pre-computed user rating counts (for norm cap)

    Returns:
        Trained model and loss history
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_users = len(user_to_idx)
    n_notes = len(note_to_idx)

    model = SimpleMF(
        n_users=n_users,
        n_notes=n_notes,
        n_factors=n_factors,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    user_indices = torch.tensor([user_to_idx[u] for u in ratings[raterParticipantIdKey]], dtype=torch.long)
    note_indices = torch.tensor([note_to_idx[n] for n in ratings[noteIdKey]], dtype=torch.long)
    targets = torch.tensor(ratings['helpful'].values, dtype=torch.float32)

    # Compute user reputation proxy (rating count) for norm cap
    if use_norm_cap:
        if user_rating_counts is None:
            user_rating_counts = compute_user_rating_counts(ratings)

        # Normalize to 0-1 range based on percentiles
        counts = np.array([user_rating_counts.get(u, 0) for u in user_to_idx.keys()])
        max_count = np.percentile(counts, 95) if len(counts) > 0 else 1
        user_reputation = np.clip(counts / max(max_count, 1), 0, 1)
    else:
        user_reputation = None

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        predictions = model(user_indices, note_indices)
        mse_loss = ((predictions - targets) ** 2).mean()

        # L2 regularization
        l2_reg = l2_lambda * (
            model.user_factors.weight.pow(2).mean() +
            model.note_factors.weight.pow(2).mean()
        )

        loss = mse_loss + l2_reg

        # Group-robust penalty (brigade resistance)
        if use_group_robust:
            group_penalty = compute_group_robust_penalty(model, user_indices)
            loss = loss + group_penalty

        loss.backward()

        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Apply norm cap after each update (brigade resistance)
        if use_norm_cap and user_reputation is not None:
            apply_norm_cap(model, user_reputation, max_norm=1.0)

        losses.append(loss.item())

        # Early stopping if loss is NaN
        if np.isnan(loss.item()):
            print(f"    Warning: NaN loss at epoch {epoch}")
            break

    return model, losses


def compute_metrics(
    model: SimpleMF,
    ratings: pd.DataFrame,
    user_to_idx: Dict,
    note_to_idx: Dict,
) -> Dict[str, float]:
    """Compute evaluation metrics."""

    # Filter to known users/notes
    valid_mask = (
        ratings[raterParticipantIdKey].isin(user_to_idx) &
        ratings[noteIdKey].isin(note_to_idx)
    )
    valid_ratings = ratings[valid_mask].copy()

    if len(valid_ratings) == 0:
        return {'rmse': float('nan'), 'auc': float('nan')}

    user_indices = torch.tensor([user_to_idx[u] for u in valid_ratings[raterParticipantIdKey]], dtype=torch.long)
    note_indices = torch.tensor([note_to_idx[n] for n in valid_ratings[noteIdKey]], dtype=torch.long)
    targets = valid_ratings['helpful'].values

    with torch.no_grad():
        predictions = model(user_indices, note_indices).numpy()

    # Handle NaN predictions
    valid_mask_preds = ~np.isnan(predictions)
    if valid_mask_preds.sum() == 0:
        return {'rmse': float('nan'), 'auc': float('nan')}

    predictions = predictions[valid_mask_preds]
    targets = targets[valid_mask_preds]

    rmse = np.sqrt(((predictions - targets) ** 2).mean())

    # AUC for binary helpful/not helpful
    binary_targets = (targets > 0.25).astype(int)
    if len(np.unique(binary_targets)) > 1:
        try:
            auc = roc_auc_score(binary_targets, predictions)
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')

    return {'rmse': rmse, 'auc': auc}


# ============================================================================
# TEST 1: BRIGADING SIMULATION
# ============================================================================

def inject_brigade(
    ratings: pd.DataFrame,
    n_brigade_users: int = 50,
    n_target_notes: int = 100,
    brigade_rating: float = 1.0,  # All rate helpful
    seed: int = 42,
) -> Tuple[pd.DataFrame, List, List]:
    """
    Inject a synthetic brigade: coordinated users who all rate the same notes
    the same way, trying to manipulate consensus.

    Returns:
        - Modified ratings DataFrame
        - List of brigade user IDs
        - List of target note IDs
    """
    np.random.seed(seed)

    # Create fake user IDs
    brigade_users = [f"BRIGADE_USER_{i}" for i in range(n_brigade_users)]

    # Pick random notes to target
    all_notes = ratings[noteIdKey].unique()
    target_notes = np.random.choice(all_notes, size=min(n_target_notes, len(all_notes)), replace=False)

    # Create brigade ratings
    brigade_ratings = []
    for user in brigade_users:
        for note in target_notes:
            brigade_ratings.append({
                raterParticipantIdKey: user,
                noteIdKey: note,
                'helpful': brigade_rating,
            })

    brigade_df = pd.DataFrame(brigade_ratings)

    # Combine with original
    combined = pd.concat([ratings, brigade_df], ignore_index=True)

    return combined, brigade_users, list(target_notes)


def test_brigading(
    ratings: pd.DataFrame,
    n_factors_list: List[int] = [1, 2, 4, 6],
    n_brigade_users: int = 50,
    n_target_notes: int = 100,
    seed: int = 42,
    test_resistance_strategies: bool = True,
) -> Dict:
    """
    Test if K>1 factors help resist brigading.

    We measure:
    1. How much the target notes' intercepts shift with/without brigade
    2. Whether model detects brigade users (low factor alignment with real users)
    3. Whether resistance strategies (norm cap, group-robust) reduce intercept shift

    Args:
        ratings: Ratings dataframe
        n_factors_list: List of factor counts to test
        n_brigade_users: Number of fake brigade users
        n_target_notes: Number of notes to target
        seed: Random seed
        test_resistance_strategies: Also test norm cap and group-robust penalty
    """

    print("\n" + "=" * 80)
    print("TEST 1: BRIGADING SIMULATION")
    print("=" * 80)

    # Data already prepared in run_all_tests

    # Train baseline models (no brigade)
    print("\nTraining baseline models (no brigade)...")
    baseline_results = {}

    user_to_idx, note_to_idx = build_index_maps(ratings)
    user_rating_counts = compute_user_rating_counts(ratings)

    for K in n_factors_list:
        model, _ = train_model(ratings, user_to_idx, note_to_idx, n_factors=K, seed=seed)

        # Get note intercepts for target notes (we'll pick them later for fair comparison)
        with torch.no_grad():
            note_intercepts = model.note_intercepts.weight.squeeze().detach().numpy()

        baseline_results[K] = {
            'model': model,
            'note_intercepts': dict(zip(note_to_idx.keys(), note_intercepts)),
        }
        print(f"  K={K}: trained")

    # Inject brigade
    print(f"\nInjecting brigade: {n_brigade_users} users, {n_target_notes} target notes...")
    brigaded_ratings, brigade_users, target_notes = inject_brigade(
        ratings, n_brigade_users, n_target_notes, brigade_rating=1.0, seed=seed
    )

    # Train models with brigade (no resistance)
    print("\nTraining models WITH brigade (NO resistance)...")
    brigade_results = {}

    user_to_idx_b, note_to_idx_b = build_index_maps(brigaded_ratings)
    user_rating_counts_b = compute_user_rating_counts(brigaded_ratings)

    for K in n_factors_list:
        model, _ = train_model(brigaded_ratings, user_to_idx_b, note_to_idx_b, n_factors=K, seed=seed)

        with torch.no_grad():
            note_intercepts = model.note_intercepts.weight.squeeze().detach().numpy()
            user_factors = model.user_factors.weight.detach().numpy()

        # Get intercepts for target notes
        target_intercepts = {}
        for nid in target_notes:
            if nid in note_to_idx_b:
                target_intercepts[nid] = note_intercepts[note_to_idx_b[nid]]

        # Get brigade user factors
        brigade_factor_norms = []
        for uid in brigade_users:
            if uid in user_to_idx_b:
                factors = user_factors[user_to_idx_b[uid]]
                brigade_factor_norms.append(np.linalg.norm(factors))

        brigade_results[K] = {
            'target_intercepts': target_intercepts,
            'brigade_factor_norm_mean': np.mean(brigade_factor_norms),
        }
        print(f"  K={K}: trained, brigade factor norm = {np.mean(brigade_factor_norms):.3f}")

    # Train models with resistance strategies
    resistance_results = {}
    if test_resistance_strategies:
        print("\nTraining models WITH brigade + RESISTANCE strategies...")

        for strategy_name, use_norm_cap, use_group_robust in [
            ('norm_cap', True, False),
            ('group_robust', False, True),
            ('both', True, True),
        ]:
            print(f"\n  Strategy: {strategy_name}")
            resistance_results[strategy_name] = {}

            for K in n_factors_list:
                model, _ = train_model(
                    brigaded_ratings, user_to_idx_b, note_to_idx_b, n_factors=K, seed=seed,
                    use_norm_cap=use_norm_cap,
                    use_group_robust=use_group_robust,
                    user_rating_counts=user_rating_counts_b,
                )

                with torch.no_grad():
                    note_intercepts = model.note_intercepts.weight.squeeze().detach().numpy()
                    user_factors = model.user_factors.weight.detach().numpy()

                target_intercepts = {}
                for nid in target_notes:
                    if nid in note_to_idx_b:
                        target_intercepts[nid] = note_intercepts[note_to_idx_b[nid]]

                # Brigade user factor norms (should be lower with resistance)
                brigade_factor_norms = []
                for uid in brigade_users:
                    if uid in user_to_idx_b:
                        factors = user_factors[user_to_idx_b[uid]]
                        brigade_factor_norms.append(np.linalg.norm(factors))

                resistance_results[strategy_name][K] = {
                    'target_intercepts': target_intercepts,
                    'brigade_factor_norm_mean': np.mean(brigade_factor_norms),
                }
                print(f"    K={K}: brigade factor norm = {np.mean(brigade_factor_norms):.3f}")

    # Compare intercept shifts
    print("\n" + "-" * 60)
    print("BRIGADING RESULTS: Intercept shift on target notes")
    print("-" * 60)

    print("\nWithout resistance strategies:")
    for K in n_factors_list:
        shifts = []
        for nid in target_notes:
            if nid in baseline_results[K]['note_intercepts'] and nid in brigade_results[K]['target_intercepts']:
                baseline = baseline_results[K]['note_intercepts'][nid]
                brigaded = brigade_results[K]['target_intercepts'][nid]
                shifts.append(brigaded - baseline)

        if shifts:
            mean_shift = np.mean(shifts)
            std_shift = np.std(shifts)
            print(f"  K={K}: Mean intercept shift = {mean_shift:+.4f} ± {std_shift:.4f}")
            brigade_results[K]['mean_shift'] = mean_shift
            brigade_results[K]['std_shift'] = std_shift

    if test_resistance_strategies:
        print("\nWith resistance strategies:")
        for strategy_name, strat_results in resistance_results.items():
            print(f"\n  [{strategy_name}]")
            for K in n_factors_list:
                shifts = []
                for nid in target_notes:
                    if nid in baseline_results[K]['note_intercepts'] and nid in strat_results[K]['target_intercepts']:
                        baseline = baseline_results[K]['note_intercepts'][nid]
                        resisted = strat_results[K]['target_intercepts'][nid]
                        shifts.append(resisted - baseline)

                if shifts:
                    mean_shift = np.mean(shifts)
                    std_shift = np.std(shifts)
                    no_resist_shift = brigade_results[K].get('mean_shift', mean_shift)
                    reduction = ((no_resist_shift - mean_shift) / abs(no_resist_shift)) * 100 if no_resist_shift != 0 else 0
                    print(f"    K={K}: Mean shift = {mean_shift:+.4f} ± {std_shift:.4f} (reduction: {reduction:+.1f}%)")
                    strat_results[K]['mean_shift'] = mean_shift
                    strat_results[K]['shift_reduction_pct'] = reduction

    return {
        'baseline': baseline_results,
        'brigade': brigade_results,
        'resistance': resistance_results,
        'target_notes': target_notes,
        'brigade_users': brigade_users,
    }


# ============================================================================
# TEST 2: TOPIC/TIME SHIFT
# ============================================================================

def test_time_shift(
    ratings: pd.DataFrame,
    n_factors_list: List[int] = [1, 2, 4, 6],
    train_fraction: float = 0.7,
    seed: int = 42,
) -> Dict:
    """
    Test if K>1 factors help with temporal generalization.

    Train on first 70% of data (by time), test on last 30%.
    This simulates topic shift as new events/topics emerge.
    """

    print("\n" + "=" * 80)
    print("TEST 2: TIME/TOPIC SHIFT")
    print("=" * 80)

    # Data already prepared in run_all_tests

    # Sort by time if available
    if createdAtMillisKey in ratings.columns:
        ratings = ratings.sort_values(createdAtMillisKey)
        print(f"  Sorted {len(ratings):,} ratings by timestamp")
    else:
        print("  Warning: No timestamp column, using row order")

    # Split
    split_idx = int(len(ratings) * train_fraction)
    train_ratings = ratings.iloc[:split_idx].copy()
    test_ratings = ratings.iloc[split_idx:].copy()

    print(f"  Train: {len(train_ratings):,} ratings")
    print(f"  Test: {len(test_ratings):,} ratings")

    # Check overlap
    train_users = set(train_ratings[raterParticipantIdKey])
    test_users = set(test_ratings[raterParticipantIdKey])
    train_notes = set(train_ratings[noteIdKey])
    test_notes = set(test_ratings[noteIdKey])

    print(f"  User overlap: {len(train_users & test_users):,} / {len(test_users):,} test users")
    print(f"  Note overlap: {len(train_notes & test_notes):,} / {len(test_notes):,} test notes")

    # Train and evaluate
    results = {}
    user_to_idx, note_to_idx = build_index_maps(train_ratings)

    print("\nTraining models on early data, testing on later data...")

    for K in n_factors_list:
        model, _ = train_model(train_ratings, user_to_idx, note_to_idx, n_factors=K, seed=seed)

        # Evaluate on train
        train_metrics = compute_metrics(model, train_ratings, user_to_idx, note_to_idx)

        # Evaluate on test (only known users/notes)
        test_metrics = compute_metrics(model, test_ratings, user_to_idx, note_to_idx)

        results[K] = {
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_auc': train_metrics['auc'],
            'test_auc': test_metrics['auc'],
            'generalization_gap': test_metrics['rmse'] - train_metrics['rmse'],
        }

        print(f"  K={K}: Train RMSE={train_metrics['rmse']:.4f}, Test RMSE={test_metrics['rmse']:.4f}, Gap={results[K]['generalization_gap']:+.4f}")

    print("\n" + "-" * 60)
    print("TIME SHIFT RESULTS")
    print("-" * 60)
    print(f"{'K':<6} {'Train RMSE':<12} {'Test RMSE':<12} {'Gap':<12} {'Train AUC':<12} {'Test AUC':<12}")
    print("-" * 60)
    for K in n_factors_list:
        r = results[K]
        print(f"{K:<6} {r['train_rmse']:<12.4f} {r['test_rmse']:<12.4f} {r['generalization_gap']:<+12.4f} {r['train_auc']:<12.4f} {r['test_auc']:<12.4f}")

    return results


# ============================================================================
# TEST 3: COLD START
# ============================================================================

def test_cold_start(
    ratings: pd.DataFrame,
    n_factors_list: List[int] = [1, 2, 4, 6],
    min_ratings_threshold: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Test cold-start performance: how well do we predict for notes/users with few ratings?

    We train on full data but evaluate separately on:
    - "Warm" ratings: users and notes both have >= threshold ratings
    - "Cold note" ratings: notes have < threshold ratings
    - "Cold user" ratings: users have < threshold ratings
    """

    print("\n" + "=" * 80)
    print("TEST 3: COLD START")
    print("=" * 80)

    # Data already prepared in run_all_tests

    # Count ratings per user and note
    user_counts = ratings.groupby(raterParticipantIdKey).size()
    note_counts = ratings.groupby(noteIdKey).size()

    ratings['user_count'] = ratings[raterParticipantIdKey].map(user_counts)
    ratings['note_count'] = ratings[noteIdKey].map(note_counts)

    # Define cold/warm
    warm_mask = (ratings['user_count'] >= min_ratings_threshold) & (ratings['note_count'] >= min_ratings_threshold)
    cold_note_mask = (ratings['user_count'] >= min_ratings_threshold) & (ratings['note_count'] < min_ratings_threshold)
    cold_user_mask = (ratings['user_count'] < min_ratings_threshold) & (ratings['note_count'] >= min_ratings_threshold)

    warm_ratings = ratings[warm_mask]
    cold_note_ratings = ratings[cold_note_mask]
    cold_user_ratings = ratings[cold_user_mask]

    print(f"  Warm ratings: {len(warm_ratings):,}")
    print(f"  Cold note ratings: {len(cold_note_ratings):,}")
    print(f"  Cold user ratings: {len(cold_user_ratings):,}")

    # Train on all data
    user_to_idx, note_to_idx = build_index_maps(ratings)

    results = {}

    print("\nTraining models and evaluating on cold-start splits...")

    for K in n_factors_list:
        model, _ = train_model(ratings, user_to_idx, note_to_idx, n_factors=K, seed=seed)

        warm_metrics = compute_metrics(model, warm_ratings, user_to_idx, note_to_idx)
        cold_note_metrics = compute_metrics(model, cold_note_ratings, user_to_idx, note_to_idx)
        cold_user_metrics = compute_metrics(model, cold_user_ratings, user_to_idx, note_to_idx)

        results[K] = {
            'warm_rmse': warm_metrics['rmse'],
            'cold_note_rmse': cold_note_metrics['rmse'],
            'cold_user_rmse': cold_user_metrics['rmse'],
            'cold_note_gap': cold_note_metrics['rmse'] - warm_metrics['rmse'],
            'cold_user_gap': cold_user_metrics['rmse'] - warm_metrics['rmse'],
        }

        print(f"  K={K}: Warm={warm_metrics['rmse']:.4f}, ColdNote={cold_note_metrics['rmse']:.4f}, ColdUser={cold_user_metrics['rmse']:.4f}")

    print("\n" + "-" * 60)
    print("COLD START RESULTS")
    print("-" * 60)
    print(f"{'K':<6} {'Warm RMSE':<12} {'Cold Note':<12} {'Gap':<10} {'Cold User':<12} {'Gap':<10}")
    print("-" * 60)
    for K in n_factors_list:
        r = results[K]
        print(f"{K:<6} {r['warm_rmse']:<12.4f} {r['cold_note_rmse']:<12.4f} {r['cold_note_gap']:<+10.4f} {r['cold_user_rmse']:<12.4f} {r['cold_user_gap']:<+10.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests(
    ratings_path: str,
    outdir: str,
    sample_size: Optional[int] = None,
):
    """Run all stress tests."""

    if not HAS_MF:
        print("ERROR: BiasedMatrixFactorization not available. Cannot run tests.")
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading ratings...")
    ratings = pd.read_csv(ratings_path, sep='\t')
    print(f"  Loaded {len(ratings):,} ratings")

    if sample_size and len(ratings) > sample_size:
        ratings = ratings.sample(n=sample_size, random_state=42)
        print(f"  Sampled to {len(ratings):,} ratings")

    # Prepare and validate data
    ratings = prepare_ratings(ratings)
    print(f"  After cleaning: {len(ratings):,} ratings")
    print(f"  Helpful values - min: {ratings['helpful'].min()}, max: {ratings['helpful'].max()}, mean: {ratings['helpful'].mean():.3f}")

    n_factors_list = [1, 2, 4, 6]

    # Run tests
    brigade_results = test_brigading(ratings, n_factors_list)
    time_results = test_time_shift(ratings, n_factors_list)
    cold_results = test_cold_start(ratings, n_factors_list)

    # Summary
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)

    print("\n1. BRIGADING: Does K>1 resist coordinated manipulation better?")
    print("   (Smaller intercept shift = more robust)")

    print("\n2. TIME SHIFT: Does K>1 generalize to new topics/time periods better?")
    print("   (Smaller gap = better generalization)")
    for K in n_factors_list:
        print(f"   K={K}: Gap = {time_results[K]['generalization_gap']:+.4f}")

    print("\n3. COLD START: Does K>1 handle new users/notes better?")
    print("   (Smaller gap = better cold start)")
    for K in n_factors_list:
        print(f"   K={K}: Cold note gap = {cold_results[K]['cold_note_gap']:+.4f}, Cold user gap = {cold_results[K]['cold_user_gap']:+.4f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Determine winner for each test
    best_time = min(n_factors_list, key=lambda k: time_results[k]['generalization_gap'])
    best_cold_note = min(n_factors_list, key=lambda k: cold_results[k]['cold_note_gap'])
    best_cold_user = min(n_factors_list, key=lambda k: cold_results[k]['cold_user_gap'])

    print(f"  Best for time shift: K={best_time}")
    print(f"  Best for cold note: K={best_cold_note}")
    print(f"  Best for cold user: K={best_cold_user}")

    return {
        'brigade': brigade_results,
        'time_shift': time_results,
        'cold_start': cold_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Stress tests for multi-factor MF')
    parser.add_argument('--ratings', required=True, help='Path to ratings TSV')
    parser.add_argument('--outdir', default='data/stress_tests', help='Output directory')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for faster testing')
    args = parser.parse_args()

    run_all_tests(args.ratings, args.outdir, args.sample)


if __name__ == "__main__":
    main()
