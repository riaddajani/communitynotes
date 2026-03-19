#!/usr/bin/env python3
"""
Ablation Study: Number of Factors and Regularization

Tests whether >1 dimension actually helps by running a controlled sweep:
- Factors: K = 1, 2, 3, 4, 6
- Regularizers: L2 baseline vs SIGReg
- Multiple random seeds for stability

Reports:
1. Held-out rating prediction (RMSE, log loss)
2. Stability across random seeds
3. Top notes per axis (qualitative)

Usage:
    python ablation_num_factors.py \
        --ratings data/ratings/ \
        --output results/ablation/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'matrix_factorization'))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    createdAtMillisKey,
    helpfulnessLevelKey,
)

# Import SIGReg loss
try:
    from matrix_factorization.sigreg_loss import SIGRegLoss
except ImportError:
    from sigreg_loss import SIGRegLoss


# Helpfulness level mappings
HELPFULNESS_LEVEL_MAP = {
    'NOT_HELPFUL': 0.0,
    'SOMEWHAT_HELPFUL': 0.5,
    'HELPFUL': 1.0,
}


class MatrixFactorizationModel(nn.Module):
    """Simple MF model for ablation study."""

    def __init__(
        self,
        n_users: int,
        n_notes: int,
        n_factors: int,
        use_sigreg: bool = False,
        sigreg_lambda: float = 0.01,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.use_sigreg = use_sigreg

        # Embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.note_factors = nn.Embedding(n_notes, n_factors)
        self.user_intercepts = nn.Embedding(n_users, 1)
        self.note_intercepts = nn.Embedding(n_notes, 1)
        self.global_intercept = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.note_factors.weight, std=0.1)
        nn.init.zeros_(self.user_intercepts.weight)
        nn.init.zeros_(self.note_intercepts.weight)

        # SIGReg loss - CRITICAL: Use TWO SEPARATE instances!
        # Using a single instance causes state coupling between user/note embeddings
        # which leads to RMSE explosion (3-5 instead of ~0.47)
        if use_sigreg:
            self.sigreg_user = SIGRegLoss(lambda_sketch=sigreg_lambda)
            self.sigreg_note = SIGRegLoss(lambda_sketch=sigreg_lambda)
        else:
            self.sigreg_user = None
            self.sigreg_note = None

    def forward(self, user_ids, note_ids):
        user_f = self.user_factors(user_ids)
        note_f = self.note_factors(note_ids)
        user_i = self.user_intercepts(user_ids).squeeze(-1)
        note_i = self.note_intercepts(note_ids).squeeze(-1)

        # Prediction: global + user_intercept + note_intercept + dot(user_factors, note_factors)
        pred = self.global_intercept + user_i + note_i + (user_f * note_f).sum(dim=1)
        return pred

    def get_regularization_loss(self):
        """Get regularization loss (L2 or SIGReg).

        IMPORTANT: SIGReg uses separate instances for user and note embeddings
        to prevent state coupling that would cause RMSE explosion.
        """
        if self.sigreg_user is not None:
            # SIGReg on factors - using separate instances for each embedding
            user_reg = self.sigreg_user(self.user_factors.weight)
            note_reg = self.sigreg_note(self.note_factors.weight)
            return user_reg + note_reg
        else:
            # Standard L2
            l2_lambda = 0.01
            l2_reg = l2_lambda * (
                self.user_factors.weight.pow(2).mean() +
                self.note_factors.weight.pow(2).mean()
            )
            return l2_reg


def load_and_prepare_data(
    ratings_dir: str,
    sample_frac: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Load ratings and create train/test split.

    Returns: train_df, test_df, user_to_idx, note_to_idx
    """
    np.random.seed(seed)

    # Load ratings
    ratings_path = Path(ratings_dir)
    if ratings_path.is_file():
        ratings = pd.read_csv(ratings_path, sep='\t')
    else:
        rating_files = list(ratings_path.glob("ratings-*.tsv"))
        if not rating_files:
            rating_files = list(ratings_path.glob("ratings*.tsv"))
        ratings_list = [pd.read_csv(f, sep='\t') for f in sorted(rating_files)]
        ratings = pd.concat(ratings_list, ignore_index=True)

    # Sample for speed
    if sample_frac < 1.0:
        ratings = ratings.sample(frac=sample_frac, random_state=seed)

    # Convert helpfulness to numeric
    if helpfulnessLevelKey in ratings.columns:
        ratings['helpful_num'] = ratings[helpfulnessLevelKey].map(HELPFULNESS_LEVEL_MAP)
    elif 'helpful' in ratings.columns:
        ratings['helpful_num'] = ratings['helpful'].astype(float)
    else:
        raise ValueError("No helpful column found")

    # Drop rows with missing values
    ratings = ratings[[noteIdKey, raterParticipantIdKey, createdAtMillisKey, 'helpful_num']].dropna()

    # Create ID mappings
    unique_users = ratings[raterParticipantIdKey].unique()
    unique_notes = ratings[noteIdKey].unique()

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    note_to_idx = {n: i for i, n in enumerate(unique_notes)}

    ratings['user_idx'] = ratings[raterParticipantIdKey].map(user_to_idx)
    ratings['note_idx'] = ratings[noteIdKey].map(note_to_idx)

    # Time-based split
    ratings = ratings.sort_values(createdAtMillisKey)
    split_idx = int(len(ratings) * (1 - test_frac))

    train_df = ratings.iloc[:split_idx].copy()
    test_df = ratings.iloc[split_idx:].copy()

    return train_df, test_df, user_to_idx, note_to_idx


def train_model(
    train_df: pd.DataFrame,
    n_users: int,
    n_notes: int,
    n_factors: int,
    use_sigreg: bool,
    seed: int,
    n_epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 0.01,
    device: str = 'cpu',
) -> MatrixFactorizationModel:
    """Train MF model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MatrixFactorizationModel(
        n_users=n_users,
        n_notes=n_notes,
        n_factors=n_factors,
        use_sigreg=use_sigreg,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # Prepare data
    user_ids = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    note_ids = torch.tensor(train_df['note_idx'].values, dtype=torch.long)
    targets = torch.tensor(train_df['helpful_num'].values, dtype=torch.float32)

    dataset = TensorDataset(user_ids, note_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_users, batch_notes, batch_targets in dataloader:
            batch_users = batch_users.to(device)
            batch_notes = batch_notes.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            preds = model(batch_users, batch_notes)
            pred_loss = mse_loss(preds, batch_targets)
            reg_loss = model.get_regularization_loss()
            loss = pred_loss + reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def evaluate_model(
    model: MatrixFactorizationModel,
    test_df: pd.DataFrame,
    user_to_idx: Dict,
    note_to_idx: Dict,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()

    # Filter test to known users/notes
    test_df = test_df[
        test_df['user_idx'].notna() &
        test_df['note_idx'].notna()
    ].copy()

    if len(test_df) == 0:
        return {'rmse': np.nan, 'log_loss': np.nan, 'n_test': 0}

    with torch.no_grad():
        user_ids = torch.tensor(test_df['user_idx'].values, dtype=torch.long).to(device)
        note_ids = torch.tensor(test_df['note_idx'].values, dtype=torch.long).to(device)
        targets = test_df['helpful_num'].values

        preds = model(user_ids, note_ids).cpu().numpy()

    # RMSE
    rmse = np.sqrt(np.mean((targets - preds) ** 2))

    # Log loss (binary)
    preds_clipped = np.clip(preds, 0.01, 0.99)
    targets_binary = (targets >= 0.5).astype(float)
    log_loss = -np.mean(
        targets_binary * np.log(preds_clipped) +
        (1 - targets_binary) * np.log(1 - preds_clipped)
    )

    return {
        'rmse': rmse,
        'log_loss': log_loss,
        'n_test': len(test_df),
    }


def get_factor_stats(model: MatrixFactorizationModel) -> Dict:
    """Get statistics about learned factors."""
    user_factors = model.user_factors.weight.detach().cpu().numpy()
    note_factors = model.note_factors.weight.detach().cpu().numpy()

    stats = {
        'user_factor_variances': user_factors.var(axis=0).tolist(),
        'note_factor_variances': note_factors.var(axis=0).tolist(),
        'user_factor_means': user_factors.mean(axis=0).tolist(),
        'note_factor_means': note_factors.mean(axis=0).tolist(),
    }

    # Variance ratio (min/max)
    if len(stats['user_factor_variances']) > 1:
        user_vars = np.array(stats['user_factor_variances'])
        note_vars = np.array(stats['note_factor_variances'])
        stats['user_variance_ratio'] = float(user_vars.min() / (user_vars.max() + 1e-8))
        stats['note_variance_ratio'] = float(note_vars.min() / (note_vars.max() + 1e-8))
    else:
        stats['user_variance_ratio'] = 1.0
        stats['note_variance_ratio'] = 1.0

    return stats


def run_ablation(
    ratings_dir: str,
    output_dir: str,
    factor_counts: List[int] = [1, 2, 3, 4, 6],
    seeds: List[int] = [42, 123, 456],
    sample_frac: float = 0.1,
    n_epochs: int = 20,
):
    """Run full ablation study."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Load data once (same split for all)
    print("Loading data...")
    train_df, test_df, user_to_idx, note_to_idx = load_and_prepare_data(
        ratings_dir, sample_frac=sample_frac, seed=42
    )
    n_users = len(user_to_idx)
    n_notes = len(note_to_idx)
    print(f"  Train: {len(train_df):,} ratings")
    print(f"  Test: {len(test_df):,} ratings")
    print(f"  Users: {n_users:,}, Notes: {n_notes:,}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Run experiments
    total_experiments = len(factor_counts) * 2 * len(seeds)  # factors × (L2, SIGReg) × seeds
    exp_num = 0

    for n_factors in factor_counts:
        for use_sigreg in [False, True]:
            reg_name = 'SIGReg' if use_sigreg else 'L2'

            seed_results = []
            for seed in seeds:
                exp_num += 1
                print(f"\n[{exp_num}/{total_experiments}] K={n_factors}, {reg_name}, seed={seed}")

                start_time = time.time()

                # Train
                model = train_model(
                    train_df, n_users, n_notes, n_factors, use_sigreg, seed,
                    n_epochs=n_epochs, device=device
                )

                # Evaluate
                metrics = evaluate_model(model, test_df, user_to_idx, note_to_idx, device)
                factor_stats = get_factor_stats(model)

                elapsed = time.time() - start_time

                result = {
                    'n_factors': n_factors,
                    'regularizer': reg_name,
                    'seed': seed,
                    'rmse': metrics['rmse'],
                    'log_loss': metrics['log_loss'],
                    'n_test': metrics['n_test'],
                    'train_time': elapsed,
                    **factor_stats,
                }
                seed_results.append(result)
                results.append(result)

                print(f"  RMSE: {metrics['rmse']:.4f}, LogLoss: {metrics['log_loss']:.4f}, Time: {elapsed:.1f}s")
                if n_factors > 1:
                    print(f"  Variance ratio - User: {factor_stats['user_variance_ratio']:.3f}, Note: {factor_stats['note_variance_ratio']:.3f}")

            # Summary for this config
            rmses = [r['rmse'] for r in seed_results]
            print(f"  → Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'ablation_results.tsv', sep='\t', index=False)

    # Print summary
    print_summary(results_df, output_path)

    return results_df


def print_summary(results_df: pd.DataFrame, output_path: Path):
    """Print and save summary of ablation results."""

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Aggregate by config
    summary = results_df.groupby(['n_factors', 'regularizer']).agg({
        'rmse': ['mean', 'std'],
        'log_loss': ['mean', 'std'],
        'user_variance_ratio': 'mean',
        'note_variance_ratio': 'mean',
    }).round(4)

    print("\n" + "-" * 80)
    print("HELD-OUT PREDICTION PERFORMANCE")
    print("-" * 80)
    print(f"{'Config':<20} {'RMSE (mean±std)':<20} {'LogLoss (mean±std)':<20}")
    print("-" * 80)

    for (n_factors, reg), row in summary.iterrows():
        config = f"K={n_factors}, {reg}"
        rmse_str = f"{row[('rmse', 'mean')]:.4f} ± {row[('rmse', 'std')]:.4f}"
        ll_str = f"{row[('log_loss', 'mean')]:.4f} ± {row[('log_loss', 'std')]:.4f}"
        print(f"{config:<20} {rmse_str:<20} {ll_str:<20}")

    # Best config
    best_idx = results_df.groupby(['n_factors', 'regularizer'])['rmse'].mean().idxmin()
    print(f"\n★ Best config: K={best_idx[0]}, {best_idx[1]}")

    # Improvement analysis
    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS (vs K=1 L2 baseline)")
    print("-" * 80)

    baseline = results_df[(results_df['n_factors'] == 1) & (results_df['regularizer'] == 'L2')]['rmse'].mean()

    for (n_factors, reg), row in summary.iterrows():
        rmse = row[('rmse', 'mean')]
        improvement = (baseline - rmse) / baseline * 100
        print(f"K={n_factors}, {reg}: {improvement:+.2f}% {'✓' if improvement > 0 else ''}")

    # Stability analysis
    print("\n" + "-" * 80)
    print("STABILITY ACROSS SEEDS (lower std = more stable)")
    print("-" * 80)

    stability = results_df.groupby(['n_factors', 'regularizer'])['rmse'].std().sort_values()
    for (n_factors, reg), std in stability.items():
        print(f"K={n_factors}, {reg}: std={std:.4f}")

    # Variance ratio (SIGReg effectiveness)
    print("\n" + "-" * 80)
    print("FACTOR USAGE (variance ratio, higher = more balanced)")
    print("-" * 80)

    for (n_factors, reg), row in summary.iterrows():
        if n_factors > 1:
            user_vr = row[('user_variance_ratio', 'mean')]
            note_vr = row[('note_variance_ratio', 'mean')]
            print(f"K={n_factors}, {reg}: user={user_vr:.3f}, note={note_vr:.3f}")

    # Save summary
    summary_path = output_path / 'ablation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary.to_string())

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Ablation study: number of factors')
    parser.add_argument('--ratings', required=True, help='Path to ratings directory')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--factors', type=str, default='1,2,3,4,6',
                        help='Comma-separated list of factor counts to test')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--sample', type=float, default=0.1,
                        help='Fraction of ratings to sample')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    args = parser.parse_args()

    factor_counts = [int(x) for x in args.factors.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    run_ablation(
        args.ratings,
        args.output,
        factor_counts=factor_counts,
        seeds=seeds,
        sample_frac=args.sample,
        n_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
