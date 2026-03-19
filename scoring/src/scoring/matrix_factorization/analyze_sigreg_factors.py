#!/usr/bin/env python3
"""
SIGreg Factor Analysis Script

Analyzes the Factor1 and Factor2 embeddings from Community Notes scoring
to understand what each dimension captures.

Usage:
    python analyze_sigreg_factors.py --notes <scored_notes.tsv> --users <helpfulness_scores.tsv>
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(notes_path: str, users_path: str):
    """Load scored notes and helpfulness scores."""
    notes = pd.read_csv(notes_path, sep='\t')
    users = pd.read_csv(users_path, sep='\t')
    return notes, users


def find_factor_columns(df: pd.DataFrame, factor_type: str = 'note'):
    """Find factor columns in dataframe."""
    if factor_type == 'note':
        # Try different naming conventions
        factor1_candidates = ['coreNoteFactor1', 'internalNoteFactor1']
        factor2_candidates = ['internalNoteFactor2', 'coreNoteFactor2']
    else:
        factor1_candidates = ['coreRaterFactor1', 'internalRaterFactor1']
        factor2_candidates = ['internalRaterFactor2', 'coreRaterFactor2']

    factor1 = None
    factor2 = None

    for col in factor1_candidates:
        if col in df.columns:
            factor1 = col
            break

    for col in factor2_candidates:
        if col in df.columns:
            factor2 = col
            break

    return factor1, factor2


def analyze_isotropy(notes: pd.DataFrame, users: pd.DataFrame):
    """Analyze isotropy of embeddings."""
    print("\n" + "="*60)
    print("1. ISOTROPY ANALYSIS")
    print("="*60)

    # Notes
    note_f1, note_f2 = find_factor_columns(notes, 'note')
    if note_f1 and note_f2:
        note_factors = notes[[note_f1, note_f2]].dropna()
        if len(note_factors) > 0:
            corr = note_factors.corr().iloc[0, 1]
            print(f"\nNote Factors ({note_f1}, {note_f2}):")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Factor1 std: {note_factors[note_f1].std():.4f}")
            print(f"  Factor2 std: {note_factors[note_f2].std():.4f}")
            print(f"  Factor1 range: [{note_factors[note_f1].min():.3f}, {note_factors[note_f1].max():.3f}]")
            print(f"  Factor2 range: [{note_factors[note_f2].min():.3f}, {note_factors[note_f2].max():.3f}]")

            # Compute covariance matrix
            cov = note_factors.cov()
            eigenvalues = np.linalg.eigvalsh(cov)
            isotropy_score = eigenvalues.min() / (eigenvalues.max() + 1e-8)
            print(f"  Isotropy score: {isotropy_score:.4f} (1.0 = perfect)")
            print(f"  Condition number: {eigenvalues.max() / (eigenvalues.min() + 1e-8):.2f}")
    else:
        print(f"\nNote factors not found. Available columns: {list(notes.columns)}")

    # Users
    user_f1, user_f2 = find_factor_columns(users, 'user')
    if user_f1 and user_f2:
        user_factors = users[[user_f1, user_f2]].dropna()
        if len(user_factors) > 0:
            corr = user_factors.corr().iloc[0, 1]
            print(f"\nUser Factors ({user_f1}, {user_f2}):")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Factor1 std: {user_factors[user_f1].std():.4f}")
            print(f"  Factor2 std: {user_factors[user_f2].std():.4f}")
            print(f"  Factor1 range: [{user_factors[user_f1].min():.3f}, {user_factors[user_f1].max():.3f}]")
            print(f"  Factor2 range: [{user_factors[user_f2].min():.3f}, {user_factors[user_f2].max():.3f}]")

            # Compute covariance matrix
            cov = user_factors.cov()
            eigenvalues = np.linalg.eigvalsh(cov)
            isotropy_score = eigenvalues.min() / (eigenvalues.max() + 1e-8)
            print(f"  Isotropy score: {isotropy_score:.4f} (1.0 = perfect)")
            print(f"  Condition number: {eigenvalues.max() / (eigenvalues.min() + 1e-8):.2f}")
    else:
        print(f"\nUser factors not found. Available columns: {list(users.columns)}")

    return note_f1, note_f2, user_f1, user_f2


def visualize_embeddings(notes: pd.DataFrame, users: pd.DataFrame,
                         note_f1: str, note_f2: str, user_f1: str, user_f2: str,
                         output_dir: str = None):
    """Create visualizations of the embedding space."""
    print("\n" + "="*60)
    print("2. EMBEDDING SPACE VISUALIZATION")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # User embedding space
    if user_f1 and user_f2:
        user_factors = users[[user_f1, user_f2]].dropna()
        ax = axes[0]
        ax.scatter(user_factors[user_f1], user_factors[user_f2],
                   alpha=0.3, s=5, c='blue')
        ax.set_xlabel('Factor 1 (Political Polarity)')
        ax.set_ylabel('Factor 2 (New Dimension)')
        ax.set_title(f'User Embedding Space (n={len(user_factors):,})')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

    # Note embedding space
    if note_f1 and note_f2:
        note_factors = notes[[note_f1, note_f2]].dropna()
        ax = axes[1]
        ax.scatter(note_factors[note_f1], note_factors[note_f2],
                   alpha=0.3, s=5, c='green')
        ax.set_xlabel('Factor 1 (Political Polarity)')
        ax.set_ylabel('Factor 2 (New Dimension)')
        ax.set_title(f'Note Embedding Space (n={len(note_factors):,})')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'sigreg_embedding_space.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved embedding visualization to: {output_path}")

    plt.show()

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if user_f1 and user_f2:
        user_factors = users[[user_f1, user_f2]].dropna()
        axes[0, 0].hist(user_factors[user_f1], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Factor 1')
        axes[0, 0].set_title('User Factor 1 Distribution')

        axes[0, 1].hist(user_factors[user_f2], bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Factor 2')
        axes[0, 1].set_title('User Factor 2 Distribution')

    if note_f1 and note_f2:
        note_factors = notes[[note_f1, note_f2]].dropna()
        axes[1, 0].hist(note_factors[note_f1], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Factor 1')
        axes[1, 0].set_title('Note Factor 1 Distribution')

        axes[1, 1].hist(note_factors[note_f2], bins=50, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Factor 2')
        axes[1, 1].set_title('Note Factor 2 Distribution')

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'sigreg_factor_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plots to: {output_path}")

    plt.show()


def analyze_factor_correlations(notes: pd.DataFrame, users: pd.DataFrame,
                                 note_f1: str, note_f2: str, user_f1: str, user_f2: str):
    """Analyze correlations of Factor2 with other available columns."""
    print("\n" + "="*60)
    print("3. FACTOR2 CORRELATION ANALYSIS")
    print("="*60)

    # For notes
    if note_f2:
        print(f"\nNote Factor2 correlations with other columns:")
        numeric_cols = notes.select_dtypes(include=[np.number]).columns
        correlations = []
        for col in numeric_cols:
            if col != note_f2 and col != note_f1:
                try:
                    corr = notes[[note_f2, col]].dropna().corr().iloc[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, corr in correlations[:15]:
            print(f"  {col}: {corr:+.4f}")

    # For users
    if user_f2:
        print(f"\nUser Factor2 correlations with other columns:")
        numeric_cols = users.select_dtypes(include=[np.number]).columns
        correlations = []
        for col in numeric_cols:
            if col != user_f2 and col != user_f1:
                try:
                    corr = users[[user_f2, col]].dropna().corr().iloc[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, corr in correlations[:15]:
            print(f"  {col}: {corr:+.4f}")


def analyze_quadrants(notes: pd.DataFrame, users: pd.DataFrame,
                      note_f1: str, note_f2: str, user_f1: str, user_f2: str):
    """Analyze the four quadrants of the embedding space."""
    print("\n" + "="*60)
    print("4. QUADRANT ANALYSIS")
    print("="*60)

    if user_f1 and user_f2:
        user_factors = users[[user_f1, user_f2]].dropna()

        q1 = len(user_factors[(user_factors[user_f1] > 0) & (user_factors[user_f2] > 0)])
        q2 = len(user_factors[(user_factors[user_f1] < 0) & (user_factors[user_f2] > 0)])
        q3 = len(user_factors[(user_factors[user_f1] < 0) & (user_factors[user_f2] < 0)])
        q4 = len(user_factors[(user_factors[user_f1] > 0) & (user_factors[user_f2] < 0)])
        total = len(user_factors)

        print(f"\nUser Quadrant Distribution:")
        print(f"  Q1 (+F1, +F2): {q1:,} ({100*q1/total:.1f}%)")
        print(f"  Q2 (-F1, +F2): {q2:,} ({100*q2/total:.1f}%)")
        print(f"  Q3 (-F1, -F2): {q3:,} ({100*q3/total:.1f}%)")
        print(f"  Q4 (+F1, -F2): {q4:,} ({100*q4/total:.1f}%)")

    if note_f1 and note_f2:
        note_factors = notes[[note_f1, note_f2]].dropna()

        q1 = len(note_factors[(note_factors[note_f1] > 0) & (note_factors[note_f2] > 0)])
        q2 = len(note_factors[(note_factors[note_f1] < 0) & (note_factors[note_f2] > 0)])
        q3 = len(note_factors[(note_factors[note_f1] < 0) & (note_factors[note_f2] < 0)])
        q4 = len(note_factors[(note_factors[note_f1] > 0) & (note_factors[note_f2] < 0)])
        total = len(note_factors)

        print(f"\nNote Quadrant Distribution:")
        print(f"  Q1 (+F1, +F2): {q1:,} ({100*q1/total:.1f}%)")
        print(f"  Q2 (-F1, +F2): {q2:,} ({100*q2/total:.1f}%)")
        print(f"  Q3 (-F1, -F2): {q3:,} ({100*q3/total:.1f}%)")
        print(f"  Q4 (+F1, -F2): {q4:,} ({100*q4/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze SIGreg factor embeddings')
    parser.add_argument('--notes', required=True, help='Path to scored_notes.tsv')
    parser.add_argument('--users', required=True, help='Path to helpfulness_scores.tsv')
    parser.add_argument('--output-dir', default=None, help='Directory to save plots')
    args = parser.parse_args()

    print("Loading data...")
    notes, users = load_data(args.notes, args.users)
    print(f"Loaded {len(notes):,} notes and {len(users):,} users")

    # Run analysis
    note_f1, note_f2, user_f1, user_f2 = analyze_isotropy(notes, users)

    if note_f2 or user_f2:
        visualize_embeddings(notes, users, note_f1, note_f2, user_f1, user_f2, args.output_dir)
        analyze_factor_correlations(notes, users, note_f1, note_f2, user_f1, user_f2)
        analyze_quadrants(notes, users, note_f1, note_f2, user_f1, user_f2)
    else:
        print("\nERROR: Factor2 columns not found in data!")
        print(f"Note columns: {list(notes.columns)}")
        print(f"User columns: {list(users.columns)}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()