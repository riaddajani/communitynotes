#!/usr/bin/env python3
"""
Factor Cluster Analysis: Visualize combinations in 2D-6D space

Analyzes clusters and patterns in multi-dimensional factor space:
1. 2D scatter plots of factor pairs
2. PCA to find main variance directions
3. Cluster analysis to find user/note groups
4. Extreme corners in multi-dimensional space

Usage:
    python factor_clusters.py \
        --notes data/notes-00000.tsv \
        --scored-notes data/scored_notes.tsv \
        --users data/helpfulness_scores.tsv \
        --output figures/
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    noteIdKey,
    raterParticipantIdKey,
    summaryKey,
    classificationKey,
    finalRatingStatusKey,
)


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


def plot_2d_factor_pairs(
    df: pd.DataFrame,
    factor_cols: List[str],
    entity_type: str,
    output_dir: Path,
    color_col: Optional[str] = None,
    sample_n: int = 5000,
):
    """Create 2D scatter plots for all factor pairs."""

    # Get valid rows
    valid = df[factor_cols].dropna()
    if len(valid) > sample_n:
        valid = valid.sample(n=sample_n, random_state=42)

    n_factors = len(factor_cols)
    n_pairs = n_factors * (n_factors - 1) // 2

    # Create subplot grid
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    pair_idx = 0
    for i, col1 in enumerate(factor_cols):
        for j, col2 in enumerate(factor_cols):
            if i >= j:
                continue

            ax = axes[pair_idx]

            x = valid[col1]
            y = valid[col2]

            # Color by another column if provided
            if color_col and color_col in df.columns:
                colors = df.loc[valid.index, color_col]
                scatter = ax.scatter(x, y, c=colors, alpha=0.3, s=5, cmap='coolwarm')
                plt.colorbar(scatter, ax=ax, label=color_col)
            else:
                ax.scatter(x, y, alpha=0.3, s=5)

            ax.set_xlabel(f'Factor {i+1}')
            ax.set_ylabel(f'Factor {j+1}')
            ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)

            # Add correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f'r={corr:.3f}')

            pair_idx += 1

    # Hide unused axes
    for idx in range(pair_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{entity_type.title()} Factor Pairs (n={len(valid):,})', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f'{entity_type}_factor_pairs.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pca_analysis(
    df: pd.DataFrame,
    factor_cols: List[str],
    entity_type: str,
    output_dir: Path,
):
    """PCA analysis to find main variance directions."""

    valid = df[factor_cols].dropna()

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid)

    # PCA
    pca = PCA()
    transformed = pca.fit_transform(scaled)

    # Plot variance explained
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Variance explained
    ax = axes[0]
    var_exp = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)
    ax.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, label='Individual')
    ax.plot(range(1, len(cum_var)+1), cum_var, 'r-o', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.set_xticks(range(1, len(var_exp)+1))

    # PC1 vs PC2
    ax = axes[1]
    sample_idx = np.random.choice(len(transformed), min(5000, len(transformed)), replace=False)
    ax.scatter(transformed[sample_idx, 0], transformed[sample_idx, 1], alpha=0.3, s=5)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
    ax.set_title('First Two Principal Components')
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)

    # Loadings
    ax = axes[2]
    loadings = pca.components_[:2].T  # First 2 PCs
    for i, (col, loading) in enumerate(zip(factor_cols, loadings)):
        ax.arrow(0, 0, loading[0], loading[1], head_width=0.05, head_length=0.02, fc='blue', ec='blue')
        ax.text(loading[0]*1.1, loading[1]*1.1, f'F{i+1}', fontsize=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('PC1 Loading')
    ax.set_ylabel('PC2 Loading')
    ax.set_title('Factor Loadings on PC1/PC2')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal')

    plt.suptitle(f'{entity_type.title()} PCA Analysis', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f'{entity_type}_pca.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Print loadings
    print(f"\n  PCA Loadings for {entity_type}:")
    print(f"  {'Factor':<25} {'PC1':>10} {'PC2':>10} {'PC3':>10}")
    print("  " + "-"*55)
    for i, col in enumerate(factor_cols):
        print(f"  {col:<25} {pca.components_[0,i]:>10.3f} {pca.components_[1,i]:>10.3f} {pca.components_[2,i]:>10.3f}")

    return pca, scaler


def find_extreme_corners(
    df: pd.DataFrame,
    factor_cols: List[str],
    notes_full: Optional[pd.DataFrame] = None,
    n_dims: int = 3,
    percentile: float = 5.0,
    n_examples: int = 3,
):
    """Find entities at extreme corners in multi-dimensional space."""

    valid = df[factor_cols[:n_dims]].dropna()

    print(f"\n  Extreme corners in {n_dims}D space (top/bottom {percentile}% on each axis):")

    # Generate all 2^n_dims corners
    corners = []
    for i in range(2**n_dims):
        corner = []
        for d in range(n_dims):
            corner.append('high' if (i >> d) & 1 else 'low')
        corners.append(corner)

    for corner in corners:
        corner_label = '/'.join([f"F{d+1}:{c[0].upper()}" for d, c in enumerate(corner)])

        # Filter to this corner
        mask = pd.Series(True, index=valid.index)
        for d, direction in enumerate(corner):
            col = factor_cols[d]
            if direction == 'low':
                threshold = np.percentile(valid[col], percentile)
                mask &= (valid[col] <= threshold)
            else:
                threshold = np.percentile(valid[col], 100 - percentile)
                mask &= (valid[col] >= threshold)

        corner_entities = valid[mask]

        if len(corner_entities) > 0:
            print(f"\n    {corner_label}: {len(corner_entities)} entities")

            # Show examples if we have note text
            if notes_full is not None and summaryKey in notes_full.columns:
                merged = df.loc[corner_entities.index].merge(
                    notes_full[[noteIdKey, summaryKey]],
                    on=noteIdKey,
                    how='left'
                )
                for _, row in merged.head(n_examples).iterrows():
                    text = str(row.get(summaryKey, ''))[:100]
                    print(f"      - {text}...")


def cluster_analysis(
    df: pd.DataFrame,
    factor_cols: List[str],
    entity_type: str,
    output_dir: Path,
    n_clusters: int = 8,
):
    """K-means clustering to find natural groups."""

    valid = df[factor_cols].dropna()

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    # Add labels back
    valid_with_labels = valid.copy()
    valid_with_labels['cluster'] = labels

    # Analyze clusters
    print(f"\n  Cluster Analysis ({n_clusters} clusters):")
    print(f"  {'Cluster':<10} {'Size':>10} ", end='')
    for i, col in enumerate(factor_cols):
        print(f"{'F'+str(i+1)+' mean':>10} ", end='')
    print()
    print("  " + "-"*(12 + 11*len(factor_cols)))

    cluster_stats = []
    for c in range(n_clusters):
        cluster_data = valid_with_labels[valid_with_labels['cluster'] == c]
        means = cluster_data[factor_cols].mean()

        print(f"  {c:<10} {len(cluster_data):>10} ", end='')
        for mean in means:
            print(f"{mean:>10.3f} ", end='')
        print()

        cluster_stats.append({
            'cluster': c,
            'size': len(cluster_data),
            **{f'F{i+1}_mean': m for i, m in enumerate(means)}
        })

    # Plot clusters in PC space
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)

    # Plot cluster centers
    centers_transformed = pca.transform(kmeans.cluster_centers_)
    for i, (x, y) in enumerate(centers_transformed):
        ax.scatter(x, y, c='black', marker='x', s=200, linewidths=3)
        ax.annotate(f'C{i}', (x, y), fontsize=12, fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{entity_type.title()} Clusters (K={n_clusters})')

    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()

    output_path = output_dir / f'{entity_type}_clusters.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return valid_with_labels, cluster_stats


def plot_factor_by_status(
    scored_notes: pd.DataFrame,
    factor_cols: List[str],
    output_dir: Path,
):
    """Plot factor distributions by note status."""

    if finalRatingStatusKey not in scored_notes.columns:
        print("  No finalRatingStatus column found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    statuses = ['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL', 'NEEDS_MORE_RATINGS']
    colors = {'CURRENTLY_RATED_HELPFUL': 'green',
              'CURRENTLY_RATED_NOT_HELPFUL': 'red',
              'NEEDS_MORE_RATINGS': 'gray'}

    for idx, col in enumerate(factor_cols):
        ax = axes[idx]

        for status in statuses:
            data = scored_notes[scored_notes[finalRatingStatusKey] == status][col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.5, label=status[:10], color=colors.get(status, 'blue'), density=True)

        ax.set_xlabel(f'Factor {idx+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'Factor {idx+1} by Status')
        ax.legend(fontsize=8)

    plt.suptitle('Factor Distributions by Note Status', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'factors_by_status.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def run_cluster_analysis(
    notes_path: str,
    scored_notes_path: str,
    users_path: str,
    output_dir: str,
):
    """Run full cluster analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    notes_full = pd.read_csv(notes_path, sep='\t')
    scored_notes = pd.read_csv(scored_notes_path, sep='\t')
    users = pd.read_csv(users_path, sep='\t')

    print(f"  Notes: {len(scored_notes):,}, Users: {len(users):,}")

    # Find factor columns
    note_factor_cols = find_factor_columns(scored_notes, 'note')
    user_factor_cols = find_factor_columns(users, 'user')

    print(f"  Found {len(note_factor_cols)} note factors, {len(user_factor_cols)} user factors")

    # Note analysis
    print("\n" + "="*60)
    print("NOTE FACTOR ANALYSIS")
    print("="*60)

    print("\n1. 2D Factor Pair Plots")
    plot_2d_factor_pairs(scored_notes, note_factor_cols, 'note', output_path)

    print("\n2. PCA Analysis")
    plot_pca_analysis(scored_notes, note_factor_cols, 'note', output_path)

    print("\n3. Extreme Corners (3D)")
    find_extreme_corners(scored_notes, note_factor_cols, notes_full, n_dims=3)

    print("\n4. K-Means Clustering")
    cluster_analysis(scored_notes, note_factor_cols, 'note', output_path)

    print("\n5. Factors by Status")
    plot_factor_by_status(scored_notes, note_factor_cols, output_path)

    # User analysis
    print("\n" + "="*60)
    print("USER FACTOR ANALYSIS")
    print("="*60)

    print("\n1. 2D Factor Pair Plots")
    plot_2d_factor_pairs(users, user_factor_cols, 'user', output_path)

    print("\n2. PCA Analysis")
    plot_pca_analysis(users, user_factor_cols, 'user', output_path)

    print("\n3. K-Means Clustering")
    cluster_analysis(users, user_factor_cols, 'user', output_path)

    print("\n" + "="*60)
    print(f"All figures saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Factor cluster analysis')
    parser.add_argument('--notes', required=True, help='Path to notes TSV (with text)')
    parser.add_argument('--scored-notes', required=True, help='Path to scored notes TSV')
    parser.add_argument('--users', required=True, help='Path to helpfulness scores TSV')
    parser.add_argument('--output', required=True, help='Output directory for figures')
    args = parser.parse_args()

    run_cluster_analysis(args.notes, args.scored_notes, args.users, args.output)


if __name__ == "__main__":
    main()
