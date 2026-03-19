"""
Visualizations for Factor2 Analysis - SIGreg Multi-Factor Embeddings

This script creates visualizations demonstrating what Factor2 captures:
- A second axis of rater polarization orthogonal to Factor1 (left-right politics)
- "Platform Loyalty / Musk Ecosystem" dimension

Usage:
    python visualize_factor2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load scored notes and merge with note content."""
    scored_notes = pd.read_csv(DATA_DIR / "scored_notes.tsv", sep='\t', low_memory=False)
    notes_data = pd.read_csv(DATA_DIR / "notes-00000.tsv", sep='\t', low_memory=False)

    merged = scored_notes.merge(
        notes_data[['noteId', 'summary']],
        on='noteId',
        how='inner'
    )

    # Filter to notes with Factor2
    factor2_col = 'internalNoteFactor2'
    merged_with_f2 = merged[merged[factor2_col].notna()].copy()

    return merged_with_f2


def plot_factor1_vs_factor2(df, output_path):
    """
    Scatter plot of Factor1 vs Factor2 showing orthogonality.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    factor1 = df['coreNoteFactor1']
    factor2 = df['internalNoteFactor2']

    # Color by status
    colors = []
    for status in df['coreRatingStatus']:
        if status == 'CURRENTLY_RATED_HELPFUL':
            colors.append('#2ecc71')  # green
        elif status == 'CURRENTLY_RATED_NOT_HELPFUL':
            colors.append('#e74c3c')  # red
        elif status == 'NEEDS_MORE_RATINGS':
            colors.append('#3498db')  # blue
        else:
            colors.append('#95a5a6')  # gray

    ax.scatter(factor1, factor2, c=colors, alpha=0.3, s=10)

    # Add quadrant labels
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Annotations for quadrants
    ax.text(0.7, 0.7, 'Right-leaning\nHigh F2', fontsize=10, ha='center',
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.3, 0.7, 'Left-leaning\nHigh F2', fontsize=10, ha='center',
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.7, 0.3, 'Right-leaning\nLow F2 (Pro-Musk)', fontsize=10, ha='center',
            transform=ax.transAxes, alpha=0.7)
    ax.text(0.3, 0.3, 'Left-leaning\nLow F2 (Pro-Musk)', fontsize=10, ha='center',
            transform=ax.transAxes, alpha=0.7)

    # Calculate correlation
    corr = np.corrcoef(factor1, factor2)[0, 1]

    ax.set_xlabel('Factor1 (Left ← → Right Political Spectrum)', fontsize=12)
    ax.set_ylabel('Factor2 (Platform Loyalty Dimension)', fontsize=12)
    ax.set_title(f'Factor1 vs Factor2: Orthogonal Dimensions\n(Correlation: {corr:.4f})', fontsize=14)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Currently Rated Helpful'),
        mpatches.Patch(color='#e74c3c', label='Currently Rated Not Helpful'),
        mpatches.Patch(color='#3498db', label='Needs More Ratings'),
        mpatches.Patch(color='#95a5a6', label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factor2_by_status(df, output_path):
    """
    Box plot of Factor2 distribution by rating status.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    statuses = ['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL',
                'NEEDS_MORE_RATINGS', 'NEEDS_YOUR_HELP', 'FIRM_REJECT']

    data = []
    labels = []
    for status in statuses:
        subset = df[df['coreRatingStatus'] == status]['internalNoteFactor2']
        if len(subset) > 0:
            data.append(subset.values)
            labels.append(f"{status}\n(n={len(subset):,})")

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#95a5a6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Factor2 Value', fontsize=12)
    ax.set_title('Factor2 Distribution by Rating Status', fontsize=14)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_extreme_factor2_content(df, output_path):
    """
    Bar chart showing content patterns at Factor2 extremes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Get extreme notes
    top_positive = df.nlargest(50, 'internalNoteFactor2')
    top_negative = df.nsmallest(50, 'internalNoteFactor2')

    # Content categories
    def categorize_content(summary):
        s = str(summary).lower()

        if any(x in s for x in ['musk', 'elon', 'tesla', 'spacex']):
            return 'Musk/Tesla/SpaceX'
        elif any(x in s for x in ['trump', 'maga', 'republican']):
            return 'Trump/MAGA'
        elif any(x in s for x in ['trans', 'gender', 'lgb', 'pronoun']):
            return 'Trans/Gender'
        elif any(x in s for x in ['biden', 'democrat', 'liberal', 'kamala']):
            return 'Biden/Democrats'
        elif any(x in s for x in ['nazi', 'salute', 'fascist', 'hitler']):
            return 'Nazi/Fascism'
        elif any(x in s for x in ['immigra', 'border', 'illegal']):
            return 'Immigration'
        elif any(x in s for x in ['covid', 'vaccine', 'pandemic']):
            return 'COVID/Vaccines'
        else:
            return 'Other'

    # Categorize
    top_positive['category'] = top_positive['summary'].apply(categorize_content)
    top_negative['category'] = top_negative['summary'].apply(categorize_content)

    # Plot positive F2
    pos_counts = top_positive['category'].value_counts()
    axes[0].barh(pos_counts.index, pos_counts.values, color='#3498db')
    axes[0].set_xlabel('Count')
    axes[0].set_title('HIGH Factor2 Notes (Top 50)\nTopic Distribution', fontsize=12)
    axes[0].invert_yaxis()

    # Plot negative F2
    neg_counts = top_negative['category'].value_counts()
    axes[1].barh(neg_counts.index, neg_counts.values, color='#e74c3c')
    axes[1].set_xlabel('Count')
    axes[1].set_title('LOW Factor2 Notes (Bottom 50)\nTopic Distribution', fontsize=12)
    axes[1].invert_yaxis()

    plt.suptitle('Factor2 Extremes: Content Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factor2_quadrant_distribution(df, output_path):
    """
    Show the 4-quadrant distribution of Factor1 x Factor2.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Categorize into quadrants
    df['quadrant'] = ''
    df.loc[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] >= 0), 'quadrant'] = 'Right + High F2'
    df.loc[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] >= 0), 'quadrant'] = 'Left + High F2'
    df.loc[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] < 0), 'quadrant'] = 'Right + Low F2'
    df.loc[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] < 0), 'quadrant'] = 'Left + Low F2'

    counts = df['quadrant'].value_counts()

    # Create 2x2 grid
    quadrant_data = [
        [counts.get('Left + High F2', 0), counts.get('Right + High F2', 0)],
        [counts.get('Left + Low F2', 0), counts.get('Right + Low F2', 0)]
    ]

    total = sum(sum(row) for row in quadrant_data)

    im = ax.imshow([[0.25, 0.25], [0.25, 0.25]], cmap='Blues', vmin=0, vmax=0.5)

    # Add text
    labels = [
        ['Left + High F2\n(Topic-focused Left)', 'Right + High F2\n(Topic-focused Right)'],
        ['Left + Low F2\n(Pro-Platform Left)', 'Right + Low F2\n(Pro-Platform Right)']
    ]

    for i in range(2):
        for j in range(2):
            count = quadrant_data[i][j]
            pct = 100 * count / total
            ax.text(j, i, f'{labels[i][j]}\n\nn={count:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Left (F1 < 0)', 'Right (F1 > 0)'], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['High F2\n(Topic-focused)', 'Low F2\n(Pro-Platform)'], fontsize=12)

    ax.set_title('Factor1 × Factor2 Quadrant Distribution\n(Near-uniform = successful orthogonalization)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factor2_variance_by_intercept(df, output_path):
    """
    Show how Factor2 variance changes with note intercept.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bin by intercept
    df['intercept_bin'] = pd.cut(df['coreNoteIntercept'],
                                  bins=[-2, -0.5, 0, 0.3, 0.5, 0.7, 0.9, 2],
                                  labels=['<-0.5', '-0.5-0', '0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '>0.9'])

    stats = df.groupby('intercept_bin')['internalNoteFactor2'].agg(['mean', 'std', 'count'])
    stats = stats[stats['count'] > 20]

    x = range(len(stats))
    ax.bar(x, stats['std'], color='#3498db', alpha=0.7, label='Factor2 Std Dev')
    ax.plot(x, stats['std'], 'ro-', markersize=8, label='Trend')

    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    ax.set_xlabel('Note Intercept Range', fontsize=12)
    ax.set_ylabel('Factor2 Standard Deviation', fontsize=12)
    ax.set_title('Factor2 Variance Increases for "Contested" Notes\n(Higher intercepts = closer to helpful threshold)', fontsize=14)

    # Add count annotations
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.annotate(f'n={int(row["count"]):,}', (i, row['std'] + 0.005),
                   ha='center', fontsize=9, alpha=0.7)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(df, output_path):
    """
    Create a summary statistics table.
    """
    factor1 = df['coreNoteFactor1']
    factor2 = df['internalNoteFactor2']

    summary = f"""
# Factor2 Analysis Summary

## Basic Statistics

| Metric | Factor1 | Factor2 |
|--------|---------|---------|
| Mean | {factor1.mean():.4f} | {factor2.mean():.4f} |
| Std Dev | {factor1.std():.4f} | {factor2.std():.4f} |
| Min | {factor1.min():.4f} | {factor2.min():.4f} |
| Max | {factor1.max():.4f} | {factor2.max():.4f} |
| Correlation | {np.corrcoef(factor1, factor2)[0,1]:.4f} | - |

## Quadrant Distribution

| Quadrant | Count | Percentage |
|----------|-------|------------|
| Left + High F2 | {len(df[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] >= 0)]):,} | {100*len(df[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] >= 0)])/len(df):.1f}% |
| Right + High F2 | {len(df[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] >= 0)]):,} | {100*len(df[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] >= 0)])/len(df):.1f}% |
| Left + Low F2 | {len(df[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] < 0)]):,} | {100*len(df[(df['coreNoteFactor1'] < 0) & (df['internalNoteFactor2'] < 0)])/len(df):.1f}% |
| Right + Low F2 | {len(df[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] < 0)]):,} | {100*len(df[(df['coreNoteFactor1'] >= 0) & (df['internalNoteFactor2'] < 0)])/len(df):.1f}% |

## Key Finding

Factor2 captures a **second axis of rater polarization** orthogonal to traditional left-right politics:

- **Low Factor2**: Notes defending Elon Musk, Trump, or platform-adjacent content
- **High Factor2**: Notes about trans/gender issues, political violence context, topic-specific debates

This dimension represents a "Platform Loyalty" vs "Topic-Focused" split that exists within both left and right political camps.
"""

    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"Saved: {output_path}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} notes with Factor2 values")

    print("\nGenerating visualizations...")

    # Generate all plots
    plot_factor1_vs_factor2(df, OUTPUT_DIR / "factor1_vs_factor2_scatter.png")
    plot_factor2_by_status(df, OUTPUT_DIR / "factor2_by_status.png")
    plot_extreme_factor2_content(df, OUTPUT_DIR / "factor2_extreme_content.png")
    plot_factor2_quadrant_distribution(df, OUTPUT_DIR / "factor2_quadrant_distribution.png")
    plot_factor2_variance_by_intercept(df, OUTPUT_DIR / "factor2_variance_by_intercept.png")
    create_summary_table(df, OUTPUT_DIR / "factor2_summary.md")

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()