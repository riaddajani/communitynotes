"""
Factor Space vs Text Heuristics for Humor/Social Note Classification

Goal: Show that factor space separates CRH vs NOT_HELPFUL humor notes
better than text heuristics alone.

If factors add predictive power, that's a real contribution showing
the latent space captures meaningful structure beyond surface text.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def extract_text_features(summary: str) -> dict:
    """Extract text-based heuristic features from note summary."""
    if pd.isna(summary):
        summary = ""

    text = str(summary).lower()

    # Length features
    char_len = len(summary)
    word_count = len(summary.split())

    # Profanity/hostility markers
    profanity_words = ['fuck', 'shit', 'damn', 'ass', 'bitch', 'crap', 'hell',
                       'bastard', 'cunt', 'dick', 'piss', 'slut', 'whore',
                       'retard', 'idiot', 'moron', 'loser', 'virgin', 'incel', 'cuck']
    profanity_count = sum(1 for w in profanity_words if w in text)
    has_profanity = profanity_count > 0

    # Dismissive phrases
    dismissive = ['shut up', 'stfu', 'get a life', 'touch grass', 'cope', 'seethe',
                  'nobody cares', 'who cares', 'no one cares', 'nnn', 'cry', 'ratio']
    dismissive_count = sum(1 for d in dismissive if d in text)

    # Emoji features
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]')
    emoji_count = len(emoji_pattern.findall(summary))

    # URL features
    has_url = 'http' in text or 'www.' in text or '.com' in text or '.org' in text
    url_count = text.count('http') + text.count('www.')

    # Rickroll detection
    is_rickroll = 'dQw4w9WgXcQ' in summary

    # Question marks / exclamation
    question_marks = summary.count('?')
    exclamation_marks = summary.count('!')

    # Caps ratio
    if len(summary) > 0:
        caps_ratio = sum(1 for c in summary if c.isupper()) / len(summary)
    else:
        caps_ratio = 0

    # Sarcasm markers
    sarcasm_markers = ['clearly', 'obviously', 'really?', 'seriously', 'literally',
                       'my brother in christ', 'bro', 'dude', 'come on']
    sarcasm_count = sum(1 for s in sarcasm_markers if s in text)

    # Factual tone markers (good for CRH)
    factual_markers = ['according to', 'source:', 'per', 'confirmed', 'states',
                       'wikipedia', 'https://en.']
    factual_count = sum(1 for f in factual_markers if f in text)

    return {
        'char_len': char_len,
        'word_count': word_count,
        'profanity_count': profanity_count,
        'has_profanity': int(has_profanity),
        'dismissive_count': dismissive_count,
        'emoji_count': emoji_count,
        'has_url': int(has_url),
        'url_count': url_count,
        'is_rickroll': int(is_rickroll),
        'question_marks': question_marks,
        'exclamation_marks': exclamation_marks,
        'caps_ratio': caps_ratio,
        'sarcasm_count': sarcasm_count,
        'factual_count': factual_count,
    }


def load_humor_candidates(path: str) -> pd.DataFrame:
    """Load humor candidates with status labels."""
    df = pd.read_csv(path, sep='\t')
    df.columns = ['noteId', 'status', 'summary']
    df['noteId'] = df['noteId'].astype(str)  # Ensure string for merge
    return df


def load_note_factors(scored_notes_path: str) -> pd.DataFrame:
    """Load factor values for notes."""
    # Try to load scored notes with factors
    df = pd.read_csv(scored_notes_path, sep='\t', low_memory=False)

    # Select key factor columns (not all 51)
    key_factors = [
        'noteId',
        'coreNoteIntercept', 'coreNoteFactor1',
        'internalNoteFactor2', 'internalNoteFactor3',
        'internalNoteFactor4', 'internalNoteFactor5', 'internalNoteFactor6',
        'expansionNoteIntercept', 'expansionNoteFactor1',
        'coverageNoteIntercept', 'coverageNoteFactor1',
    ]
    keep_cols = [c for c in key_factors if c in df.columns]

    result = df[keep_cols].copy()
    result['noteId'] = result['noteId'].astype(str)  # Ensure string for merge
    return result


def run_classification_comparison(humor_df: pd.DataFrame, factors_df: pd.DataFrame = None):
    """
    Compare classification performance:
    1. Text features only
    2. Factor features only
    3. Text + Factor features
    """

    # Filter to notes with definitive outcomes
    labeled = humor_df[humor_df['status'].isin(['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL'])].copy()

    print(f"Labeled notes: {len(labeled)}")
    print(f"  CRH: {(labeled['status'] == 'CURRENTLY_RATED_HELPFUL').sum()}")
    print(f"  NOT_HELPFUL: {(labeled['status'] == 'CURRENTLY_RATED_NOT_HELPFUL').sum()}")

    if len(labeled) < 20:
        print("Warning: Very small sample size, results will be noisy")

    # Create target: 1 = CRH (success), 0 = NOT_HELPFUL (failure)
    y = (labeled['status'] == 'CURRENTLY_RATED_HELPFUL').astype(int).values

    # Extract text features
    text_features = labeled['summary'].apply(extract_text_features).apply(pd.Series)
    X_text = text_features.values
    text_feature_names = list(text_features.columns)

    print(f"\nText features ({len(text_feature_names)}): {text_feature_names}")

    results = {}

    # 1. Text features only
    print("\n" + "="*60)
    print("1. TEXT FEATURES ONLY")
    print("="*60)

    scaler = StandardScaler()
    X_text_scaled = scaler.fit_transform(X_text)

    # Use stratified CV with small folds due to small sample
    cv = StratifiedKFold(n_splits=min(5, min(y.sum(), (1-y).sum())), shuffle=True, random_state=42)

    clf_text = LogisticRegression(max_iter=1000, random_state=42)
    scores_text = cross_val_score(clf_text, X_text_scaled, y, cv=cv, scoring='roc_auc')

    results['text_only'] = {
        'mean_auc': scores_text.mean(),
        'std_auc': scores_text.std(),
        'n_features': X_text.shape[1]
    }

    print(f"AUC: {scores_text.mean():.3f} (+/- {scores_text.std():.3f})")

    # Fit on all data to see feature importance
    clf_text.fit(X_text_scaled, y)
    print("\nTop text features (positive = predicts CRH):")
    importance = list(zip(text_feature_names, clf_text.coef_[0]))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in importance[:8]:
        direction = "→ CRH" if coef > 0 else "→ NOT_HELPFUL"
        print(f"  {feat}: {coef:+.3f} {direction}")

    # 2. Factor features only (if available)
    if factors_df is not None and len(factors_df) > 0:
        print("\n" + "="*60)
        print("2. FACTOR FEATURES ONLY")
        print("="*60)

        # Merge factors with labeled data
        labeled_with_factors = labeled.merge(factors_df, on='noteId', how='left')

        factor_cols = [c for c in factors_df.columns if c != 'noteId']
        X_factors = labeled_with_factors[factor_cols].values

        # Check for missing values
        valid_mask = ~np.isnan(X_factors).any(axis=1)
        if valid_mask.sum() < len(labeled):
            print(f"Notes with factor values: {valid_mask.sum()}/{len(labeled)}")

        if valid_mask.sum() >= 10:
            X_factors_valid = X_factors[valid_mask]
            y_valid = y[valid_mask]

            scaler_f = StandardScaler()
            X_factors_scaled = scaler_f.fit_transform(X_factors_valid)

            cv_f = StratifiedKFold(n_splits=min(5, min(y_valid.sum(), (1-y_valid).sum())), shuffle=True, random_state=42)

            clf_factors = LogisticRegression(max_iter=1000, random_state=42)
            scores_factors = cross_val_score(clf_factors, X_factors_scaled, y_valid, cv=cv_f, scoring='roc_auc')

            results['factors_only'] = {
                'mean_auc': scores_factors.mean(),
                'std_auc': scores_factors.std(),
                'n_features': X_factors_valid.shape[1],
                'n_samples': len(y_valid)
            }

            print(f"AUC: {scores_factors.mean():.3f} (+/- {scores_factors.std():.3f})")

            # Feature importance
            clf_factors.fit(X_factors_scaled, y_valid)
            print("\nFactor importance (positive = predicts CRH):")
            importance_f = list(zip(factor_cols, clf_factors.coef_[0]))
            importance_f.sort(key=lambda x: abs(x[1]), reverse=True)
            for feat, coef in importance_f[:6]:
                direction = "→ CRH" if coef > 0 else "→ NOT_HELPFUL"
                print(f"  {feat}: {coef:+.3f} {direction}")

            # 3. Combined features
            print("\n" + "="*60)
            print("3. TEXT + FACTOR FEATURES")
            print("="*60)

            X_text_valid = X_text[valid_mask]
            X_combined = np.hstack([X_text_valid, X_factors_valid])

            scaler_c = StandardScaler()
            X_combined_scaled = scaler_c.fit_transform(X_combined)

            clf_combined = LogisticRegression(max_iter=1000, random_state=42)
            scores_combined = cross_val_score(clf_combined, X_combined_scaled, y_valid, cv=cv_f, scoring='roc_auc')

            results['combined'] = {
                'mean_auc': scores_combined.mean(),
                'std_auc': scores_combined.std(),
                'n_features': X_combined.shape[1]
            }

            print(f"AUC: {scores_combined.mean():.3f} (+/- {scores_combined.std():.3f})")

            # Compare
            print("\n" + "="*60)
            print("COMPARISON SUMMARY")
            print("="*60)

            print(f"\n{'Model':<20} {'AUC':>10} {'Std':>10} {'Features':>10}")
            print("-" * 52)
            for name, r in results.items():
                print(f"{name:<20} {r['mean_auc']:>10.3f} {r['std_auc']:>10.3f} {r['n_features']:>10}")

            # Calculate improvement
            if 'factors_only' in results and 'text_only' in results:
                improvement = results['factors_only']['mean_auc'] - results['text_only']['mean_auc']
                print(f"\nFactor-only vs Text-only: {improvement:+.3f} AUC")

                if improvement > 0.05:
                    print("✓ Factors provide meaningful signal beyond text heuristics!")
                elif improvement > 0:
                    print("~ Factors provide marginal additional signal")
                else:
                    print("✗ Factors don't improve over text heuristics")
        else:
            print("Not enough notes with factor values for analysis")

    return results


def analyze_factor_distributions(humor_df: pd.DataFrame, factors_df: pd.DataFrame):
    """Analyze how factors differ between CRH and NOT_HELPFUL humor notes."""

    labeled = humor_df[humor_df['status'].isin(['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL'])].copy()
    merged = labeled.merge(factors_df, on='noteId', how='left')

    factor_cols = [c for c in factors_df.columns if c != 'noteId']

    print("\n" + "="*60)
    print("FACTOR DISTRIBUTIONS BY OUTCOME")
    print("="*60)

    crh = merged[merged['status'] == 'CURRENTLY_RATED_HELPFUL']
    not_helpful = merged[merged['status'] == 'CURRENTLY_RATED_NOT_HELPFUL']

    print(f"\n{'Factor':<30} {'CRH mean':>12} {'NOT_H mean':>12} {'Diff':>10}")
    print("-" * 66)

    for col in factor_cols:
        crh_mean = crh[col].mean()
        nh_mean = not_helpful[col].mean()
        diff = crh_mean - nh_mean

        if not np.isnan(crh_mean) and not np.isnan(nh_mean):
            marker = " ***" if abs(diff) > 0.1 else ""
            print(f"{col:<30} {crh_mean:>12.4f} {nh_mean:>12.4f} {diff:>+10.4f}{marker}")


if __name__ == "__main__":
    import sys

    # Paths
    base_path = Path(__file__).parent.parent.parent / "data"
    humor_path = base_path / "humor_candidates_manual_review.tsv"

    # Try multiple possible scored notes paths
    scored_notes_paths = [
        base_path / "scored_notes.tsv",
        base_path / "noteStatusHistory-00000.tsv",
        base_path / "notes-00000.tsv",
    ]

    print("Loading humor candidates...")
    humor_df = load_humor_candidates(str(humor_path))
    print(f"Loaded {len(humor_df)} humor candidates")

    # Try to load factors
    factors_df = None
    for path in scored_notes_paths:
        if path.exists():
            print(f"\nLoading factors from {path.name}...")
            try:
                factors_df = load_note_factors(str(path))
                if len(factors_df.columns) > 1:  # Has actual factor columns
                    print(f"  Found {len(factors_df.columns) - 1} factor columns")
                    break
                else:
                    factors_df = None
            except Exception as e:
                print(f"  Error: {e}")
                factors_df = None

    # Run comparison
    print("\n" + "="*60)
    print("RUNNING CLASSIFICATION COMPARISON")
    print("="*60)

    results = run_classification_comparison(humor_df, factors_df)

    if factors_df is not None:
        analyze_factor_distributions(humor_df, factors_df)
