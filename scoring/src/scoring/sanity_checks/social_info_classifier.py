"""
Social vs Info Classifier for CRH Notes

Goal: Show that factors uniquely identify "joke/social utility" notes
among CRH notes, beyond what text heuristics capture.

Pipeline:
1. Restrict to CRH notes only
2. Compute text-based info_score and social_score
3. Use hand-labeled seed set to train separator
4. Compare: factors-only vs text-only vs combined
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')


def compute_info_score(summary: str) -> dict:
    """
    Info score: measures informational content.
    High = factual, sourced, evidence-based.
    """
    if pd.isna(summary):
        summary = ""
    text = str(summary).lower()

    # URL/source presence
    has_url = int('http' in text or 'www.' in text)
    url_count = text.count('http') + text.count('www.')

    # Wikipedia/reliable sources
    has_wikipedia = int('wikipedia' in text)
    has_reliable = int(any(s in text for s in ['reuters', 'bbc', 'snopes', 'factcheck', 'apnews']))

    # Numbers (dates, statistics, measurements)
    numbers = re.findall(r'\b\d+\.?\d*\b', summary)
    number_count = len(numbers)
    has_year = int(any(re.match(r'(19|20)\d{2}', n) for n in numbers))

    # Factual claim markers
    factual_markers = ['according to', 'confirmed', 'states that', 'reported',
                       'evidence', 'study', 'research', 'data', 'official',
                       'source:', 'per', 'via']
    factual_count = sum(1 for m in factual_markers if m in text)

    # Hedging (shows careful reasoning)
    hedging = ['appears to', 'seems to', 'likely', 'possibly', 'may be', 'could be']
    hedging_count = sum(1 for h in hedging if h in text)

    # Named entities (rough proxy)
    # Capitalized words that aren't at sentence start
    caps_words = len(re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', summary))

    # Quote presence
    has_quotes = int('"' in summary or '"' in summary or "'" in summary)

    # Length (longer often = more info)
    word_count = len(summary.split())

    # Aggregate info score (simple weighted sum)
    info_score = (
        has_url * 2 +
        url_count * 0.5 +
        has_wikipedia * 2 +
        has_reliable * 2 +
        number_count * 0.5 +
        has_year * 1 +
        factual_count * 1 +
        hedging_count * 0.5 +
        caps_words * 0.2 +
        has_quotes * 0.5 +
        min(word_count / 20, 2)  # Cap length contribution
    )

    return {
        'info_score': info_score,
        'info_has_url': has_url,
        'info_url_count': url_count,
        'info_has_wikipedia': has_wikipedia,
        'info_number_count': number_count,
        'info_factual_markers': factual_count,
        'info_word_count': word_count,
    }


def compute_social_score(summary: str) -> dict:
    """
    Social score: measures social/humor content.
    High = punchy, sarcastic, memey, entertaining.
    """
    if pd.isna(summary):
        summary = ""
    text = str(summary).lower()
    original = str(summary)

    # Brevity (jokes are often short)
    word_count = len(summary.split())
    is_short = int(word_count <= 10)
    is_very_short = int(word_count <= 5)

    # Punchline structure (short + period/link)
    ends_with_period = int(original.rstrip().endswith('.'))

    # Sarcasm markers
    sarcasm = ['clearly', 'obviously', 'really', 'seriously', 'literally',
               'of course', 'sure', 'right', 'totally', 'definitely']
    sarcasm_count = sum(1 for s in sarcasm if s in text)

    # Rhetorical questions
    rhetorical = text.count('?')

    # Second-person address (calling out the poster)
    second_person = ['you', 'your', "you're", 'yourself']
    second_person_count = sum(1 for p in second_person if p in text.split())

    # Meme phrases
    meme_phrases = ['my brother in christ', 'bro', 'dude', 'bruh', 'lmao', 'lol',
                    'cope', 'seethe', 'touch grass', 'get a life', 'ratio',
                    'based', 'cringe', 'yikes', 'oof', 'rip', 'f in the chat',
                    'nnn', 'no note needed', 'skill issue']
    meme_count = sum(1 for m in meme_phrases if m in text)

    # Emoji presence
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]')
    emoji_count = len(emoji_pattern.findall(summary))

    # Clapback structure: short declarative + source
    # e.g., "No." + wikipedia link
    is_clapback = int(is_short and ('http' in text or 'wikipedia' in text))

    # Deadpan (very short + factual correction)
    deadpan_starters = ['no', 'yes', 'false', 'true', 'wrong', 'correct', 'nope', 'yep']
    starts_deadpan = int(any(text.strip().startswith(d) for d in deadpan_starters))

    # Exasperation markers
    exasperation = ['shut up', 'stfu', 'come on', 'oh my god', 'jesus', 'christ',
                    'for fucks sake', 'ffs', 'wtf', 'what the']
    exasperation_count = sum(1 for e in exasperation if e in text)

    # Exclamation (emphasis)
    exclamation_count = original.count('!')

    # All caps words (emphasis)
    all_caps = len(re.findall(r'\b[A-Z]{2,}\b', original))

    # Pop culture references (rough)
    pop_culture = ['star trek', 'star wars', 'rick astley', 'rickroll', 'meme',
                   'knowyourmeme', 'urban dictionary']
    pop_count = sum(1 for p in pop_culture if p in text)

    # Aggregate social score
    social_score = (
        is_short * 2 +
        is_very_short * 2 +
        is_clapback * 3 +
        starts_deadpan * 2 +
        sarcasm_count * 1 +
        rhetorical * 0.5 +
        meme_count * 2 +
        emoji_count * 1 +
        exasperation_count * 1 +
        exclamation_count * 0.3 +
        all_caps * 0.5 +
        pop_count * 2
    )

    return {
        'social_score': social_score,
        'social_is_short': is_short,
        'social_is_very_short': is_very_short,
        'social_is_clapback': is_clapback,
        'social_starts_deadpan': starts_deadpan,
        'social_meme_count': meme_count,
        'social_emoji_count': emoji_count,
        'social_sarcasm_count': sarcasm_count,
    }


def load_crh_notes(scored_notes_path: str) -> pd.DataFrame:
    """Load all CRH notes with factors and summaries."""
    df = pd.read_csv(scored_notes_path, sep='\t', low_memory=False)

    # Filter to CRH only
    status_col = 'finalRatingStatus' if 'finalRatingStatus' in df.columns else 'currentStatus'
    crh = df[df[status_col] == 'CURRENTLY_RATED_HELPFUL'].copy()

    # Keep key columns
    factor_cols = ['noteId',
                   'coreNoteIntercept', 'coreNoteFactor1',
                   'internalNoteFactor2', 'internalNoteFactor3',
                   'internalNoteFactor4', 'internalNoteFactor5', 'internalNoteFactor6']
    keep = [c for c in factor_cols if c in crh.columns]
    crh = crh[keep]

    # Load summaries from notes file
    notes_path = Path(scored_notes_path).parent / "notes-00000.tsv"
    if notes_path.exists():
        print(f"Loading summaries from {notes_path.name}...")
        notes = pd.read_csv(notes_path, sep='\t', low_memory=False, usecols=['noteId', 'summary'])
        crh = crh.merge(notes, on='noteId', how='left')

    return crh


def load_seed_labels(humor_path: str, crh_notes: pd.DataFrame) -> pd.DataFrame:
    """
    Create seed labels:
    - humor_candidates marked CRH = "social" (label=1)
    - Sample of other CRH notes = "info" (label=0)
    """
    # Load humor candidates
    humor = pd.read_csv(humor_path, sep='\t')
    humor.columns = ['noteId', 'status', 'summary_humor']

    # Filter to CRH humor notes - use int for matching
    humor_crh = humor[humor['status'] == 'CURRENTLY_RATED_HELPFUL']['noteId'].astype(int).tolist()
    humor_crh_set = set(humor_crh)

    crh_notes = crh_notes.copy()

    # Ensure noteId is int for matching
    crh_notes['noteId'] = crh_notes['noteId'].astype(int)

    # Label: 1 = social/joke, 0 = info
    crh_notes['is_social'] = crh_notes['noteId'].isin(humor_crh_set).astype(int)

    print(f"  Humor CRH noteIds: {len(humor_crh_set)}")
    print(f"  Matched in scored_notes: {crh_notes['is_social'].sum()}")

    return crh_notes


def run_classification(crh_notes: pd.DataFrame):
    """
    Compare classifiers:
    1. Text heuristics (info_score, social_score features)
    2. Factor features only
    3. Combined
    """

    # Compute text scores
    print("Computing info and social scores...")
    info_features = crh_notes['summary'].apply(compute_info_score).apply(pd.Series)
    social_features = crh_notes['summary'].apply(compute_social_score).apply(pd.Series)

    # Combine
    crh_notes = pd.concat([crh_notes, info_features, social_features], axis=1)

    # Filter to notes with labels and factors
    factor_cols = ['coreNoteIntercept', 'coreNoteFactor1',
                   'internalNoteFactor2', 'internalNoteFactor3',
                   'internalNoteFactor4', 'internalNoteFactor5', 'internalNoteFactor6']
    factor_cols = [c for c in factor_cols if c in crh_notes.columns]

    # Need factor values
    has_factors = crh_notes[factor_cols].notna().all(axis=1)
    labeled = crh_notes[has_factors].copy()

    print(f"\nCRH notes with factors: {len(labeled)}")
    print(f"  Social/joke: {labeled['is_social'].sum()}")
    print(f"  Info: {(~labeled['is_social'].astype(bool)).sum()}")

    if labeled['is_social'].sum() < 5:
        print("Not enough social labels - need more hand-labeled examples")
        return None

    # Features
    text_cols = [c for c in crh_notes.columns if c.startswith('info_') or c.startswith('social_')]

    X_text = labeled[text_cols].values
    X_factors = labeled[factor_cols].values
    X_combined = np.hstack([X_text, X_factors])
    y = labeled['is_social'].values

    # Normalize
    scaler_t = StandardScaler()
    scaler_f = StandardScaler()
    scaler_c = StandardScaler()

    X_text_scaled = scaler_t.fit_transform(X_text)
    X_factors_scaled = scaler_f.fit_transform(X_factors)
    X_combined_scaled = scaler_c.fit_transform(X_combined)

    # Cross-validation
    n_social = y.sum()
    n_splits = min(5, n_social)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    # 1. Text heuristics only
    print("\n" + "="*60)
    print("1. TEXT HEURISTICS (info_score + social_score features)")
    print("="*60)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_text_scaled, y, cv=cv, scoring='roc_auc')
    results['text'] = {'auc': scores.mean(), 'std': scores.std()}
    print(f"AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    clf.fit(X_text_scaled, y)
    print("\nTop text features:")
    importance = sorted(zip(text_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in importance[:6]:
        print(f"  {feat}: {coef:+.3f}")

    # 2. Factors only
    print("\n" + "="*60)
    print("2. FACTORS ONLY (intercept + 6 factors)")
    print("="*60)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_factors_scaled, y, cv=cv, scoring='roc_auc')
    results['factors'] = {'auc': scores.mean(), 'std': scores.std()}
    print(f"AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    clf.fit(X_factors_scaled, y)
    print("\nFactor importance:")
    importance = sorted(zip(factor_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in importance:
        print(f"  {feat}: {coef:+.3f}")

    # 3. Combined
    print("\n" + "="*60)
    print("3. TEXT + FACTORS")
    print("="*60)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_combined_scaled, y, cv=cv, scoring='roc_auc')
    results['combined'] = {'auc': scores.mean(), 'std': scores.std()}
    print(f"AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Can factors identify social/joke notes?")
    print("="*60)
    print(f"\n{'Model':<25} {'AUC':>10}")
    print("-" * 37)
    for name, r in results.items():
        print(f"{name:<25} {r['auc']:>10.3f}")

    improvement = results['factors']['auc'] - results['text']['auc']
    print(f"\nFactors vs Text: {improvement:+.3f} AUC")

    if improvement > 0.05:
        print("✓ Factors provide UNIQUE signal for identifying social/joke notes!")
    elif improvement > 0:
        print("~ Factors add marginal value")
    else:
        print("✗ Text heuristics sufficient - factors don't add value")

    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE NOTES BY PREDICTED SOCIAL SCORE")
    print("="*60)

    clf.fit(X_factors_scaled, y)
    probs = clf.predict_proba(X_factors_scaled)[:, 1]
    labeled['social_prob'] = probs

    print("\nTop 10 'social/joke' predictions:")
    top_social = labeled.nlargest(10, 'social_prob')
    for _, row in top_social.iterrows():
        label = "✓ LABELED SOCIAL" if row['is_social'] else "? unlabeled"
        print(f"  [{row['social_prob']:.2f}] {label}: {str(row['summary'])[:60]}...")

    print("\nTop 10 'info' predictions:")
    top_info = labeled.nsmallest(10, 'social_prob')
    for _, row in top_info.iterrows():
        label = "✓ LABELED SOCIAL" if row['is_social'] else "  info"
        print(f"  [{row['social_prob']:.2f}] {label}: {str(row['summary'])[:60]}...")

    return results, labeled


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent / "data"

    print("Loading CRH notes...")
    crh = load_crh_notes(str(base_path / "scored_notes.tsv"))
    print(f"Loaded {len(crh)} CRH notes")

    print("\nLoading seed labels from humor candidates...")
    labeled = load_seed_labels(str(base_path / "humor_candidates_manual_review.tsv"), crh)

    run_classification(labeled)
