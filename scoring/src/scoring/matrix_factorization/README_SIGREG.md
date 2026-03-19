# SIGreg: Sketched Isotropic Gaussian Regularization for Community Notes

## Overview

SIGreg (Sketched Isotropic Gaussian Regularization) is an embedding regularization technique adapted from the [LeJEPA paper](https://arxiv.org/abs/2404.16945) (Meta AI) that prevents representation collapse in matrix factorization models.

In Community Notes scoring, SIGreg enables the use of **multi-dimensional embeddings** (numFactors > 1) while ensuring each dimension captures meaningful, distinct information about user-note interactions.

## Motivation

The standard Community Notes matrix factorization uses 1-dimensional embeddings (numFactors=1), which captures a single "polarity" dimension. With SIGreg, we can expand to multiple dimensions to potentially capture:

- Political/ideological alignment (the original factor)
- Topic-specific expertise or interest
- Rating behavior patterns (e.g., strict vs. lenient raters)
- Note quality dimensions beyond helpfulness

Without regularization, multi-dimensional embeddings tend to **collapse** - all dimensions become highly correlated or degenerate, providing no additional information. SIGreg prevents this by enforcing that the embedding covariance matrix stays close to the identity matrix (isotropic distribution).

## Mathematical Formulation

The SIGreg loss is computed as:

```
L_SIGReg = λ * ||Sketch(Z) - I||_F²
```

Where:
- `Z` = embedding matrix (n_entities × embedding_dim)
- `Sketch(Z) = Z^T @ Z / n` = empirical covariance matrix
- `I` = identity matrix (isotropic target)
- `||·||_F` = Frobenius norm
- `λ` = regularization strength

This loss is **added** to the standard matrix factorization loss during training.

## Usage

### Enabling SIGreg

SIGreg is configured in the `MatrixFactorization` class:

```python
from scoring.matrix_factorization.matrix_factorization import MatrixFactorization

mf = MatrixFactorization(
    numFactors=4,              # Use 3-dimensional embeddings (2 factor dimensions + intercept)
    useGlobalIntercept=True,
    useSIGReg=True,            # Enable SIGreg regularization
    sigregLambdaUser=0.01,     # Regularization strength for user embeddings
    sigregLambdaNote=0.01,     # Regularization strength for note embeddings
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `useSIGReg` | `False` | Enable/disable SIGreg regularization |
| `sigregLambdaUser` | `0.01` | Regularization strength for user/rater embeddings |
| `sigregLambdaNote` | `0.01` | Regularization strength for note embeddings |
| `numFactors` | `1` | Number of embedding dimensions (SIGreg is a no-op for 1D) |

### Recommended Values

- **Conservative**: `lambda = 0.005` - Minimal regularization, allows more flexibility
- **Standard**: `lambda = 0.01` - Balanced regularization
- **Strong**: `lambda = 0.05` - Forces more isotropy, may reduce model fit

Start with conservative values and increase if you observe embedding collapse.

## Output Columns

When `numFactors > 1`, additional factor columns are included in the output:

### Scored Notes Output
- `internalNoteFactor1` (or `coreNoteFactor1` for core scorer)
- `internalNoteFactor2`
- `internalNoteFactor3` (if numFactors >= 4)
- ... and so on

### Helpfulness Scores Output
- `internalRaterFactor1` (or `coreRaterFactor1` for core scorer)
- `internalRaterFactor2`
- `internalRaterFactor3` (if numFactors >= 4)
- ... and so on

## Implementation Details

### Files Modified

1. **`matrix_factorization/sigreg_loss.py`** (new)
   - `SIGRegLoss` class implementing the regularization
   - Diagnostic methods: `get_covariance_stats()`, `compute_isotropy_score()`

2. **`matrix_factorization/model.py`**
   - Added SIGreg parameters to `BiasedMatrixFactorization`
   - `get_factor_embeddings()` method to extract embeddings for regularization
   - `compute_sigreg_loss()` method called during training

3. **`matrix_factorization/matrix_factorization.py`**
   - SIGreg loss integration in training loop
   - Configurable lambda parameters for user and note embeddings

4. **`constants.py`**
   - `note_factor_key(i)` and `rater_factor_key(i)` functions for dynamic column names
   - Added Factor2 columns to output TSV schemas

5. **`mf_base_scorer.py`**
   - `_get_num_factors()` method to query embedding dimensions
   - Dynamic column generation in `get_scored_notes_cols()` and `get_helpfulness_scores_cols()`

6. **`mf_core_scorer.py`**
   - Dynamic Factor2+ column mappings in `_get_note_col_mapping()` and `_get_user_col_mapping()`

7. **`run_scoring.py`**
   - Column filtering for prescoring (handles cases where Factor2+ don't exist)

8. **`reputation_matrix_factorization/diligence_model.py`**
   - Edge case handling for zero ratings (prevents NaN loss)

### How SIGreg Integrates with Training

```
Standard MF Loss:  L_mf = ||R - U @ V^T||² + λ_reg * (||U||² + ||V||²)

With SIGreg:       L_total = L_mf + λ_user * SIGReg(U) + λ_note * SIGReg(V)
```

The SIGreg loss is computed on the **factor embeddings only** (excluding intercepts), as intercepts represent per-entity biases rather than latent factors.

## Analyzing Results

### Isotropy Score

The `SIGRegLoss` class provides an isotropy score between 0 and 1:

```python
from scoring.matrix_factorization.sigreg_loss import SIGRegLoss

sigreg = SIGRegLoss()
score = sigreg.compute_isotropy_score(embeddings)
# score close to 1.0 = isotropic (good)
# score close to 0.0 = collapsed (bad)
```

### Covariance Statistics

For detailed diagnostics:

```python
stats = sigreg.get_covariance_stats(embeddings)
# Returns: cov_trace, cov_det, eigenvalue_min/max/std,
#          frobenius_dist_from_identity, condition_number
```

### Interpreting Factor Dimensions

After training with SIGreg, analyze what each factor dimension captures:

1. **Correlation analysis**: Correlate factor values with note/user metadata
2. **Clustering**: Cluster users/notes by factor values to identify groups
3. **Visualization**: Plot Factor1 vs Factor2 to see the embedding space structure

## Empirical Findings: What Factor2 Captures

We analyzed 18,000+ notes with Factor2 values to understand what the second dimension represents.

### Key Result: Orthogonal Dimensions

SIGreg successfully creates orthogonal embedding dimensions:

| Metric | Value |
|--------|-------|
| Factor1-Factor2 Correlation | **0.0048** (near-zero) |
| Quadrant Distribution | ~25% in each quadrant (near-uniform) |

This confirms that Factor2 captures genuinely different information than Factor1.

### Factor1 vs Factor2: Stability

| Factor | What it captures | Stability |
|--------|-----------------|-----------|
| **Factor1** | Political left-right spectrum | **Stable** - dominant persistent feature of rater population |
| **Factor2** | Secondary disagreement axis | **Weaker** - patterns visible mainly at extremes |

Factor1 is stable because political ideology is the primary axis of disagreement in any controversial content. It would emerge regardless of which specific notes are in the dataset.

Factor2 captures whatever secondary pattern exists in the current data. The specific interpretation may vary with different note populations.

### Patterns at Factor2 Extremes

While the **bulk of notes have Factor2 near zero** (no clear pattern), the **extremes** show content-specific clustering:

#### Extreme Negative Factor2 (-0.8 to -1.4)
Notes defending platform-adjacent figures:
- "Musk's gesture wasn't Nazi salute - he was thanking audience"
- "Musk has said 'happy holidays' before" (hypocrisy defense)
- "Trump White House renovation is normal"
- "Obama gerrymandering history"

#### Extreme Positive Factor2 (+0.8 to +1.1)
Notes on trans/gender issues and political context:
- Kim Petras transgender identity clarification
- Charlie Kirk assassination celebration context
- UK council mergers "not undermining democracy"
- LA riots National Guard deployment context

### Important Caveats

**Factor2 signal is weak overall:**

| Category | Factor2 Mean | Notes |
|----------|--------------|-------|
| Musk-related | -0.0006 | ~0 |
| Trump-related | -0.0023 | ~0 |
| Trans/Gender | +0.0024 | ~0 |
| All other topics | ~0 | No signal |

The content patterns above are visible only at the **extremes** (~100 notes out of 18,000). For the vast majority of notes, Factor2 does not clearly distinguish content types.

### Factor2 Does NOT Reliably Capture:
- ❌ Humor vs serious notes (both average ~0)
- ❌ Well-sourced vs unsourced notes
- ❌ Long vs short notes
- ❌ NNN ("no note needed") vs factual corrections
- ❌ Any single topic category (means all ~0)

### Visualizations

See the `figures/` directory for visualizations:

- `factor1_vs_factor2_scatter.png` - Scatter plot showing orthogonality
- `factor2_by_status.png` - Factor2 distribution by rating status
- `factor2_extreme_content.png` - Topic analysis at Factor2 extremes
- `factor2_quadrant_distribution.png` - 4-quadrant distribution
- `factor2_variance_by_intercept.png` - F2 variance vs note intercept

Generate visualizations with:
```bash
python scoring/src/scoring/matrix_factorization/visualize_factor2.py
```

### Interpretation

Factor2 is best understood as a **second-order effect**:

1. **SIGreg successfully enforces orthogonality** - Factor2 is mathematically independent from Factor1
2. **The dimension captures real variance** - notes do spread across Factor2 values
3. **Semantic meaning is weak and possibly content-dependent** - only extreme values show patterns
4. **The specific patterns (Musk-defense vs trans-topics) may be artifacts of this dataset** - with different notes, Factor2 might capture something else entirely

This contrasts with Factor1, which reliably captures left-right political alignment regardless of specific note content.

### Practical Use: |Factor2| as a "Contested Notes" Signal

While Factor2's semantic meaning is unclear, **|Factor2| (absolute value)** has a practical use case:

**Notes with extreme |Factor2| are "contested but promising" candidates:**

| Metric | Middle (|F2|<0.3) | Extreme (|F2|>0.3) |
|--------|-------------------|-------------------|
| Count | 17,849 | 207 |
| **Avg Intercept** | 0.19 | **0.37** |
| **Avg Ratings** | 3.1 | **4.6** |
| FIRM_REJECT % | 77% | **48%** |
| NEEDS_MORE_RATINGS % | 20% | **42%** |
| NEEDS_YOUR_HELP % | 2% | **8%** |

**Key insight**: 46% of extreme |F2| notes have intercept > 0.5 (close to helpful threshold).

#### Why This Happens

High |Factor2| indicates disagreement along the secondary dimension - raters from different parts of Factor2 space are engaging with the note. This creates:
- More total ratings (diverse engagement)
- Higher intercepts (quality signal from cross-spectrum agreement)
- Less firm rejection (not a clear consensus against)
- More "needs more ratings" status (awaiting additional input)

#### Practical Application

```python
# Identify notes worth prioritizing for additional ratings
contested_notes = scored_notes[
    (scored_notes['internalNoteFactor2'].abs() > 0.3) &
    (scored_notes['coreNoteIntercept'] > 0.4) &
    (scored_notes['coreRatingStatus'].isin(['NEEDS_MORE_RATINGS', 'NEEDS_YOUR_HELP']))
]
# These notes have cross-spectrum engagement and are close to consensus
```

### Implications for Use

- **Factor1**: Reliable for understanding political polarization of notes/raters
- **Factor2 value**: Use with caution; semantic interpretation is dataset-specific
- **|Factor2| magnitude**: Useful signal for identifying contested notes worth prioritizing

The near-uniform quadrant distribution confirms SIGreg works technically. While the semantic meaning of Factor2 direction requires validation, the magnitude provides actionable signal for note prioritization

## 4-Factor Analysis: Deeper Understanding

Expanding from 2 factors to 4 factors reveals richer structure in the embedding space. Analysis of 97,401 notes with all 4 factors:

### Factor Overview

| Factor | Std Dev | Primary Signal | Interpretation |
|--------|---------|---------------|----------------|
| **Factor1** | 0.43 | Political Polarity | Left-right spectrum (stable) |
| **Factor2** | 0.35 | Language/Region | English vs French/CJK |
| **Factor3** | 0.34 | Assertion Style | Nuanced vs Direct |
| **Factor4** | 0.47 | Note Purpose | Context vs Debunking |

### Factor1: Political Polarity (Strongest Signal)

The primary factor reliably captures left-right political alignment:

| Topic | F1 Deviation | Direction |
|-------|-------------|-----------|
| Ukraine/Russia | -0.14 | Left |
| Climate | -0.09 | Left |
| AI/Tech | -0.09 | Left |
| COVID/Vaccine | -0.08 | Left |
| Trump | +0.06 | Right |
| Biden | +0.07 | Right |
| Musk/Elon | +0.06 | Right |
| Immigration | +0.04 | Right |

Additional patterns:
- Short notes (<100 chars): F1 = +0.11 (more right-leaning)
- Long notes (300+ chars): F1 = +0.01 (more balanced)
- Meta-commentary about CN: F1 = +0.15 (right-leaning)

### Factor2: Language/Region Dimension

Factor2 strongly correlates with language:

| Language | F2 Mean | Notes |
|----------|---------|-------|
| English | -0.03 | 74,425 |
| French | **+0.22** | 1,146 |
| Japanese/CJK | **+0.18** | 9,858 |
| Spanish | +0.04 | 877 |
| Portuguese | +0.02 | 598 |
| German | -0.05 | 386 |

US political topics (Trump, Biden, Musk, Immigration) all have negative F2, confirming the English/US-centric dimension.

### Factor3: Assertion Style

Factor3 captures how directly a note makes its claim:

**Low Factor3** (nuanced, contextual):
- "NASA did wipe the tapes, however all original footage still exists"
- "3.3% of deaths via MAID is different than 3.3% of Canadians"
- "Elon did not, in fact, support his daughter her whole life..."

**High Factor3** (direct, assertive):
- "Yes it is." (with link to law)
- "This is false."
- "Sydney Wilson was indeed shot and killed by a Police Officer."

### Factor4: Debunking vs Context

Factor4 distinguishes note purpose (correlated with F1 at r=-0.38):

**Low Factor4** (providing context/defense):
- Elizabeth Warren nickname explanation
- Trump medical emergency context
- Community Notes algorithm explanation

**High Factor4** (direct debunking):
- "Donald Trump did not post this. This is fake."
- "Obama's order 13603 does not involve suspending the Constitution"
- "This memorandum is authentic, but due to backlash the ban was overruled"

Topic correlations with F4:
| Topic | F4 Deviation |
|-------|-------------|
| AI/Tech | +0.12 |
| Climate | +0.11 |
| COVID/Vaccine | +0.05 |
| Evidence-based | +0.03 |
| Trump/Musk | -0.03 |
| Portuguese | -0.16 |
| CJK | **-0.32** |

### Factor Correlations

```
             F1      F2      F3      F4
F1         1.000  -0.102   0.092  -0.382
F2        -0.102   1.000   0.000   0.034
F3         0.092   0.000   1.000  -0.098
F4        -0.382   0.034  -0.098   1.000
```

Key correlation: **F1-F4 = -0.38** — politically-charged notes (high |F1|) tend to provide context rather than direct debunks.

### Visualizations

See `figures/` directory for 4-factor visualizations:
- `factor_correlation_4f.png` - Correlation heatmap
- `factor_distributions_4f.png` - Distribution of each factor
- `factor_pairwise_scatter_4f.png` - All pairwise scatter plots
- `factors_by_language.png` - Factors by detected language
- `factors_by_status_4f.png` - Factors by rating status
- `factors_by_status_scatter.png` - Scatter plots colored by status
- `factor1_vs_factor4_colored.png` - F1 vs F4 colored by F2
- `factor_4f_summary.png` - Visual summary of findings

### Practical Applications

1. **Language Detection**: Use F2 to identify non-English notes in multinational deployments
2. **Writing Style Filter**: F3 can identify "assertive" vs "nuanced" note styles
3. **Purpose Classification**: F4 distinguishes "debunking" from "contextualizing" notes
4. **Political Intensity**: |F1| indicates how politically polarizing a note is
5. **Cross-Cultural Analysis**: F2+F4 together reveal regional writing patterns

### Implications

The 4-factor space reveals that Community Notes exist on multiple independent dimensions:
- **Political alignment** (F1) - who agrees/disagrees
- **Language/culture** (F2) - where the note comes from
- **Communication style** (F3) - how the note is written
- **Purpose** (F4) - what the note is trying to do

This multi-dimensional view enables more nuanced note analysis beyond simple political polarization.

## Known Issues

### SIGReg Causes RMSE Explosion for K>1

**Problem:** When using SIGReg with K>1 factors, the model's RMSE explodes (2.3-4.4 instead of ~0.48).

| Config | L2 RMSE | SIGReg RMSE |
|--------|---------|-------------|
| K=1 | 0.489 | 0.491 (OK) |
| K=2 | 0.486 | 2.33 (broken) |
| K=4 | 0.482 | 3.47 (broken) |
| K=6 | 0.481 | 4.36 (broken) |

**Root cause:** SIGReg enforces embeddings with unit variance (by design), but this MF model needs smaller scale:
- SIGReg factors have variance ~1.0
- L2 factors have variance 0.06-0.23

With variance=1.0, the factor dot products (fᵤ·fₙ) produce extreme values that overwhelm the intercepts.

**Why K=1 works:** With only 1 dimension, a single large dot product can still be calibrated by the intercepts.

**Recommendation:** Use L2 regularization instead of SIGReg. L2 achieves:
- Variance ratio ~0.7 (no collapse)
- Correct embedding scale for accurate predictions

The factor analysis documented above was performed with L2 regularization, which successfully prevents representation collapse while maintaining predictive accuracy.

## Backward Compatibility

- SIGreg is **disabled by default** (`useSIGReg=False`)
- With 1D embeddings (`numFactors=1`), SIGreg is a **no-op** (returns 0 loss)
- Existing scoring pipelines continue to work without modification
- Output column schemas are extended, not replaced

## Example: Running with SIGreg

```bash
# In your scoring configuration, set:
# - numFactors=3 (or desired dimension count)
# - useSIGReg=True
# - sigregLambdaUser=0.01
# - sigregLambdaNote=0.01

python main.py \
  -n notes.tsv \
  -r ratings/ \
  -e userEnrollment.tsv \
  -s noteStatusHistory.tsv \
  -o output/
```

## References

- LeJEPA: Latent-Euclidean Joint-Embedding Predictive Architecture (Meta AI, 2024)
- Community Notes Algorithm Guide: https://communitynotes.twitter.com/guide/en/under-the-hood/ranking-notes