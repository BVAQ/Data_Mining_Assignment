# Complete Pipeline Explanation: Raw Data → submission_v3_tv_q.csv

## Overview

This document explains the complete process that produced the selected submission `submission_v3_tv_q.csv`, which achieved a **public QWK of 0.75557** and a **private QWK of 0.73479** on the DM252 ASP Paper Ordinal Classification competition. The pipeline transforms raw paper metadata (title, venue, year, authors, DOI) into ordinal relevance labels 1–5 using a multi-stage stacking ensemble with 23 features.

---

## Phase 1: Raw Data Understanding

### What Each File Contains

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train.csv` | 2,494 | 7 | Training data with labels |
| `public_test.csv` | 298 | 6 | Public leaderboard test set |
| `private_test.csv` | 298 | 6 | Private leaderboard test set |
| `Test_Submission.csv` | 596 | 2 | Submission template (id + Label) |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique paper identifier |
| `title` | string | Paper title |
| `venue` | string | Publication venue (conference name) |
| `year` | int | Publication year |
| `authors` | string | Author names (comma-separated), 192 missing in train |
| `doi` | string | DOI or Semantic Scholar URL |
| `Label` | int (1–5) | Target: ordinal relevance to Answer Set Programming |

### The Prediction Target

The label represents how closely related a paper is to Answer Set Programming (ASP) and related areas of logic programming:
- **Label 1**: Not relevant or peripherally relevant (e.g., proceedings front-matter, unrelated topics)
- **Label 5**: Highly relevant (core ASP research)
- **Labels 2–4**: Intermediate degrees of relevance

This is an **ordinal classification** problem because the labels have a natural ordering: misclassifying a Label-5 paper as Label-1 is worse than misclassifying it as Label-4.

### Why QWK (Quadratic Weighted Kappa)

Quadratic Weighted Kappa is the appropriate metric because:
1. It penalizes predictions proportionally to the square of the distance from the true label
2. It accounts for agreement expected by chance
3. It ranges from -1 (systematic disagreement) through 0 (chance agreement) to 1 (perfect agreement)
4. It naturally handles the ordinal structure, punishing large mis-rankings more than small ones

### Label Distribution (Train)

| Label | Count | Percentage |
|-------|-------|------------|
| 1 | 903 | 36.2% |
| 2 | 514 | 20.6% |
| 3 | 438 | 17.6% |
| 4 | 367 | 14.7% |
| 5 | 272 | 10.9% |

The distribution is skewed toward Label 1, with a monotonically decreasing frequency.

---

## Phase 2: Data Loading and Inspection

### Loading Process

All four CSV files are loaded using pandas. The public and private test sets are concatenated into a single test DataFrame of 596 rows.

### Checks Performed

1. **Shape validation**: Train has 2,494 rows with 7 columns; each test set has 298 rows with 6 columns (no Label column).
2. **Missing values**: Only `authors` has missing values (192 in train, 19 in public test, 22 in private test). All other columns are complete.
3. **Duplicate IDs**: No duplicate IDs in any dataset.
4. **Venue analysis**: Train has 5 venues (`cav`, `iclp`, `kr`, `lics`, `lpnmr`); test has only 4 (`cav`, `kr`, `lics`, `lpnmr`). The critical finding is that **`iclp` (456 train rows) is absent from the test set**, meaning 18.3% of training data comes from a venue not represented in test.
5. **Label range**: All labels are integers from 1 to 5.
6. **Submission ID consistency**: The 596 IDs in `Test_Submission.csv` match exactly with the combined public + private test IDs.

### Key Issues Found

- **Venue shift**: `iclp` absent from test creates a distribution shift. The test-venue subset of train (2,038 rows from `cav`, `kr`, `lics`, `lpnmr`) provides a more representative validation signal.
- **Missing authors**: 7.7% of train rows lack author information, but this was found to be a weak signal anyway.
- **Venue-label correlation**: Each venue has a distinct mean label (`lics`=1.94, `cav`=2.12, `kr`=2.73, `iclp`=2.94, `lpnmr`=3.14), making venue a strong prior for the ordinal target.

---

## Phase 3: Data Preprocessing

### Missing Value Handling

| Column | Strategy |
|--------|----------|
| `title` | Fill with empty string `""` |
| `venue` | Fill with `"unknown"` (no actual missing values exist) |
| `authors` | Fill with empty string `""` (used only in S2 pipeline) |
| `year` | Convert to numeric, fill missing with 2020 (used only in S2) |
| `doi` | Fill with empty string `""` (minimal direct use) |

### Text Cleaning

The `title` field undergoes minimal cleaning because the pipeline relies on TF-IDF and transformer tokenizers that handle raw text well:
- In the GM pipeline, a `norm_text()` function lowercases, replaces HTML entities (`&amp;`, `&apos;`), removes special characters, and collapses whitespace
- In the TF-IDF Ridge features, raw titles are used directly with `sublinear_tf=True` for better term frequency scaling
- For transformer fine-tuning, titles are prepended with venue information: `"venue: <venue>. <title>"`

### Metadata Normalization

- **Venue**: Used directly as categorical; also used to construct venue-prefixed input text for transformers
- **Year**: Converted to integer, normalized as `year - 2016` for numerical features
- **Authors**: Count of authors (`n_auth`) extracted as a metadata feature in the S2 pipeline
- **DOI**: Boolean feature `has_doi10` (whether DOI starts with `10.`) in the S2 pipeline

### Train/Test Consistency

- All preprocessing steps are applied identically to train and test data
- TF-IDF vectorizers are fit only on training folds and applied to validation/test (no data leakage)
- Embedding models process train and test in the same batch
- Submission IDs are checked to match `Test_Submission.csv` ordering

---

## Phase 4: Feature Engineering

The final pipeline uses **23 features**, organized into 4 groups:

### Group 1: GM Base Features (16 features)

These features were computed in Stage 1 (`gm/` directory) and stored in `gm/_feats.npz`:

**4 Fine-tuned transformer OOF/test predictions:**
- `ft_scibert`: SciBERT fine-tuned with MSE regression head on title + abstract
- `ft_scibert_s2`: SciBERT with Semantic Scholar abstract enrichment
- `ft_specter2ft`: SPECTER2 fine-tuned similarly
- `ft_specter2ft_s2`: SPECTER2 with S2 abstract enrichment

**6 Frozen embedding → linear head features:**
- `specter2_title_ridge`, `specter2_title_lrev`: SPECTER2 title embeddings → Ridge / LogReg expected-value
- `bge_title_ridge`, `bge_title_lrev`: BGE-base title embeddings → Ridge / LogReg expected-value
- `specter2_titabs_ridge`, `specter2_titabs_lrev`: SPECTER2 title+abstract embeddings → Ridge / LogReg

**1 Title TF-IDF feature:**
- `title_tfidf_lrev`: TF-IDF (1,2)-grams → LogisticRegression → expected value

**2 Venue target-encoding features:**
- `te_venue`: Smoothed mean-target encoding of venue (CV-safe, smooth=10)
- `te_venue_year`: Smoothed mean-target encoding of venue × year_bin (smooth=15)

**1 Deterministic rule feature:**
- `frontmatter`: Binary flag for proceedings/conference front-matter titles (100% precision for Label 1)

**2 KNN retrieval features:**
- `knn10`, `knn25`: Cosine-similarity KNN (k=10, k=25) over SPECTER2 title embeddings → weighted mean of neighbor labels

### Group 2: Extra Fine-tuned Predictions (3 features)

Additional fine-tunes with venue-prefixed title input (`"venue: <v>. <title>"`):
- `ft2_bge_venue_title`: BGE-base fine-tuned with venue prefix
- `ft2_scibert_venue_title`: SciBERT fine-tuned with venue prefix
- `ft2_specter2_venue_title`: SPECTER2 fine-tuned with venue prefix

Two models were excluded for underperformance:
- `ft2_scibert_venue_titabs` (title+abstract dilutes signal)
- `ft2_deberta_venue_title` (general-domain model weaker than scientific encoders)

### Group 3: S2 Meta-Learner (1 feature)

- `s2_meta`: Ridge meta-learner output from a separate 10-model stacking ensemble that included:
  - Word TF-IDF Ridge, Char TF-IDF Ridge, Word+Char LogReg
  - SPECTER2/BGE Ridge heads, SPECTER2/BGE KNN retrieval
  - LightGBM with metadata + SVD features
  - Fine-tuned SciBERT and SPECTER2 (with S2 abstracts)

### Group 4: Fresh TF-IDF Ridge Features (3 features)

Built directly in the final pipeline for additional diversity:
- `tfidf_word_a3.0`: Word (1,3)-gram TF-IDF → Ridge (α=3.0)
- `tfidf_char_a3.0`: Character (3,6)-gram TF-IDF → Ridge (α=3.0)
- `tfidf_combined`: Concatenated word+char TF-IDF → Ridge (α=3.0)

### Why These Features Help

- **Transformers** (SciBERT, SPECTER2, BGE) capture semantic meaning of paper titles, understanding that "Answer Set Programming" and "Logic Programming" are related concepts
- **TF-IDF features** capture surface-level word patterns (e.g., papers with "answer set" tend to be Label 4–5)
- **Venue target-encoding** exploits the strong venue→label correlation
- **KNN retrieval** uses the principle that similar papers (by embedding distance) have similar labels
- **Venue-prefixed input** for transformers embeds the venue prior directly into the text representation
- **MSE regression** (instead of classification) respects the ordinal structure of labels
- **Frontmatter detection** handles a deterministic rule with 100% precision

---

## Phase 5: Modeling

### Multi-Seed Ridge Stacking

The 23 features are combined via Ridge regression as a meta-learner:

1. **Feature scaling**: StandardScaler fitted per fold on training features
2. **Ridge regression**: `Ridge(alpha=α)` trained to predict labels (1–5) as continuous values
3. **Multi-seed averaging**: Each configuration is run with 5 different random seeds (42, 123, 456, 789, 1234) and 5-fold StratifiedKFold per seed, then predictions are averaged across seeds for variance reduction
4. **Alpha selection**: 6 alpha values tested (0.3, 0.5, 1.0, 2.0, 5.0, 10.0); the best is selected by test-venue OOF QWK. The selected alpha was **5.0**.

### Why Ridge Stacking

- Ridge regression is fast, stable, and effective for combining heterogeneous feature views
- Multi-seed averaging reduces variance from fold assignments
- The linear combination respects the ordinal structure (higher feature values → higher predicted labels)
- Ridge's L2 regularization prevents overfitting on the relatively small dataset

### Ordinal Label Handling

Rather than classification (which ignores label ordering), the pipeline uses regression throughout:
- Base models predict continuous scores (not discrete classes)
- The meta-learner produces continuous scores
- Continuous scores are converted to labels 1–5 via threshold/calibration at the very end

---

## Phase 6: Validation

### Cross-Validation Design

- **5-fold Stratified K-Fold**: Preserves label proportions in each fold
- **Out-of-fold (OOF) predictions**: Each sample is predicted by a model trained without it
- **Test-venue evaluation**: Because `iclp` is absent from test, the OOF QWK on the test-venue subset (`cav`, `kr`, `lics`, `lpnmr`) of training data provides a more representative validation signal

### OOF QWK Results

| Model | QWK (all train) | QWK (test-venue) |
|-------|-----------------|-------------------|
| S2 Meta (baseline) | ~0.65 | ~0.65 |
| GM Finalize | ~0.67 | ~0.67 |
| **Rebuild v3 (this)** | **~0.67** | **~0.67** |

### Overfitting Control

- No hyperparameter tuning on the public leaderboard
- Threshold optimization uses coordinate ascent on OOF (never on test data)
- Multi-seed averaging reduces variance
- The calibration strategy (tv_q) uses the known test-venue distribution from training data, not the test labels

---

## Phase 7: Submission Generation

### Calibration Strategy: Test-Venue Quantile (tv_q)

The key insight is that OOF-optimal thresholds maximize QWK on the OOF but may not generalize well because the test distribution differs from the full training distribution. The selected strategy:

1. **Compute the test-venue label distribution** from training data: [0.386, 0.214, 0.178, 0.138, 0.084] for labels 1–5
2. **Rank** the continuous test predictions by percentile
3. **Map** percentile ranks to labels such that the predicted distribution matches the test-venue distribution
4. **Override** frontmatter papers → Label 1 (14 papers)

### Integer Label Conversion

The quantile calibration directly produces integer labels 1–5 (no rounding needed). Labels are clipped to ensure they stay in range [1, 5].

### Final CSV Format

```
id,Label
2494,1
2495,2
...
```
- 596 rows (298 public + 298 private)
- IDs match `Test_Submission.csv` ordering
- All labels are integers from 1 to 5

### Pre-Submission Checks

1. Row count matches (596)
2. ID order matches `Test_Submission.csv`
3. All labels in range [1, 5]
4. Label distribution is reasonable (close to test-venue training distribution)

---

## Phase 8: Final Leaderboard Result

### Selected Submission

| | Score |
|---|---|
| **Submission file** | `submission_v3_tv_q.csv` |
| **Public QWK** | **0.75557** |
| **Private QWK** | **0.73479** |

### Predicted Label Distribution

| Label | Count | Percentage | Train % |
|-------|-------|------------|---------|
| 1 | 229 | 38.4% | 36.2% |
| 2 | 128 | 21.5% | 20.6% |
| 3 | 106 | 17.8% | 17.6% |
| 4 | 82 | 13.8% | 14.7% |
| 5 | 51 | 8.6% | 10.9% |

### Public vs. Private Score Gap

The gap of **0.02078** (0.75557 → 0.73479) between public and private scores implies:
- Some degree of favorable random sampling on the public set (298 samples → QWK standard deviation ≈ 0.036)
- The model generalizes well: a gap of ~0.02 is well within statistical expectations for a 298-sample evaluation
- No evidence of public leaderboard overfitting, since the calibration strategy (tv_q) was derived from training data, not from leaderboard probing

### Comparison with Earlier Submissions

| Approach | Public QWK | Notes |
|----------|------------|-------|
| Earlier attempts (S3–S8) | ~0.69–0.72 | Various calibration and model strategies |
| `s3b_ensemble_weighted` | ~0.720 | Previous best baseline |
| GM pipeline submissions | ~0.700–0.707 | Better ranking but wrong calibration |
| **`submission_v3_tv_q`** | **0.75557** | Improved ranking + correct calibration |

The improvement from ~0.72 to ~0.756 came from two factors:
1. **Better ranking** (+0.009 in test-venue OOF QWK via additional features and stacking)
2. **Correct calibration** (test-venue quantile matching instead of OOF-optimal thresholds)
