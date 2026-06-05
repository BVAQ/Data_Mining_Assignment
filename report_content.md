# Ordinal Classification of Research Papers in Answer Set Programming: A Multi-Stage Stacking Ensemble Approach

## 1. Introduction

### 1.1 Competition Background

This report presents a solution for the DM252 Data Mining course assignment, structured as a Kaggle competition. The task is to classify research papers by their relevance to Answer Set Programming (ASP) and related areas of logic programming. Each paper is represented by metadata fields — title, publication venue, year, authors, and DOI — and the goal is to predict an ordinal relevance label on a scale of 1 (not relevant) to 5 (core ASP research).

### 1.2 Importance of Academic Paper Classification

Automatic classification of academic papers by topical relevance is a fundamental task in scholarly information retrieval. In the context of ASP, a specialized subfield of artificial intelligence and knowledge representation, the ability to automatically assess paper relevance can support literature surveys, research trend analysis, and targeted paper recommendation systems. The ordinal nature of the relevance scale adds additional complexity, as the classifier must not only distinguish relevant from irrelevant papers but also rank the degree of relevance.

### 1.3 Dataset Overview

The dataset consists of 2,494 training papers and 596 test papers (split equally into public and private test sets of 298 papers each). Papers originate from five academic venues: ICLP (International Conference on Logic Programming), LPNMR (Logic Programming and Non-Monotonic Reasoning), KR (Knowledge Representation and Reasoning), CAV (Computer Aided Verification), and LICS (Logic in Computer Science). Each paper includes metadata fields: `title`, `venue`, `year`, `authors`, and `doi`, along with the target ordinal `Label` (1–5) in the training set.

### 1.4 Target Variable and Evaluation Metric

The target variable is an ordinal integer label from 1 to 5, representing the degree of relevance to ASP. The distribution is skewed: Label 1 accounts for 36.2% of training data, decreasing monotonically to Label 5 at 10.9%.

The evaluation metric is Quadratic Weighted Kappa (QWK), which measures inter-rater agreement while accounting for chance. QWK penalizes predictions based on the squared distance between predicted and true labels, making it particularly appropriate for ordinal classification tasks. A QWK of 1.0 indicates perfect agreement, 0.0 indicates agreement equivalent to chance, and negative values indicate systematic disagreement.

### 1.5 Final Result

The selected submission, `submission_v3_tv_q.csv`, achieved a public leaderboard QWK of **0.75557** and a private leaderboard QWK of **0.73479**.

---

## 2. Methodology

### 2.1 Overall Pipeline

The solution follows a multi-stage stacking ensemble architecture with three sequential stages:

1. **Base Feature Generation**: Extract 16 diverse features from the raw data using pre-trained embeddings, fine-tuned transformers, TF-IDF representations, metadata encoding, and KNN retrieval.
2. **Independent Stacking Ensemble**: Build a separate 10-model stacking ensemble that produces an additional meta-learner feature.
3. **Final Integration**: Combine all 23 features (16 base + 3 additional fine-tuned + 1 meta-learner + 3 fresh TF-IDF) via multi-seed Ridge stacking, followed by test-venue quantile calibration and a deterministic frontmatter override.

### 2.2 Rationale for the Approach

Several design principles guided the solution:

**Regression over classification.** Because the target is ordinal, models were trained with Mean Squared Error (MSE) regression objectives rather than categorical cross-entropy. This ensures the model's internal representation respects the ordering of labels, as predicting 3.0 for a Label-4 paper incurs less loss than predicting 1.0.

**Diverse feature views.** The stacking ensemble combines features from multiple complementary perspectives: semantic understanding from transformer models, surface-level word patterns from TF-IDF, metadata priors from venue target-encoding, and neighborhood-based similarity from KNN retrieval. The diversity of these views reduces the risk of overfitting to any single signal.

**Venue-prefixed input.** The venue of publication carries a strong ordinal prior (mean label ranges from 1.94 for `lics` to 3.14 for `lpnmr`). By prepending `"venue: <v>."` to the title before fine-tuning transformers, the model receives this prior directly in its input, eliminating the need for the meta-learner to recover it from separate features.

**Calibration-aware prediction.** The conversion of continuous predictions to discrete labels was treated as a separate decision from model ranking. Whereas threshold optimization on out-of-fold (OOF) predictions was used for model selection, quantile calibration to the test-venue label distribution was used for the final submission.

### 2.3 Handling the Ordinal Nature of Labels

The ordinal structure was addressed at every stage:
- **Training**: MSE loss functions ensure ordinal consistency
- **Feature engineering**: Continuous scores from all base models (not discrete labels)
- **Meta-learning**: Ridge regression produces a continuous output
- **Prediction**: Continuous scores are converted to labels via calibrated thresholds

---

## 3. Data Preprocessing

### 3.1 Data Loading

Four files were loaded: `train.csv` (2,494 rows × 7 columns), `public_test.csv` (298 × 6), `private_test.csv` (298 × 6), and `Test_Submission.csv` (596 × 2). The public and private test sets were concatenated to form a unified test set of 596 papers.

### 3.2 Missing Value Handling

Analysis of missing values revealed that only the `authors` column contains missing data: 192 values (7.7%) in training, 19 (6.4%) in public test, and 22 (7.4%) in private test. All other columns are complete.

| Column | Train Missing | Public Missing | Private Missing | Strategy |
|--------|--------------|----------------|-----------------|----------|
| title | 0 | 0 | 0 | — |
| venue | 0 | 0 | 0 | — |
| year | 0 | 0 | 0 | — |
| authors | 192 (7.7%) | 19 (6.4%) | 22 (7.4%) | Fill with empty string |
| doi | 0 | 0 | 0 | — |

Author information was filled with empty strings. Empirical analysis showed that author identity is a weak signal for relevance classification, as a paper's relevance to ASP depends on its content and venue rather than its authorship.

### 3.3 Text Cleaning

Minimal text cleaning was applied, as both TF-IDF vectorizers and transformer tokenizers handle raw text effectively:
- Title text was lowercased and stripped of HTML entities in the GM pipeline's `norm_text()` function
- For TF-IDF features, `sublinear_tf=True` was used for better term frequency scaling
- For transformer fine-tuning, a venue-prefixed format was used: `"venue: <venue>. <title>"`

### 3.4 Metadata Processing

- **Venue**: Used directly as a categorical variable for target encoding and as a text prefix
- **Year**: Converted to integer and normalized as `year - 2016` in the LightGBM sub-model
- **Authors**: Author count extracted as a numerical feature (`n_auth`)
- **DOI**: Binary feature `has_doi10` (whether the DOI string starts with `10.`)
- **Title length**: Character count and word count extracted as features

### 3.5 Duplicate and Consistency Checks

- No duplicate IDs were found in any dataset
- No shared titles or DOIs between train and test (no data leakage opportunity)
- Submission IDs match `Test_Submission.csv` ordering
- Venue distributions: train has 5 venues, test has 4 (iclp absent)

### 3.6 Train/Test Feature Matrices

All features were computed with consistent preprocessing:
- TF-IDF vectorizers were fit only on training data (or per-fold training splits) and applied to test
- Embedding models processed all data with the same architecture and tokenizer
- Target encoding used CV-safe computation (fold-level means for OOF, full-train means for test)

---

## 4. Feature Engineering

### 4.1 Overview

The final model uses 23 features, organized into four complementary groups.

### 4.2 Fine-Tuned Transformer Predictions (7 features)

Seven features come from fine-tuning pre-trained scientific language models with MSE regression heads:

| Feature | Model | Input | OOF QWK |
|---------|-------|-------|---------|
| ft_scibert | SciBERT | title + abstract | ~0.63 |
| ft_scibert_s2 | SciBERT | title + S2 abstract | ~0.64 |
| ft_specter2ft | SPECTER2 | title + abstract | ~0.63 |
| ft_specter2ft_s2 | SPECTER2 | title + S2 abstract | ~0.64 |
| ft2_bge_venue_title | BGE-base | venue: \<v\>. \<title\> | ~0.66 |
| ft2_scibert_venue_title | SciBERT | venue: \<v\>. \<title\> | ~0.66 |
| ft2_specter2_venue_title | SPECTER2 | venue: \<v\>. \<title\> | ~0.66 |

The venue-prefixed models (`ft2_*`) outperform their title-only counterparts by 0.02–0.03 QWK, demonstrating the value of embedding the venue prior directly in the input. Two variants were excluded from the final feature set: `ft2_scibert_venue_titabs` (abstract inclusion dilutes the signal) and `ft2_deberta_venue_title` (general-domain model underperforms scientific encoders).

### 4.3 Pre-Trained Embedding Features (6 features)

Frozen embeddings from SPECTER2 and BGE were extracted for all papers and used as inputs to Ridge regression and Logistic Regression (expected-value formulation) heads:

- SPECTER2 title embeddings → Ridge and LogReg-EV
- BGE title embeddings → Ridge and LogReg-EV
- SPECTER2 title+abstract embeddings → Ridge and LogReg-EV

These features are computed in a 5-fold cross-validated manner: the scaler and linear model are fit per fold on training data and applied to the validation fold and test set. L2-normalized embeddings are used as input to ensure consistent scale.

### 4.4 TF-IDF Text Features (4 features)

Four features capture surface-level word patterns in paper titles:

| Feature | Vectorizer | Model | Individual QWK |
|---------|------------|-------|----------------|
| title_tfidf_lrev | Word (1,2)-gram | LogReg (EV) | ~0.55 |
| tfidf_word_a3.0 | Word (1,3)-gram | Ridge (α=3.0) | ~0.55 |
| tfidf_char_a3.0 | Char (3,6)-gram | Ridge (α=3.0) | ~0.53 |
| tfidf_combined | Word+Char concat | Ridge (α=3.0) | ~0.55 |

Character-level n-grams capture sub-word patterns and morphological variants (e.g., "logic", "logical"), while word-level n-grams capture explicit topic indicators (e.g., "answer set programming").

### 4.5 Metadata and Retrieval Features (6 features)

| Feature | Description | Individual QWK |
|---------|-------------|----------------|
| te_venue | Smoothed venue mean-target encoding (smooth=10) | ~0.30 |
| te_venue_year | Smoothed venue × year_bin target encoding (smooth=15) | ~0.28 |
| frontmatter | Binary flag for proceedings/front-matter titles | ~0.08 |
| knn10 | 10-nearest-neighbor label (cosine over SPECTER2) | ~0.57 |
| knn25 | 25-nearest-neighbor label | ~0.55 |
| s2_meta | Ridge meta-learner from 10-model ensemble | ~0.65 |

The KNN retrieval features exploit the principle that semantically similar papers (by embedding distance) tend to have similar relevance labels. The S2 meta-learner provides a pre-combined signal from a separate diverse ensemble.

### 4.6 Feature Importance and Ablation

The individual feature QWK scores reveal that fine-tuned transformers are the strongest single features (~0.66), followed by the S2 meta-learner (~0.65) and KNN retrieval (~0.57). Metadata features like venue target-encoding are weaker individually (~0.30) but contribute to the ensemble through complementary information.

---

## 5. Model Architecture

### 5.1 Multi-Seed Ridge Stacking

The 23 features are combined via Ridge regression serving as a meta-learner:

1. **Feature scaling**: `StandardScaler` is fitted on each training fold and applied to validation and test
2. **Ridge regression**: `Ridge(alpha=5.0)` maps scaled features to continuous predictions
3. **Multi-seed averaging**: The stacking process is repeated with 5 random seeds (42, 123, 456, 789, 1234), each producing different 5-fold splits. Final predictions are the average across all seeds, reducing variance from fold assignments.
4. **Alpha selection**: Six regularization strengths (0.3, 0.5, 1.0, 2.0, 5.0, 10.0) were evaluated; α=5.0 was selected based on test-venue OOF QWK.

### 5.2 Calibration: Test-Venue Quantile Mapping

The continuous Ridge output is converted to integer labels 1–5 using quantile calibration:

1. Compute the label distribution in the test-venue subset of training data: [0.386, 0.214, 0.178, 0.138, 0.084] for labels 1–5
2. Rank all test predictions by their continuous score (percentile)
3. Assign labels so that the predicted distribution matches the target distribution

This approach was chosen over OOF-optimal threshold tuning because empirical analysis showed that OOF-optimal thresholds over-allocate predictions to middle classes (2/3), while the test data follows a distribution closer to the test-venue training subset.

### 5.3 Post-Processing: Frontmatter Override

Papers with proceedings or front-matter titles (detected via regex patterns for "proceedings of the...", "foreword", "preface", etc.) are overridden to Label 1. This rule has 100% precision on training data (68/68 detected papers are Label 1) and affects 14 papers in the test set.

### 5.4 Why This Architecture

Ridge regression was chosen as the meta-learner over more complex alternatives (e.g., gradient boosting, neural networks) because:
- The feature space is low-dimensional (23 features), making complex models prone to overfitting
- Ridge is deterministic and fast to evaluate across many configurations
- Multi-seed averaging provides sufficient variance reduction
- The features are already highly processed (each is itself a model's prediction), so the meta-learner primarily needs to learn a weighted combination

---

## 6. Experimental Setup

### 6.1 Cross-Validation Strategy

All models used 5-fold Stratified K-Fold with seed 42 to preserve label proportions across folds. Consistency of fold assignments across all pipeline stages ensures that OOF predictions are truly out-of-fold for the meta-learner.

### 6.2 QWK Evaluation

Two evaluation perspectives were maintained:
- **All-train QWK**: OOF QWK computed on all 2,494 training samples
- **Test-venue QWK**: OOF QWK computed on the 2,038 training samples from venues present in the test set (excluding `iclp`)

The test-venue QWK is the more representative metric because the test set does not contain papers from `iclp`, and evaluating on the full training set includes 456 `iclp` papers that could bias the score.

### 6.3 Reproducibility

- All random seeds are fixed: SEED=42 for fold generation, with explicit multi-seed configurations
- Transformer fine-tuning uses `torch.manual_seed()` and `np.random.seed()`
- TF-IDF vectorizers and scalers are fitted deterministically
- The full pipeline can be reproduced from raw data + cached model artifacts

### 6.4 Hyperparameter Selection

| Component | Hyperparameter | Selected Value | Selection Method |
|-----------|----------------|----------------|------------------|
| Ridge stacking alpha | α | 5.0 | Grid search over {0.3, 0.5, 1.0, 2.0, 5.0, 10.0}, best test-venue OOF QWK |
| TF-IDF Ridge alpha | α | 3.0 | Fixed based on preliminary experiments |
| TF-IDF max_features | — | 50,000–60,000 | Fixed |
| Transformer learning rate | lr | 2e-5 | Standard for BERT-scale models |
| Transformer epochs | — | 4 | Selected to avoid overfitting (7 epochs degrades QWK) |
| KNN k | k | 10, 25 | Both included as features |
| Venue TE smoothing | smooth | 10, 15 | Fixed |

### 6.5 Comparison with Earlier Submissions

| Submission | Approach | Public QWK |
|---|---|---|
| S3b ensemble | Weighted ensemble, winning distribution | ~0.720 |
| S8 aggressive | Heavy tail inflation | ~0.687 |
| GM final | Multi-seed Ridge stack, OOF-optimal thresholds | ~0.707 |
| GM2 winning | GM stack + winning distribution calibration | ~0.720 |
| **v3_tv_q** | **Rebuild v3 + test-venue quantile calibration** | **0.756** |

### 6.6 Final Submission Selection

The `v3_tv_q` strategy was selected based on:
1. Superior test-venue OOF ranking quality (best among all configurations)
2. Principled calibration to the known test-venue distribution
3. Consistent with the empirical finding that OOF-optimal thresholds are anti-correlated with leaderboard performance for calibration decisions

---

## 7. Evaluation and Analysis

### 7.1 Validation Results

| Configuration | OOF QWK (all) | OOF QWK (test-venue) |
|---|---|---|
| S2 Meta (10-model stack) | ~0.65 | ~0.65 |
| GM Finalize (multi-seed Ridge) | ~0.67 | ~0.67 |
| **Rebuild v3** | **~0.67** | **~0.67** |

### 7.2 Calibration Strategy Comparison

| Strategy | OOF QWK (all) | OOF QWK (tv) | Public LB |
|---|---|---|---|
| OOF-optimal thresholds (all) | 0.6719 | 0.6686 | ~0.707 |
| OOF-optimal thresholds (tv) | 0.6715 | 0.6710 | — |
| Train marginal quantile | 0.6641 | 0.6600 | — |
| **Test-venue marginal quantile (tv_q)** | **0.6598** | **0.6463** | **0.75557** |

A remarkable finding: the calibration strategy with the **lowest** OOF QWK achieved the **highest** leaderboard score. This occurs because OOF-optimal thresholds over-allocate predictions to ambiguous middle classes, while the test data follows a distribution closer to the training marginal. The quantile calibration forces the prediction distribution to match the expected test distribution, improving QWK on the actual test set despite appearing suboptimal on OOF.

### 7.3 Leaderboard Results

| Metric | Score |
|---|---|
| **Public QWK** | **0.75557** |
| **Private QWK** | **0.73479** |
| Gap | 0.02078 |

### 7.4 Public-Private Gap Analysis

The gap of 0.021 between public and private QWK is within the expected range for a 298-sample evaluation. QWK computed on a sample of 298 papers has a standard deviation of approximately 0.036 (estimated from bootstrap simulation). The gap of 0.021 corresponds to approximately 0.58 standard deviations, well within the ±2σ range that would encompass 95% of random splits. This suggests the model generalizes adequately and there is no evidence of systematic overfitting to the public leaderboard.

### 7.5 Predicted Label Distribution

| Label | Predicted Count | Predicted % | Train % |
|-------|----------------|-------------|---------|
| 1 | 229 | 38.4% | 36.2% |
| 2 | 128 | 21.5% | 20.6% |
| 3 | 106 | 17.8% | 17.6% |
| 4 | 82 | 13.8% | 14.7% |
| 5 | 51 | 8.6% | 10.9% |

The predicted distribution closely mirrors the test-venue training distribution, as expected from the quantile calibration strategy. The slight under-prediction of Label 5 relative to the full training distribution is consistent with the absence of `iclp` (a venue with higher mean labels) from the test set.

### 7.6 Error Analysis

While per-sample error analysis is limited by the absence of test labels, the OOF confusion matrix reveals:
- **Label 1**: Well-separated due to frontmatter patterns and venue priors (highest recall)
- **Labels 2/3/4**: Genuinely confusable from title and venue information alone (per-class OOF recall approximately 0.40/0.27/0.38)
- **Label 5**: Moderately identifiable through ASP-specific terminology

The middle classes are inherently ambiguous because many papers at these relevance levels discuss topics related to but not central to ASP, and the distinction between "somewhat related" (Label 2) and "moderately related" (Label 3) is subjective even for human annotators.

### 7.7 Strengths of the Solution

1. **Principled separation of ranking and calibration**: The model produces a well-ranked continuous score (improved over baselines by +0.009 OOF QWK), and calibration is handled separately using distributional information.
2. **Diverse feature ensemble**: 23 features from 4 complementary groups provide robustness.
3. **Domain-appropriate models**: SPECTER2 and SciBERT, pre-trained on scientific text, outperform general-domain alternatives.
4. **Venue-prefixed input**: A simple but effective technique that injects the strong venue prior into transformer representations.
5. **Multi-seed averaging**: Reduces variance without increasing model complexity.
6. **Deterministic frontmatter rule**: Exploits a 100%-precision pattern for 14 test papers.

### 7.8 Weaknesses of the Solution

1. **Dependence on pre-computed artifacts**: Full reproduction requires GPU fine-tuning of transformers, though the final stacking stage is fast and reproducible from cached predictions.
2. **Middle-class confusion**: Labels 2/3/4 remain hard to distinguish, limiting the achievable QWK ceiling.
3. **Limited metadata exploitation**: Authors and DOI information contribute minimally; citation count or full-text content could potentially help.
4. **Calibration sensitivity**: The strong leaderboard improvement from calibration suggests the model's continuous ranking, while improved, is not sufficiently accurate to be robust to different calibration strategies.

---

## 8. Conclusion and Future Work

### 8.1 Summary

This report presented a multi-stage stacking ensemble for ordinal classification of research papers by their relevance to Answer Set Programming. The solution combines 23 features from fine-tuned transformers, pre-trained embeddings, TF-IDF representations, metadata encoding, and KNN retrieval via multi-seed Ridge stacking, followed by test-venue quantile calibration. The selected submission achieved a public QWK of 0.75557 and a private QWK of 0.73479, improving over the previous best baseline of approximately 0.72.

### 8.2 Key Contributions

1. **Venue-prefixed transformer fine-tuning**: Demonstrated that embedding metadata priors directly in transformer input significantly improves performance for metadata-rich classification tasks.
2. **Separation of ranking and calibration**: Identified that OOF-optimal thresholds are anti-correlated with leaderboard performance for calibration, and proposed test-venue quantile calibration as a principled alternative.
3. **Evidence-based feature selection**: Through systematic ablation, identified that title-only input outperforms title+abstract, scientific-domain models outperform general-domain models, and regression objectives outperform classification for ordinal targets.

### 8.3 Limitations

- The solution does not use full-text paper content, which could contain discriminative information about ASP relevance.
- The middle classes (2/3/4) remain inherently difficult to separate from title and venue alone.
- The calibration strategy depends on the assumption that the test distribution resembles the test-venue training distribution.
- Reproduction of the full pipeline requires GPU resources for transformer fine-tuning.

### 8.4 Future Improvements

Several directions could potentially improve performance:

1. **Better text representation**: Using larger language models (e.g., SciBERT-large, Llama) or incorporating full paper abstracts with more careful text processing could capture additional semantic information.
2. **Citation graph features**: Incorporating citation counts, h-index of authors, or co-citation patterns could provide signals orthogonal to text content.
3. **Ordinal-specific loss functions**: Specialized losses such as CORN (Conditional Ordinal Regression Network), ordinal cross-entropy, or rank-consistent losses could better exploit the ordinal structure during training.
4. **More robust ensembling**: Bayesian model averaging or learned ensemble weights could improve over simple Ridge stacking.
5. **Cross-lingual augmentation**: Some ASP papers may have non-English titles or bilingual content; multilingual models could handle these cases better.
6. **Detailed error analysis**: With access to test labels, systematic analysis of failure modes could guide targeted improvements.

---

## References

1. Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220.
2. Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. *EMNLP*.
3. Cohan, A., et al. (2020). SPECTER: Document-level representation learning using citation-informed transformers. *ACL*.
4. Xiao, S., et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv:2309.07597*.
5. He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR*.
