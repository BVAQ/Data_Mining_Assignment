# Presentation Content — DM252 ASP Paper Classification
## 3–5 Minute Video Presentation

**Timing guide**: 8 slides × ~25–30 seconds each ≈ 3.5–4 minutes

---

## Slide 1 — Title and Competition Overview
**Duration: ~20 seconds**

### Slide Content
- **Title**: Ordinal Classification of Research Papers in Answer Set Programming
- **Course**: Data Mining (DM252) — Kaggle Competition
- **Student**: QUÂN BÙI VIẾT ANH
- **Selected Submission**: `submission_v3_tv_q.csv`
- **Final Score**: Public QWK = 0.75557 | Private QWK = 0.73479

### Speaker Notes
> "This presentation describes my solution for the DM252 Kaggle competition on ordinal classification of research papers. The task is to classify academic papers by their relevance to Answer Set Programming on a scale of 1 to 5. My final submission achieved a Quadratic Weighted Kappa of 0.756 on the public leaderboard and 0.735 on the private leaderboard."

### Visual
- Competition title and course info
- Leaderboard screenshot showing the final score

---

## Slide 2 — Problem Understanding
**Duration: ~30 seconds**

### Slide Content
- **Input**: Paper metadata — title, venue, year, authors, DOI
- **Target**: Ordinal label 1–5 (relevance to ASP)
- **Metric**: Quadratic Weighted Kappa (QWK)
  - Penalizes large mis-rankings quadratically
  - Accounts for chance agreement
- **Data**: 2,494 train | 298 public test | 298 private test
- **Key insight**: Label distribution is skewed (36% Label 1 → 11% Label 5)
- **Challenge**: `iclp` venue (456 rows, 18%) appears in train but NOT in test

### Speaker Notes
> "The dataset contains metadata of research papers published across five venues. The target is an ordinal relevance label from 1, meaning not related, to 5, meaning core ASP research. We use Quadratic Weighted Kappa, which is appropriate for ordinal targets because it penalizes predictions based on the squared distance from the true label. A critical observation is that the venue iclp is present in training but absent from the test set, creating a distribution shift that impacts validation strategy."

### Visual
- Table: Dataset summary (train/public/private sizes)
- Bar chart: Label distribution (1–5) with percentages
- Table: Venue mean labels showing ordinal pattern

---

## Slide 3 — Data Preprocessing
**Duration: ~25 seconds**

### Slide Content
- **Missing values**: Only `authors` has missing data (7.7% in train)
  - Filled with empty strings; authors are a weak signal
- **Text cleaning**: Minimal — relied on TF-IDF tokenizers and transformer tokenizers
- **Venue-prefixed input**: `"venue: <v>. <title>"` — embeds venue prior into text
- **Frontmatter detection**: Regex patterns for proceedings, preface, editorial titles
  - 68 in train → ALL Label 1 (100% precision)
  - 14 detected in test → deterministic override to Label 1
- **Train/test consistency**: TF-IDF fitted per fold (no leakage), same preprocessing for all splits

### Speaker Notes
> "Data preprocessing was straightforward. Only the authors column had missing values, which were filled with empty strings since author identity proved to be a weak signal. A critical preprocessing step was prepending the venue name to the paper title before feeding it to transformers, giving the model direct access to the venue prior. We also identified frontmatter titles such as conference proceedings and prefaces, which are 100% consistently Label 1 in the training data. These were deterministically overridden in predictions."

### Visual
- Table: Missing value summary
- Example of venue-prefixed text transformation
- Example frontmatter titles detected

---

## Slide 4 — Feature Engineering
**Duration: ~35 seconds**

### Slide Content
- **23 total features** from 4 groups:
  1. **Pre-trained embeddings** (6 features): SPECTER2, BGE → Ridge/LogReg heads
  2. **Fine-tuned transformers** (7 features): SciBERT, SPECTER2, BGE with venue prefix + MSE regression head
  3. **TF-IDF features** (4 features): Word/char n-grams → Ridge/LogReg
  4. **Metadata features** (6 features): Venue target-encoding, KNN retrieval, frontmatter flag
- **Key design choices**:
  - MSE regression heads (not classification) — respects ordinal structure
  - Venue-prefixed input — gives transformers the strong venue prior
  - Title-only > title+abstract — abstracts dilute the ASP relevance signal

### Speaker Notes
> "Our feature engineering produced 23 features from four complementary groups. Pre-trained scientific embeddings from SPECTER2 and BGE were used with Ridge and LogReg heads. Fine-tuned transformers used a regression objective with MSE loss, which respects the ordinal nature of labels better than classification. A key finding was that venue-prefixed input — prepending the venue name to the title — significantly improved transformer performance. We also found that using abstracts actually diluted the signal, so title-only models performed better. Metadata features like venue target-encoding and KNN retrieval over embedding space provided additional complementary signals."

### Visual
- Feature group diagram (4 boxes with feature counts)
- Table: Individual feature QWK scores (top 5–6 features)

---

## Slide 5 — Model Development and Training
**Duration: ~30 seconds**

### Slide Content
- **Architecture**: Multi-stage stacking ensemble
  - Stage 1: 16 base features (GM pipeline)
  - Stage 2: 10-model Ridge meta-learner → 1 feature
  - Stage 3: 23 features → Multi-seed Ridge stacking
- **Multi-seed averaging**: 5 seeds × 5-fold StratifiedKFold → averaged for variance reduction
- **Ridge stacking**: `Ridge(alpha=5.0)` with StandardScaler
- **Calibration**: Test-venue quantile mapping to match expected label distribution
- **Post-processing**: Frontmatter override → Label 1

### Speaker Notes
> "The final model is a multi-stage stacking ensemble. In Stage 1, we compute 16 base features from transformers, embeddings, and metadata. Stage 2 runs a separate 10-model stack to produce a meta-learner score. In Stage 3, all 23 features are combined via Ridge regression with multi-seed averaging — each configuration runs with 5 different random seeds, and predictions are averaged to reduce variance. The continuous predictions are then calibrated using test-venue quantile mapping, which forces the predicted distribution to match the expected test distribution. Finally, detected frontmatter papers are overridden to Label 1."

### Visual
- Pipeline diagram: Raw Data → Features → Stacking → Calibration → Submission
- Table: Stacking configuration (seeds, alpha, folds)

---

## Slide 6 — Validation Strategy
**Duration: ~25 seconds**

### Slide Content
- **5-fold Stratified K-Fold** with seed 42
- **Out-of-fold (OOF) predictions** for all base models and the meta-learner
- **Test-venue evaluation**: QWK computed on the test-venue subset of train (2,038 / 2,494 rows)
  - Excludes `iclp` which is absent from test
  - More representative of actual test performance
- **Coordinate-ascent threshold optimization**: Iteratively optimizes 4 thresholds to maximize QWK
- **Key validation finding**: OOF QWK is reliable for ranking models, but NOT for calibration selection
  - The best OOF-calibrated model scored 0.707 on public LB
  - The "worst" OOF-calibrated model (tv_q) scored 0.756 on public LB

### Speaker Notes
> "Validation used 5-fold Stratified K-Fold. A critical design choice was evaluating on the test-venue subset of training data, which excludes iclp papers and better represents the test distribution. We found that OOF QWK is reliable for comparing model rankings — our improved features consistently outperformed baselines — but is anti-correlated with leaderboard performance for calibration strategy selection. This led to using the test-venue quantile calibration, which has lower OOF QWK but better leaderboard generalization."

### Visual
- Table: OOF QWK for different calibration strategies
- Annotation showing the anti-correlation between OOF QWK and LB QWK

---

## Slide 7 — Experimental Results and Discussion
**Duration: ~30 seconds**

### Slide Content

| Submission | Approach | Public QWK | Private QWK |
|---|---|---|---|
| Earlier baselines | Various strategies | ~0.69–0.72 | — |
| `s3b_ensemble` | Previous best | 0.720 | — |
| GM pipeline | Better ranking, wrong calibration | 0.700–0.707 | — |
| **`v3_tv_q`** | **Improved ranking + correct calibration** | **0.75557** | **0.73479** |

- **Improvement sources**:
  1. Better feature ranking: +0.009 OOF QWK from additional features and stacking
  2. Correct calibration: Test-venue quantile matching vs. OOF-optimal thresholds
- **Public-private gap**: 0.021 — within statistical expectation for 298-sample QWK (σ ≈ 0.036)
- **No leaderboard overfitting**: Calibration derived from training data, not LB probing

### Speaker Notes
> "Our experimental journey shows a clear progression. Earlier submissions scored around 0.69 to 0.72 using various strategies. The GM pipeline improved the ranking quality but initially scored only 0.70 due to incorrect calibration. The breakthrough came from separating the ranking and calibration decisions: using our improved stacking for ranking and test-venue quantile matching for calibration. This produced our final score of 0.756 public and 0.735 private. The gap of 0.021 between public and private is within the expected statistical variation for 298 samples, suggesting good generalization."

### Visual
- Model comparison table
- Leaderboard screenshot

---

## Slide 8 — Conclusion
**Duration: ~25 seconds**

### Slide Content
- **Key achievements**:
  - Public QWK 0.756 | Private QWK 0.735
  - Improved from ~0.72 baseline through principled feature engineering and calibration
- **Key lessons**:
  1. Venue-prefixed transformer input captures domain priors effectively
  2. MSE regression outperforms classification for ordinal targets
  3. Calibration strategy matters as much as model quality
  4. OOF validates ranking, but test distribution determines calibration
- **Limitations**:
  - Middle classes (2/3/4) remain genuinely confusable from title+venue alone
  - Per-class recall: 0.40/0.27/0.38 for Labels 2/3/4
- **Future improvements**:
  - Full-text or abstract analysis with better models
  - Citation graph features
  - Ordinal-specific loss functions (e.g., CORN, ordinal cross-entropy)

### Speaker Notes
> "In conclusion, our solution achieved a Quadratic Weighted Kappa of 0.756 on the public leaderboard and 0.735 on the private leaderboard. The key lessons from this project are: first, embedding domain knowledge directly into transformer input through venue prefixing is highly effective. Second, regression objectives are better suited than classification for ordinal targets. Third, the calibration strategy — how we convert continuous predictions to discrete labels — is as important as the model quality itself. A limitation is that the middle classes remain genuinely difficult to distinguish from title and venue alone. Future work could explore citation graph features and ordinal-specific loss functions. Thank you."

### Visual
- Summary table of final scores
- Bullet points for lessons and future work
