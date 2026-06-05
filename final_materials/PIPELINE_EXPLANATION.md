# Step-by-Step Pipeline Explanation

## 1. Scope and Reproducibility

The selected final Kaggle submission is `outputs/submission_v3_tv_q.csv`. It achieved a public QWK of `0.75557` and a private QWK of `0.73479`.

The final submission has been reproduced with `reproduce_submission_v3_tv_q.py`. The script writes `outputs/reproduced_submission_v3_tv_q.csv` and verifies exact agreement with the selected file:

- ID order match: true
- Label match: true
- Label agreement: `1.0000`

Important limitation: the selected v3 file was created from cached out-of-fold and test predictions produced during experimentation. The retained source scripts show how those cached predictions were created, but not every expensive base learner is retrained inside one script. The consolidated reproduction script therefore reproduces the final selected CSV from raw CSV files plus saved artifacts. The relevant cached artifact is `outputs/rebuild_v3_oof.npz`.

## 2. Raw Data Understanding

The task is an ordinal classification problem for academic papers related to Answer Set Programming and nearby research areas. Each training row represents one paper and contains metadata:

- `id`: paper identifier.
- `title`: paper title.
- `venue`: publication venue.
- `year`: publication year.
- `authors`: author list.
- `doi`: DOI or link.
- `Label`: ordered target label from 1 to 5.

The test data are split into `public_test.csv` and `private_test.csv`, each with 298 rows. The required final output follows the ID order in `Test_Submission.csv`.

Because labels are ordered, predicting label 5 instead of label 4 is less severe than predicting label 5 instead of label 1. Quadratic Weighted Kappa is appropriate because it measures agreement between predicted and true ordinal labels while penalizing larger ordinal errors more heavily.

## 3. Data Loading and Inspection

The reproduction script loads:

- `data/raw/train.csv`
- `data/raw/public_test.csv`
- `data/raw/private_test.csv`
- `data/raw/Test_Submission.csv`

The raw data checks found:

- Train size: 2,494 rows.
- Public test size: 298 rows.
- Private test size: 298 rows.
- Final submission size: 596 rows.
- No duplicated IDs in train, public test, private test, or final submission template.
- Missing values appear only in `authors`: 192 train rows, 19 public test rows, and 22 private test rows.
- The concatenated public + private test order matches `Test_Submission.csv`.
- The final submission IDs match `Test_Submission.csv`.
- All predicted labels are integers from 1 to 5.

The train labels are imbalanced:

- Label 1: 903 rows, 36.21%.
- Label 2: 514 rows, 20.61%.
- Label 3: 438 rows, 17.56%.
- Label 4: 367 rows, 14.72%.
- Label 5: 272 rows, 10.91%.

The train venues are `cav`, `iclp`, `kr`, `lics`, and `lpnmr`. The test venues are `cav`, `kr`, `lics`, and `lpnmr`; `iclp` appears in training but not in the test files. This venue shift motivated the test-venue calibration step.

## 4. Preprocessing

The final v3 stack uses conservative preprocessing:

- Missing titles are replaced with an empty string.
- Missing venues are replaced with `unknown`.
- Missing authors and DOI/link fields are replaced with empty strings when those fields are inspected or used by cached feature stages.
- Year values are converted to numeric form where required by metadata features.
- Public and private test rows are concatenated and checked against `Test_Submission.csv` so that predictions keep the required order.

No direct train-test title leakage was used. The consistency check found zero shared normalized titles between train and test. Three DOI/link strings appeared in both train and test; these were inspected and not used as deterministic leakage.

## 5. Feature Engineering

The final v3 stack uses 23 continuous or binary feature columns. Most are out-of-fold predictions from base models, so each feature represents a model's estimated ordinal score for a paper.

Feature groups:

1. Fine-tuned transformer regression predictions:
   - `ft_scibert`
   - `ft_scibert_s2`
   - `ft_specter2ft`
   - `ft_specter2ft_s2`
   - `ft2_bge_venue_title`
   - `ft2_scibert_venue_title`
   - `ft2_specter2_venue_title`

2. Frozen embedding heads:
   - SPECTER2 title Ridge and logistic expected-value heads.
   - BGE title Ridge and logistic expected-value heads.
   - SPECTER2 title+abstract Ridge and logistic expected-value heads.

3. Text TF-IDF features:
   - `title_tfidf_lrev`
   - `tfidf_word_a3.0`
   - `tfidf_char_a3.0`
   - `tfidf_combined`

4. Metadata and retrieval features:
   - `te_venue`
   - `te_venue_year`
   - `frontmatter`
   - `knn10`
   - `knn25`

5. Earlier stacked ensemble feature:
   - `s2_meta`

The title is the most important raw text field because it carries topical signals such as "answer set", "logic programming", "reasoning", "verification", and related terms. Venue features are useful because the train data show different mean labels by venue. Retrieval features estimate label similarity from nearest neighbors in embedding space. The frontmatter flag captures proceedings, prefaces, and similar non-research-paper entries, which were strongly associated with label 1 in the training data.

## 6. Modeling

The selected v3 pipeline is a stacked regression ensemble:

1. Base models produce continuous predictions on a 1-5 scale.
2. A Ridge regression meta-learner combines the 23 feature columns.
3. The meta-learner is trained with 5-fold StratifiedKFold.
4. Multiple random fold seeds are used: 42, 123, 456, 789, and 1234.
5. Candidate Ridge `alpha` values are evaluated: 0.3, 0.5, 1.0, 2.0, 5.0, and 10.0.
6. The saved final v3 artifact selected `alpha=5.0`.

The model predicts a continuous ordinal score. The continuous score is then converted to integer labels using calibration.

## 7. Validation

Validation uses Quadratic Weighted Kappa on out-of-fold predictions. The stack was evaluated both on all training rows and on the subset of training rows whose venues appear in the test set. This subset excludes `iclp`, matching the observed test-venue composition more closely.

From the saved final v3 artifact:

- Ridge stack alpha: `5.0`.
- Threshold-tuned OOF QWK on all train rows: `0.6719`.
- Threshold-tuned OOF QWK on the test-venue subset: `0.6708`.
- Test-venue training subset size: 2,038 rows out of 2,494.

These validation scores evaluate the quality of the continuous ranking. The final Kaggle file additionally uses quantile calibration, so the leaderboard result reflects both ranking quality and label distribution calibration.

## 8. Calibration and Post-processing

The selected file is the `tv_q` calibration variant. It maps continuous test scores to integer labels so that the predicted label distribution approximately follows the training distribution restricted to venues present in the test set.

The TV marginal calibration target is:

- Label 1: 0.3857
- Label 2: 0.2139
- Label 3: 0.1781
- Label 4: 0.1379
- Label 5: 0.0844

After calibration, a deterministic post-processing rule sets frontmatter/proceedings-style titles to label 1. There are 14 such rows in the test set.

The final predicted distribution in `submission_v3_tv_q.csv` is:

- Label 1: 229 rows, 38.42%.
- Label 2: 128 rows, 21.48%.
- Label 3: 106 rows, 17.79%.
- Label 4: 82 rows, 13.76%.
- Label 5: 51 rows, 8.56%.

## 9. Submission Generation

The final CSV is created with two columns:

- `id`
- `Label`

The script verifies:

- Exactly 596 rows.
- ID order equals `Test_Submission.csv`.
- Labels are integers from 1 to 5.
- The reproduced file exactly matches `outputs/submission_v3_tv_q.csv`.

## 10. Final Leaderboard Result

The selected final submission is:

- File: `submission_v3_tv_q.csv`
- Public QWK: `0.75557`
- Private QWK: `0.73479`

The public score is higher than the private score by `0.02078`. This gap suggests that the public split was slightly more favorable to the final ranking/calibration than the private split, or that the public and private samples differ modestly in distribution. The private score remains the official grading result according to the assignment PDF.
