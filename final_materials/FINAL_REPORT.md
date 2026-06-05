# Final Report: Ordinal Classification of ASP-Related Research Papers

## 1. Introduction

This project was completed for the Data Mining 252 Stage 2 Kaggle competition. The task is to classify academic papers related to Answer Set Programming and neighboring research areas into ordered relevance labels from 1 to 5. Each paper is represented by metadata fields including title, venue, publication year, authors, and DOI or link. The training data contain the target label, while the public and private test files require predicted labels in the format specified by `Test_Submission.csv`.

The problem is an ordinal classification task rather than a standard nominal multiclass task. The labels have a natural order: an error between adjacent labels is less severe than an error between distant labels. Therefore, the competition metric is Quadratic Weighted Kappa (QWK), which measures agreement between predictions and ground truth while applying larger penalties to larger ordinal disagreements.

The selected final submission is `submission_v3_tv_q.csv`. It achieved a public leaderboard QWK of `0.75557` and a private leaderboard QWK of `0.73479`. According to the assignment PDF, the private leaderboard score is the official competition result for grading.

## 2. Methodology

The final methodology follows a stacked ensemble design. The pipeline starts from the raw CSV files, validates their structure and consistency, preprocesses missing values, builds or loads model-based features, trains a Ridge regression meta-model, calibrates continuous predictions into ordinal labels, and writes the final submission file.

The final v3 submission was generated from cached out-of-fold and test predictions produced during experimentation. These cached artifacts include transformer fine-tuning outputs, frozen scientific embedding heads, TF-IDF models, metadata encodings, and an earlier stacked model. The consolidated reproduction script, `reproduce_submission_v3_tv_q.py`, reproduces the exact selected CSV from the raw data plus the saved final artifact `outputs/rebuild_v3_oof.npz`. The script verifies that the reproduced file exactly matches `outputs/submission_v3_tv_q.csv`.

The main rationale for the approach is that the paper title contains strong topical information, while the venue supplies an important prior about expected label distribution. The final solution combines these two information sources with multiple model families. Text models capture lexical and semantic similarity; venue and venue-year features capture structured metadata patterns; nearest-neighbor features capture similarity to labeled papers; and the Ridge meta-model combines these signals into a continuous ordinal score.

The ordinal nature of the labels is handled in two ways. First, most base learners and the final meta-learner predict continuous scores on the 1 to 5 scale instead of only class probabilities. Second, validation and calibration are performed with QWK-aware thresholding or quantile mapping so that continuous scores are converted into integer labels in a way suitable for the competition metric.

## 3. Data Preprocessing

The raw files used in the project are:

- `train.csv`: 2,494 labeled papers with 7 columns.
- `public_test.csv`: 298 unlabeled papers with 6 columns.
- `private_test.csv`: 298 unlabeled papers with 6 columns.
- `Test_Submission.csv`: 596 IDs in the required submission order.

The train columns are `id`, `title`, `venue`, `year`, `authors`, `doi`, and `Label`. The public and private test files contain the same metadata columns except for `Label`.

Initial validation checked data shapes, missing values, duplicate IDs, label distribution, venue distribution, and submission ID consistency. No duplicate IDs were found in train, public test, private test, or the submission template. Missing values occurred mainly in the `authors` column: 192 rows in train, 19 rows in public test, and 22 rows in private test. Titles, venues, years, and DOI/link values were complete in the raw files.

Preprocessing used conservative normalization. Missing titles were replaced with an empty string, missing venues with `unknown`, and missing authors or DOI/link values with empty strings when those fields were used. Year values were converted to numeric form for metadata features. Public and private test rows were concatenated, and the resulting ID order was checked against `Test_Submission.csv`.

Additional consistency checks were performed. There were no shared normalized titles between train and test, so title-based leakage was not available. Three DOI/link strings appeared in both train and test; these were inspected but not used as deterministic leakage. The final submission was also checked to confirm exactly 596 rows, correct ID order, no missing predictions, and labels within the valid range 1 to 5.

The training label distribution is imbalanced. Label 1 is the largest class with 903 rows, or 36.21% of the training set. Labels 2, 3, 4, and 5 account for 20.61%, 17.56%, 14.72%, and 10.91%, respectively. This imbalance motivated the use of stratified cross-validation and careful calibration.

The venue distribution is also important. The training set contains `cav`, `iclp`, `kr`, `lics`, and `lpnmr`, while the test set contains only `cav`, `kr`, `lics`, and `lpnmr`. The absence of `iclp` from the test set motivated validation and calibration on the train subset whose venues match the test venues.

## 4. Feature Engineering

The final v3 stack uses 23 feature columns. Most are continuous out-of-fold predictions from base models. This design allows heterogeneous model families to be combined while preserving a clean validation structure: each training-row feature is generated by a model that did not train on that row.

The first feature group consists of fine-tuned transformer regression predictions. These include SciBERT, SPECTER2-based, and BGE-based models. The retained project notes indicate that venue-prefixed title input, such as `venue: <venue>. <title>`, was useful because it exposes the venue prior directly to the text model. Regression heads are appropriate because the labels are ordered.

The second feature group consists of frozen scientific embedding heads. SPECTER2 and BGE embeddings were used with Ridge regression and logistic expected-value heads. These features represent semantic similarity between paper titles or title-plus-abstract text. They are useful because papers with similar scientific language and topics often have related relevance labels.

The third feature group consists of TF-IDF text models. Word n-grams capture topical phrases such as "answer set", "logic programming", and "reasoning". Character n-grams provide robustness to morphological variations, abbreviations, and venue-specific terminology. The final stack includes title TF-IDF logistic expected-value predictions, word-level Ridge predictions, character-level Ridge predictions, and combined word-character Ridge predictions.

The fourth feature group consists of metadata and retrieval features. Venue target encoding and venue-year target encoding capture structured label priors. Nearest-neighbor features (`knn10` and `knn25`) estimate a paper's label by averaging labels of similar training papers in embedding space. The `frontmatter` feature detects proceedings, prefaces, and similar frontmatter titles, which were strongly associated with label 1 in training.

The fifth feature group is `s2_meta`, an earlier stacked ensemble prediction used as an additional base signal. This makes the final v3 stack a second-level ensemble over both individual base learners and an earlier meta-model.

The complete feature list is saved in `final_materials/tables/v3_feature_list.csv`.

## 5. Model Architecture

The final model architecture is a stacked regression ensemble. Each base feature is a continuous score related to the ordinal target. These scores are combined by a Ridge regression meta-learner.

The stack was trained using 5-fold StratifiedKFold. Stratification preserves the label distribution in each fold, which is important because the label classes are imbalanced. Multiple fold seeds were evaluated to reduce dependence on a single random split. The evaluated Ridge alpha values were 0.3, 0.5, 1.0, 2.0, 5.0, and 10.0. The saved final v3 artifact selected `alpha=5.0`.

The final meta-model outputs a continuous score. This score is not used directly as the final label. Instead, it is calibrated into labels 1 to 5. The selected submission uses the `tv_q` calibration strategy, which maps continuous test-score ranks to the label distribution of the training subset whose venues also appear in the test set. This was chosen because `iclp` is present in training but absent from test.

After quantile calibration, a deterministic post-processing rule sets frontmatter or proceedings-style titles to label 1. In the test set, 14 rows matched this rule. This rule is narrow and interpretable: it targets non-standard paper entries such as proceedings, prefaces, editorials, and tables of contents.

## 6. Experimental Setup

The main validation metric was QWK, matching the Kaggle evaluation metric. Out-of-fold predictions were used to avoid evaluating a model on rows it had trained on. Continuous predictions were converted to labels using threshold optimization for validation. The saved final v3 artifact records the continuous out-of-fold predictions, continuous test predictions, feature names, selected alpha, thresholds, frontmatter masks, and submission IDs.

Two validation views were considered:

- All training rows.
- The test-venue subset, which includes only training rows from venues that appear in the test data.

This distinction is important because the test set excludes `iclp`. A validation estimate on all training rows may partially reflect a venue distribution not present in the final evaluation set.

From the saved final v3 artifact, the threshold-tuned Ridge stack achieved:

- OOF QWK on all training rows: `0.6719`.
- OOF QWK on the test-venue subset: `0.6708`.

The final Kaggle submission uses quantile calibration rather than only threshold-tuned labels. Therefore, the OOF values above should be interpreted as validation of ranking quality, while the leaderboard scores evaluate the final combination of ranking and calibration.

The project notes record earlier submissions around a public score of approximately 0.72. Exact private scores for those intermediate submissions are not available from the current files. The selected final submission improved the public score to 0.75557 and achieved a private score of 0.73479.

## 7. Evaluation and Analysis

The final selected submission is `submission_v3_tv_q.csv`. Its leaderboard results are:

- Public QWK: `0.75557`.
- Private QWK: `0.73479`.

The public-private gap is `0.02078`, with the public score higher than the private score. This indicates that the public split was more favorable to the selected model and calibration, or that there is a modest distribution difference between the public and private samples. Because the assignment uses the private leaderboard for grading, the private QWK of 0.73479 is the main result.

The final prediction distribution is:

- Label 1: 229 rows, 38.42%.
- Label 2: 128 rows, 21.48%.
- Label 3: 106 rows, 17.79%.
- Label 4: 82 rows, 13.76%.
- Label 5: 51 rows, 8.56%.

This distribution is close to the train subset restricted to test venues, which was the target of the `tv_q` calibration. This calibration is reasonable because the test data do not contain all training venues.

The strengths of the solution are its use of complementary signals and its explicit handling of the ordinal target. Title text captures topical relevance. Scientific embeddings capture semantic similarity. Venue features encode structured prior information. Stacking combines these signals into a more stable continuous ranking than any single feature group. Calibration then adapts the ranking to the observed test-venue composition.

The main weakness is that the final pipeline depends on cached base predictions from earlier experiments. The selected final CSV is exactly reproducible from the available artifacts, but retraining every base model from scratch would require the original model cache, GPU environment, and longer runtime. Another limitation is that the middle classes are likely harder to distinguish from title and venue alone. Without full abstracts or stronger external signals, labels 2, 3, and 4 can be semantically close.

The available files do not contain a complete per-class error analysis on the hidden private labels, because those labels are not available. Error analysis on training OOF predictions could be expanded in future work, especially for confusion among labels 2, 3, and 4.

## 8. Conclusion and Future Work

This project developed a stacked ordinal regression ensemble for classifying academic papers into ordered relevance labels. The final solution combines title text models, scientific transformer and embedding predictions, venue-based metadata features, nearest-neighbor retrieval, and a deterministic frontmatter rule. A Ridge regression meta-model produces continuous ordinal scores, and the selected `tv_q` calibration maps those scores to labels according to the test-venue training distribution.

The selected final submission, `submission_v3_tv_q.csv`, achieved a public QWK of `0.75557` and a private QWK of `0.73479`. The private score is the official competition result for grading.

Future work could improve the solution in several ways:

- Train a fully reproducible end-to-end pipeline that regenerates all base predictions without relying on cached artifacts.
- Use a more explicit ordinal learning objective, such as cumulative link models or ordinal neural heads.
- Perform deeper OOF error analysis for middle-class confusion.
- Improve calibration robustness under venue and public-private distribution shifts.
- Evaluate whether richer paper metadata, if permitted by competition rules, can improve class separation.

The final result demonstrates that combining text, metadata, semantic embeddings, and QWK-aware calibration is effective for this ordinal data mining task while leaving room for improved reproducibility and more detailed error analysis.
