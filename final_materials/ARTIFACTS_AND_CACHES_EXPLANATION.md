# Artifact and Cache Explanation for `reproduce_submission_v3_tv_q.py`

## 1. Why Artifacts Are Needed

The selected final submission is `outputs/submission_v3_tv_q.csv`. It was not produced by a single lightweight model trained from raw CSV files only. It was produced by a stacked ensemble whose base features include cached transformer fine-tuning predictions, frozen scientific embedding models, TF-IDF models, metadata encodings, nearest-neighbor retrieval features, and an earlier stacked model.

For that reason, exact reproduction of the historical leaderboard file requires saved prediction artifacts. Without these artifacts, retraining from raw CSV files can produce a valid end-to-end submission, but it will not be guaranteed to match `submission_v3_tv_q.csv` exactly.

The current `reproduce_submission_v3_tv_q.py` restores the exact v3 reproduction behavior:

- Default mode: fast exact reproduction from `outputs/rebuild_v3_oof.npz`.
- Audit mode: `--rebuild-stack`, which rebuilds the v3 stack from cached base predictions and fresh TF-IDF features.

## 2. Default Artifact Loaded by the Script

### `outputs/rebuild_v3_oof.npz`

This is the final saved v3 artifact. It is the only model artifact loaded in the default path.

The script loads it in `reproduce_from_saved_v3_artifact()`:

```python
artifact_path = root / "outputs" / "rebuild_v3_oof.npz"
artifact = np.load(artifact_path, allow_pickle=True)
```

Contents:

| key | shape | meaning |
|---|---:|---|
| `oof` | `(2494,)` | Final continuous out-of-fold scores for train rows. |
| `test` | `(596,)` | Final continuous scores for public + private test rows in `Test_Submission.csv` order. |
| `y` | `(2494,)` | Ground-truth train labels. |
| `fm_train` | `(2494,)` | Boolean frontmatter mask for train rows. |
| `fm_test` | `(596,)` | Boolean frontmatter mask for test rows. |
| `tv_mask` | `(2494,)` | Boolean mask for train rows whose venues appear in test: `cav`, `lics`, `kr`, `lpnmr`. |
| `sub_ids` | `(596,)` | Submission IDs in final output order. |
| `feature_names` | `(23,)` | Names of the 23 stacked base features. |
| `alpha` | scalar | Selected Ridge stack regularization parameter, `5.0`. |
| `th` | `(4,)` | Thresholds optimized on all OOF train rows. |
| `th_tv` | `(4,)` | Thresholds optimized on the test-venue OOF subset. |

The default script does not retrain the stack. It uses `artifact["test"]` as the final continuous ranking, then reproduces the selected `tv_q` calibration:

1. Compute label distribution from train rows where `tv_mask=True`.
2. Convert continuous test scores to labels by quantile calibration.
3. Override frontmatter-style test titles to label 1.
4. Save `outputs/reproduced_submission_v3_tv_q.csv`.
5. Compare it to `outputs/submission_v3_tv_q.csv`.

Verified result:

- ID order match: true.
- Exact label match: true.
- Label agreement: `1.0000`.

## 3. Final 23 Features Stored in the v3 Artifact

`outputs/rebuild_v3_oof.npz` records these 23 features:

| index | feature | source |
|---:|---|---|
| 1 | `ft_scibert` | Cached transformer fine-tuning prediction consumed by `gm/_feats.npz`. |
| 2 | `ft_scibert_s2` | Cached transformer fine-tuning/S2 variant consumed by `gm/_feats.npz`. |
| 3 | `ft_specter2ft` | Cached SPECTER2 fine-tuning prediction consumed by `gm/_feats.npz`. |
| 4 | `ft_specter2ft_s2` | Cached SPECTER2/S2 variant consumed by `gm/_feats.npz`. |
| 5 | `specter2_title_ridge` | Ridge regression head on SPECTER2 title embeddings. |
| 6 | `specter2_title_lrev` | Logistic-regression expected-value head on SPECTER2 title embeddings. |
| 7 | `bge_title_ridge` | Ridge regression head on BGE title embeddings. |
| 8 | `bge_title_lrev` | Logistic-regression expected-value head on BGE title embeddings. |
| 9 | `specter2_titabs_ridge` | Ridge regression head on SPECTER2 title+abstract embeddings. |
| 10 | `specter2_titabs_lrev` | Logistic-regression expected-value head on SPECTER2 title+abstract embeddings. |
| 11 | `title_tfidf_lrev` | Title TF-IDF logistic-regression expected-value prediction. |
| 12 | `te_venue` | Cross-validation-safe venue target encoding. |
| 13 | `te_venue_year` | Cross-validation-safe venue-year target encoding. |
| 14 | `frontmatter` | Proceedings/preface/frontmatter binary feature. |
| 15 | `knn10` | Average label of 10 nearest neighbors in SPECTER2 title embedding space. |
| 16 | `knn25` | Average label of 25 nearest neighbors in SPECTER2 title embedding space. |
| 17 | `ft2_bge_venue_title` | BGE transformer regression fine-tune using `venue: <venue>. <title>`. |
| 18 | `ft2_scibert_venue_title` | SciBERT transformer regression fine-tune using `venue: <venue>. <title>`. |
| 19 | `ft2_specter2_venue_title` | SPECTER2 transformer regression fine-tune using `venue: <venue>. <title>`. |
| 20 | `s2_meta` | Earlier Stage-2 stacked ensemble meta prediction from `outputs/s2new_oof.npz`. |
| 21 | `tfidf_word_a3.0` | Fresh word-level title TF-IDF Ridge feature rebuilt by `src/rebuild_v3.py`. |
| 22 | `tfidf_char_a3.0` | Fresh character-level title TF-IDF Ridge feature rebuilt by `src/rebuild_v3.py`. |
| 23 | `tfidf_combined` | Fresh word+character title TF-IDF Ridge feature rebuilt by `src/rebuild_v3.py`. |

## 4. Artifacts Loaded in `--rebuild-stack` Audit Mode

The optional audit mode does not use `outputs/rebuild_v3_oof.npz`. Instead, it reconstructs the final feature matrix from cached base predictions, then retrains the Ridge stack.

Command:

```powershell
.\env\Scripts\python.exe reproduce_submission_v3_tv_q.py --rebuild-stack
```

Artifacts loaded in this mode:

### 4.1 `gm/_feats.npz`

This file contains 16 aligned base features:

| key | shape | meaning |
|---|---:|---|
| `X_oof` | `(2494, 16)` | OOF feature matrix for train rows. |
| `X_test` | `(596, 16)` | Test feature matrix in submission order. |
| `y` | `(2494,)` | Train labels. |
| `names` | `(16,)` | Names of the 16 base features. |
| `train_venue` | `(2494,)` | Train venues. |
| `test_venue` | `(596,)` | Test venues in submission order. |
| `sub_ids` | `(596,)` | Submission IDs. |
| `fold` | `(2494,)` | Fold assignment used for OOF construction. |

`gm/_feats.npz` was generated by `gm/features.py`.

The 16 features inside it are:

1. `ft_scibert`
2. `ft_scibert_s2`
3. `ft_specter2ft`
4. `ft_specter2ft_s2`
5. `specter2_title_ridge`
6. `specter2_title_lrev`
7. `bge_title_ridge`
8. `bge_title_lrev`
9. `specter2_titabs_ridge`
10. `specter2_titabs_lrev`
11. `title_tfidf_lrev`
12. `te_venue`
13. `te_venue_year`
14. `frontmatter`
15. `knn10`
16. `knn25`

### 4.2 Precomputed fine-tuning artifacts consumed by `gm/_feats.npz`

`gm/features.py` consumes these files from `outputs/`:

- `outputs/ft_scibert_oof.npy`
- `outputs/ft_scibert_test.npy`
- `outputs/ft_scibert_ids_test.npy`
- `outputs/ft_scibert_s2_oof.npy`
- `outputs/ft_scibert_s2_test.npy`
- `outputs/ft_scibert_s2_ids_test.npy`
- `outputs/ft_specter2ft_oof.npy`
- `outputs/ft_specter2ft_test.npy`
- `outputs/ft_specter2ft_ids_test.npy`
- `outputs/ft_specter2ft_s2_oof.npy`
- `outputs/ft_specter2ft_s2_test.npy`
- `outputs/ft_specter2ft_s2_ids_test.npy`

These are cached OOF/test predictions from earlier transformer fine-tuning experiments. Based on the retained fine-tuning script `src/s2_finetune.py`, the method is:

- Transformer encoder from Hugging Face.
- Regression head predicting a continuous score on the 1-5 label scale.
- MSE loss.
- 5-fold StratifiedKFold OOF generation.
- Test predictions averaged across folds.
- Input text built from venue, year, title, and available abstract text.

The exact command logs for the `_s2` variants are not fully available in the current files, so the report should describe them as cached S2/enriched variants rather than inventing unverified details.

### 4.3 Frozen embedding caches consumed by `gm/_feats.npz`

`gm/features.py` also consumes embedding files from `gm/emb/`, generated by `gm/emb_extract.py`.

Relevant embedding models:

| embedding tag | Hugging Face model | text variant | pooling | used in final features |
|---|---|---|---|---|
| `specter2_title` | `allenai/specter2_base` | title only | CLS | yes |
| `specter2_titabs` | `allenai/specter2_base` | title + abstract | CLS | yes |
| `bge_title` | `BAAI/bge-base-en-v1.5` | title only | CLS | yes |
| `bge_titabs` | `BAAI/bge-base-en-v1.5` | title + abstract | CLS | generated but not used in final 16 features |
| `mpnet_titabs` | `sentence-transformers/all-mpnet-base-v2` | title + abstract | mean | generated but not used in final 16 features |
| `scibert_titabs` | `allenai/scibert_scivocab_uncased` | title + abstract | mean | generated but not used in final 16 features |

From these embeddings, `gm/features.py` trains CV-safe heads:

- Ridge regression heads.
- Logistic regression expected-value heads.
- kNN retrieval label features over SPECTER2 title embeddings.

### 4.4 Extra GM fine-tune artifacts from `gm/ft/`

In audit mode, `reproduce_submission_v3_tv_q.py` scans:

```python
gm/ft/ft2_*_oof.npy
```

It includes only these three extra fine-tune features:

- `gm/ft/ft2_bge_venue_title_oof.npy`
- `gm/ft/ft2_bge_venue_title_test.npy`
- `gm/ft/ft2_scibert_venue_title_oof.npy`
- `gm/ft/ft2_scibert_venue_title_test.npy`
- `gm/ft/ft2_specter2_venue_title_oof.npy`
- `gm/ft/ft2_specter2_venue_title_test.npy`

It explicitly excludes:

- `ft2_scibert_venue_titabs`
- `ft2_deberta_venue_title`

These were generated by `gm/finetune.py`. The retained script shows:

- Models:
  - `BAAI/bge-base-en-v1.5`
  - `allenai/scibert_scivocab_uncased`
  - `allenai/specter2_base`
- Input variant: `venue_title`
  - Text format: `venue: <venue>. <title>`
- Training:
  - Transformer encoder + linear regression head.
  - CLS token representation.
  - MSE loss.
  - AdamW optimizer.
  - OneCycleLR scheduler.
  - 5-fold StratifiedKFold.
  - 4 epochs in the recorded `gm/run_all.py` command.
  - Seeds `0,1,2,3` in the recorded `gm/run_all.py` command.

### 4.5 `outputs/s2new_oof.npz`

This is an earlier Stage-2 stacked ensemble artifact generated by `src/s2_stack.py`.

Contents:

| key | shape | meaning |
|---|---:|---|
| `names` | `(10,)` | Names of the S2 base models. |
| `Z` | `(2494, 10)` | S2 base OOF prediction matrix. |
| `Zte` | `(596, 10)` | S2 base test prediction matrix. |
| `y` | `(2494,)` | Train labels. |
| `meta_oof` | `(2494,)` | S2 meta-model OOF prediction. |
| `meta_te` | `(596,)` | S2 meta-model test prediction. |
| `th` | `(4,)` | S2 thresholds. |
| `n_pub` | scalar | Number of public test rows. |
| `n_pri` | scalar | Number of private test rows. |
| `ids_test` | `(596,)` | Test IDs. |

The final v3 script uses only:

- `meta_oof`
- `meta_te`

and appends them as the single feature `s2_meta`.

The S2 base models recorded in the artifact are:

1. `word_ridge`
2. `char_ridge`
3. `wc_logreg`
4. `specter2_ridge`
5. `specter2_knn`
6. `bge_ridge`
7. `bge_knn`
8. `lgbm_meta`
9. `ft_scibert`
10. `ft_specter2ft`

From `src/s2_stack.py`, these correspond to:

- Word TF-IDF Ridge regression on titles.
- Character TF-IDF Ridge regression on titles.
- Word+character TF-IDF LogisticRegression converted to expected label value.
- SPECTER2 embedding Ridge and kNN heads.
- BGE embedding Ridge and kNN heads.
- LightGBM regression on metadata, text SVD, embedding SVD, and target encoding.
- Cached transformer fine-tune predictions.

### 4.6 Fresh TF-IDF features rebuilt inside the v3 script

The final three v3 features are not loaded as artifacts in `--rebuild-stack` mode. They are rebuilt inside the script:

- `tfidf_word_a3.0`
- `tfidf_char_a3.0`
- `tfidf_combined`

They are trained with 5-fold StratifiedKFold:

- Word TF-IDF: `ngram_range=(1, 3)`, `min_df=2`, Ridge `alpha=3.0`.
- Character TF-IDF: `analyzer="char_wb"`, `ngram_range=(3, 6)`, `min_df=3`, Ridge `alpha=3.0`.
- Combined word+character TF-IDF: horizontal stack of word and character matrices, Ridge `alpha=3.0`.

## 5. Final Stack and Calibration

The final v3 stack combines all 23 features with a multi-seed Ridge regression stack.

Recorded final configuration from `outputs/rebuild_v3_oof.npz`:

- Selected Ridge alpha: `5.0`.
- Thresholds on all train rows:
  - `[1.848108, 2.487640, 3.280659, 3.732595]`
- Thresholds on test-venue subset:
  - `[1.788418, 2.487640, 2.948361, 3.732595]`

The final selected CSV uses `tv_q` calibration, not direct threshold labels:

1. Compute the label distribution among training rows where venue is in the test venue set.
2. Rank the continuous test scores.
3. Assign labels by quantiles to match this target distribution.
4. Override detected frontmatter rows to label 1.

The target test-venue training distribution is:

- Label 1: `0.3857`
- Label 2: `0.2139`
- Label 3: `0.1781`
- Label 4: `0.1379`
- Label 5: `0.0844`

The final prediction distribution is:

- Label 1: 229 rows, 38.42%.
- Label 2: 128 rows, 21.48%.
- Label 3: 106 rows, 17.79%.
- Label 4: 82 rows, 13.76%.
- Label 5: 51 rows, 8.56%.

## 6. How to Describe This in the Report

Suggested report wording:

> The final selected submission is exactly reproducible from the raw competition files and saved model-prediction artifacts. These artifacts store out-of-fold and test predictions from transformer fine-tuning, frozen scientific embeddings, TF-IDF models, metadata encodings, nearest-neighbor retrieval features, and an earlier stacked ensemble. The final reproduction script validates the raw data, loads the saved v3 continuous test predictions, applies the selected test-venue quantile calibration, applies a deterministic frontmatter rule, and writes a CSV whose labels exactly match the selected leaderboard submission.

Suggested limitation wording:

> A fully raw-data-only retraining script cannot be guaranteed to reproduce the exact historical leaderboard file because several base learners are transformer models whose predictions depend on saved training runs, model caches, random seeds, GPU determinism, and library versions. Therefore, deterministic reproduction is performed from saved OOF/test prediction artifacts, while the retained source scripts document how those artifacts were generated.
