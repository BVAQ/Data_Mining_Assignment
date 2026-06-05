# Presentation Content for 3-5 Minute Video

Recommended total duration: about 4 minutes 20 seconds.

## Slide 1 - Title and Competition Overview

Estimated time: 25 seconds.

Bullets:

- Data Mining 252 Kaggle Competition - Stage 2.
- Task: classify academic papers into ordinal labels 1 to 5.
- Final selected submission: `submission_v3_tv_q.csv`.
- Public QWK: `0.75557`; private QWK: `0.73479`.

Speaker notes:

This project addresses the Kaggle competition for Data Mining 252 Stage 2. The goal is to classify research papers related to Answer Set Programming and neighboring areas into five ordered relevance classes. My final selected submission is `submission_v3_tv_q.csv`, which achieved a public Quadratic Weighted Kappa of 0.75557 and a private score of 0.73479.

Suggested visual:

- Title slide with the leaderboard screenshot.
- Make sure the screenshot shows student/team name, private leaderboard rank, and final private score.

## Slide 2 - Problem Understanding

Estimated time: 35 seconds.

Bullets:

- Input: title, venue, year, authors, DOI/link.
- Target: ordered label from 1 to 5.
- Metric: Quadratic Weighted Kappa.
- Larger ordinal mistakes receive stronger penalties.

Speaker notes:

Each row represents one academic paper. The model receives metadata such as title, venue, publication year, author list, and DOI or link. The target is an ordered label from 1 to 5, so this is not just ordinary multiclass classification. Predicting a nearby class is less harmful than predicting a very distant class. For that reason, I used Quadratic Weighted Kappa during validation, because it evaluates agreement while penalizing large ordinal errors more strongly.

Suggested visual:

- Small table showing the input columns and target column.
- One-line metric formula or diagram showing that 1 versus 5 is a larger error than 1 versus 2.

## Slide 3 - Data Preprocessing

Estimated time: 35 seconds.

Bullets:

- Train: 2,494 rows.
- Public test: 298 rows; private test: 298 rows.
- Missing values mainly in `authors`.
- No duplicate IDs; final IDs match `Test_Submission.csv`.
- Test venues exclude `iclp`.

Speaker notes:

I first validated the raw files. The training set contains 2,494 labeled papers, and the public and private test sets contain 298 rows each. Missing values were mainly in the author field, while titles, venues, years, and DOI/link values were complete. I checked duplicate IDs, train-test title overlap, and final submission ID order. A useful observation is that the training set contains five venues, but the test set excludes `iclp`. This influenced the calibration strategy later.

Suggested visual:

- `figures/missing_values_summary.png`.
- Dataset summary table from `tables/dataset_summary.csv`.
- Venue summary table from `tables/venue_summary.csv`.

## Slide 4 - Feature Engineering

Estimated time: 45 seconds.

Bullets:

- Title text features: word and character TF-IDF.
- Scientific embedding features: SPECTER2 and BGE.
- Fine-tuned transformer regression predictions.
- Metadata features: venue and venue-year target encoding.
- Retrieval and rule features: kNN labels and frontmatter flag.

Speaker notes:

The final solution combines text and metadata. Title text is represented with word and character TF-IDF models. I also used cached predictions from scientific transformer models and embedding-based models such as SPECTER2 and BGE. Venue is important because different venues have different label distributions, so venue and venue-year encodings were included. I also used nearest-neighbor retrieval features in embedding space and a frontmatter flag for proceedings or preface-style titles, which are usually low relevance.

Suggested visual:

- Pipeline diagram: `figures/pipeline_diagram.png`.
- Feature group table from `tables/v3_feature_list.csv`.

## Slide 5 - Model Development and Training

Estimated time: 45 seconds.

Bullets:

- Base predictions are continuous scores on the 1-5 scale.
- Meta-model: multi-seed Ridge regression stack.
- 5-fold StratifiedKFold.
- Selected stack alpha: `5.0`.
- Final feature matrix: 23 features.

Speaker notes:

The final model is a stacked regression ensemble. Instead of treating labels only as unrelated classes, the base learners produce continuous scores on the 1 to 5 ordinal scale. These base predictions are combined by a Ridge regression meta-learner. I used five stratified folds and multiple random fold seeds to reduce dependence on one fold split. The final v3 artifact selected Ridge alpha 5.0 and used 23 stacked feature columns.

Suggested visual:

- Stacking diagram: base learners feeding a Ridge meta-model.
- Include a small box with `alpha=5.0`, `5 folds`, and `23 features`.

## Slide 6 - Validation Strategy

Estimated time: 40 seconds.

Bullets:

- OOF predictions used for validation.
- QWK optimized on continuous predictions through thresholds.
- Evaluated on all train rows and test-venue subset.
- Test-venue subset excludes `iclp`.
- Rebuild v3 OOF QWK: about `0.6719` all, `0.6708` test-venue.

Speaker notes:

Validation was based on out-of-fold predictions, so each training paper was predicted by a model that did not train on that row. I evaluated QWK after mapping continuous predictions to labels. Because the test set contains only four of the five training venues, I also evaluated the training subset whose venues match the test venues. The saved final v3 ranking reached approximately 0.6719 QWK on all training rows and 0.6708 on the test-venue subset.

Suggested visual:

- Cross-validation diagram.
- Experiment comparison table from `tables/experiment_comparison.csv`.

## Slide 7 - Experimental Results and Discussion

Estimated time: 45 seconds.

Bullets:

- Earlier submissions were around the 0.72 public plateau.
- Final selected submission: `submission_v3_tv_q.csv`.
- Public QWK: `0.75557`.
- Private QWK: `0.73479`.
- Public-private gap: `0.02078`.

Speaker notes:

Earlier experimentation reached a public leaderboard plateau around 0.72. The selected v3 submission improved the public score to 0.75557 and achieved a private score of 0.73479. The public score is higher than the private score by about 0.02078. This indicates that the public split was somewhat more favorable, or that there is a mild distribution difference between public and private samples. The private score is the official result for grading.

Suggested visual:

- Final score table from `tables/final_submission_score.csv`.
- Prediction distribution figure: `figures/prediction_distribution_v3_tv_q.png`.
- Train versus prediction distribution: `figures/train_vs_prediction_distribution.png`.

## Slide 8 - Conclusion

Estimated time: 35 seconds.

Bullets:

- Strongest signals came from title text, scientific embeddings, and venue.
- Ordinal regression and QWK-aware calibration were important.
- Final private QWK: `0.73479`.
- Future work: stronger ordinal models, more robust calibration, and deeper error analysis.

Speaker notes:

In conclusion, the final solution combines title-based text modeling, scientific embeddings, fine-tuned transformer predictions, venue metadata, and a stacked Ridge meta-model. Treating the target as ordinal and calibrating predictions carefully were important for QWK performance. The final selected private score is 0.73479. Future improvements could include a more explicit ordinal learning objective, stronger calibration on distribution shifts, and a more detailed error analysis of confusing middle classes.

Suggested visual:

- Summary slide with three key components: text, metadata, stacking.
- Add final score and one short future-work list.

## Timing Summary

| Slide | Topic | Time |
|---|---:|---:|
| 1 | Overview | 0:25 |
| 2 | Problem understanding | 0:35 |
| 3 | Data preprocessing | 0:35 |
| 4 | Feature engineering | 0:45 |
| 5 | Model development | 0:45 |
| 6 | Validation strategy | 0:40 |
| 7 | Results and discussion | 0:45 |
| 8 | Conclusion | 0:35 |
| Total |  | 4:20 |
