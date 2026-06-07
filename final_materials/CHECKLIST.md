# Assignment Submission Checklist

This checklist is based on `Requirement/DM_ass_252 (1).pdf` and the final selected submission `submission_v3_tv_q.csv`.

## 1. Presentation Video Requirement

Required duration: 3-5 minutes.

Covered content:

- Problem understanding: covered in `PRESENTATION_CONTENT.md`, Slide 2.
- Data preprocessing methods: covered in Slide 3.
- Feature engineering techniques: covered in Slide 4.
- Model development and training: covered in Slide 5.
- Validation strategy: covered in Slide 6.
- Experimental results and discussion: covered in Slide 7.

Status: ready to record.

Recommended video structure:

- Use the 8-slide outline in `PRESENTATION_CONTENT.md`.
- Target about 4 minutes 20 seconds.
- Include the leaderboard screenshot on Slide 1 or Slide 7.

## 2. Private Leaderboard Screenshot Requirement

The assignment PDF requires a screenshot clearly showing:

- Student/team name.
- Private leaderboard ranking.
- Final private leaderboard score.

Current final selected result:

- Submission: `submission_v3_tv_q.csv`.
- Public QWK: `0.75557`.
- Private QWK: `0.73479`.

Status: needs final visual confirmation.

Important note:

- The screenshot in the prompt shows the student name and scores, but the visible crop may not show the private leaderboard ranking.
- For LMS submission, include a full private leaderboard or final submissions screenshot where the private rank is visible.

## 3. Final Report Requirement

Required sections from assignment PDF:

- Introduction: covered in `FINAL_REPORT.md`, Section 1.
- Methodology: covered in Section 2.
- Data preprocessing: covered in Section 3.
- Feature engineering: covered in Section 4.
- Model architecture: covered in Section 5.
- Experimental setup: covered in Section 6.
- Evaluation and analysis: covered in Section 7.
- Conclusion and future work: covered in Section 8.

Status: report content prepared.

Recommended attachments/figures:

- `figures/pipeline_diagram.png`
- `figures/label_distribution_train.png`
- `figures/missing_values_summary.png`
- `figures/train_vs_prediction_distribution.png`
- `figures/prediction_distribution_v3_tv_q.png`

Recommended tables:

- `tables/dataset_summary.csv`
- `tables/label_distribution_train.csv`
- `tables/missing_values_summary.csv`
- `tables/experiment_comparison.csv`
- `tables/final_submission_score.csv`
- `tables/consistency_checks.csv`

## 4. Code Reproducibility Requirement

Prepared file:

- `reproduce_submission_v3_tv_q.py`

Verified command:

```powershell
.\env\Scripts\python.exe reproduce_submission_v3_tv_q.py
```

Verification result:

- Writes `outputs/reproduced_submission_v3_tv_q.csv`.
- Prints final submission summary in the terminal.
- Validates output shape, columns, and label range.
- Default exact reproduction uses `outputs/rebuild_v3_oof.npz`.
- Optional exact comparison can be run with `--expected outputs/submission_v3_tv_q.csv`.

Status: verified.

Reproducibility limitation to mention if asked:

- The exact selected v3 file depends on saved OOF/test artifacts from earlier model training.
- The final CSV is exactly reproducible from the available raw files plus saved artifacts.
- Full retraining of every base model from raw CSV only cannot be guaranteed to reproduce the same historical labels because transformer training and embedding/model caches introduce nondeterminism and version dependence.

## 5. LMS Submission Package

Submit through LMS before the Phase 3 deadline stated in the PDF.

Recommended package contents:

- Final report PDF or document generated from `FINAL_REPORT.md`.
- Presentation video file or link.
- Private leaderboard screenshot showing name, private rank, and private score.
- Final selected submission CSV: `outputs/submission_v3_tv_q.csv`.
- Reproduction script: `reproduce_submission_v3_tv_q.py`.
- Optional support folder: `final_materials/` with tables and figures.

Status: materials prepared; final screenshot/rank visibility still needs confirmation.
