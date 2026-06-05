# Final Project Materials

This folder contains the report and presentation materials for the selected final Kaggle submission:

- Final selected submission: `outputs/submission_v3_tv_q.csv`
- Public leaderboard QWK: `0.75557`
- Private leaderboard QWK: `0.73479`
- Reproduction script: `../reproduce_submission_v3_tv_q.py`

## Files

- `PIPELINE_EXPLANATION.md` - step-by-step explanation from raw data to final submission.
- `PRESENTATION_CONTENT.md` - 3-5 minute slide outline, bullets, speaker notes, and visual suggestions.
- `FINAL_REPORT.md` - formal academic report content.
- `TABLES_AND_FIGURES.md` - generated tables/figures and suggested placement.
- `CHECKLIST.md` - assignment requirement and LMS submission checklist.
- `tables/` - CSV tables generated from the raw data, saved artifacts, and final submission.
- `figures/` - PNG figures for report and slides.

## Reproduction

Run from the project root with the project Python environment:

```powershell
.\env\Scripts\python.exe reproduce_submission_v3_tv_q.py
```

The script writes `outputs/reproduced_submission_v3_tv_q.csv` and verifies that it exactly matches `outputs/submission_v3_tv_q.csv`.

For an audit rebuild of the final v3 stack from cached base predictions and fresh TF-IDF features:

```powershell
.\env\Scripts\python.exe reproduce_submission_v3_tv_q.py --rebuild-stack
```

The default path is fast and exact because it uses `outputs/rebuild_v3_oof.npz`, the saved final v3 OOF/test artifact. The slower `--rebuild-stack` path is retained for transparency but may take substantially longer because it repeats threshold searches across multiple Ridge stack configurations.
