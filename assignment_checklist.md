# Assignment Checklist — DM252 Data Mining

## Presentation Video Requirements (3–5 minutes)

- [x] Problem understanding — Slide 2: ordinal classification, QWK metric, data overview
- [x] Data preprocessing methods — Slide 3: missing values, text cleaning, venue prefix, frontmatter
- [x] Feature engineering techniques — Slide 4: 23 features from 4 groups
- [x] Model development and training — Slide 5: multi-stage stacking, Ridge meta-learner
- [x] Validation strategy — Slide 6: 5-fold CV, test-venue evaluation, calibration comparison
- [x] Experimental results and discussion — Slide 7: score progression, final scores, generalization
- [ ] **ACTION REQUIRED**: Record the video narrating the slides (3–5 min)
- [ ] **ACTION REQUIRED**: Include leaderboard screenshot in the video

## Final Report Requirements

- [x] Introduction — Section 1: competition background, dataset, metric, final score
- [x] Methodology — Section 2: pipeline overview, design rationale
- [x] Data preprocessing — Section 3: missing values, text cleaning, metadata processing
- [x] Feature engineering — Section 4: all 23 features with descriptions and individual QWK
- [x] Model architecture — Section 5: Ridge stacking, calibration, post-processing
- [x] Experimental setup — Section 6: CV strategy, reproducibility, hyperparameters, comparison
- [x] Evaluation and analysis — Section 7: validation, LB results, gap analysis, error analysis
- [x] Conclusion and future work — Section 8: summary, limitations, improvements

## Code Requirements

- [x] Consolidated reproduction script: `reproduce_submission_v3_tv_q.py`
- [x] Data loading and validation
- [x] Feature engineering pipeline
- [x] Model training (multi-seed Ridge stacking)
- [x] Calibration and prediction
- [x] Output verification (exact match with original submission)
- [x] Fixed random seeds for reproducibility
- [x] Clear comments and section structure

## Tables and Figures

- [x] Dataset summary table — in pipeline explanation and report
- [x] Label distribution table — in pipeline explanation and report
- [x] Missing value summary — in report Section 3
- [x] Feature list with individual QWK — in report Section 4
- [x] Model/experiment comparison table — in report Section 6.5
- [x] Calibration strategy comparison — in report Section 7.2
- [x] Final submission score table — in report Section 7.3
- [x] Predicted label distribution — in report Section 7.5
- [x] Pipeline diagram description — in presentation Slide 5

## Deliverable Files

| File | Status | Description |
|------|--------|-------------|
| `reproduce_submission_v3_tv_q.py` | ✅ Created | Consolidated reproduction code |
| `pipeline_explanation.md` | ✅ Created | Step-by-step pipeline explanation |
| `presentation_content.md` | ✅ Created | Slide content + speaker notes |
| `report_content.md` | ✅ Created | Formal academic report |
| `assignment_checklist.md` | ✅ Created | This checklist |
| `submission_v3_tv_q.csv` | ✅ Exists | Final selected submission |

## LMS Submission Requirements

- [ ] **ACTION REQUIRED**: Upload final submission CSV to Kaggle (already done)
- [ ] **ACTION REQUIRED**: Take leaderboard screenshot showing final scores
- [ ] **ACTION REQUIRED**: Upload report (PDF format) to LMS
- [ ] **ACTION REQUIRED**: Upload presentation video to LMS
- [ ] **ACTION REQUIRED**: Upload source code to LMS
- [ ] **ACTION REQUIRED**: Verify all file sizes within LMS limits
