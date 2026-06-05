#!/bin/bash
# Cleanup script - removes unnecessary files, keeps only what's needed for 0.71+ scores
# Review this file before running: bash cleanup.sh

cd "$(dirname "$0")"

echo "=== CLEANUP SCRIPT ==="
echo "This will delete ~225 files and keep 31 essential files"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Delete old reports
rm -f DIAGNOSTIC_REPORT.md EXECUTIVE_SUMMARY.md README_STAGE2.md REPORT_BREAKTHROUGH.md STAGE3_FINAL_REPORT.md SUBMISSION_SUMMARY.md

# Delete old scripts (keep only s5_core.py, s7_inflation.py, build_dataset.py)
rm -f check.py competition_breakthrough.py data.py eval_xgb.py explore.py explore_labels.py fix_test_llm.py
rm -f pull_available_models.py search_log.py test_json_api.py test_llm.py
rm -f s5_exp1.py s5_exp2.py s5_exp3.py s5_final_a.py s5_final_b.py s5_finetune.py s5_submissions.py s6_venue_aggressive.py

# Delete catboost logs
rm -rf catboost_info/

# Delete intermediate outputs
rm -f outputs/_*.npy outputs/_*.npz outputs/_*.txt
rm -f outputs/*.json outputs/*.txt

# Delete logs
rm -f outputs/_log_*.txt

# Delete embeddings (can regenerate if needed)
rm -rf outputs/emb/

# Delete stage8 cache
rm -rf outputs/stage8_cache/

# Delete old preprocessed data (keep only s5_*.parquet and caches)
# Note: Keeping all data/raw/ files untouched
rm -f data/preprocessed/llm_predictions_cache.json
rm -f data/preprocessed/stage8_llm_retrieval_predictions.json

# Delete failed S4 submissions
rm -f outputs/submission_s4_*.csv

# Delete S6 submissions (venue approach failed)
rm -f outputs/submission_s6_*.csv

# Delete other S3 variants that didn't score well
rm -f outputs/submission_s3_enhanced.csv
rm -f outputs/submission_s3_distmatch.csv
rm -f outputs/submission_s3b_calibrated.csv
rm -f outputs/submission_s3b_distmatch_*.csv
rm -f outputs/submission_s3b_venue_corrected.csv
rm -f outputs/submission_s3b_ensemble_majority.csv

# Delete all other old submissions
rm -f outputs/submission_ensemble.csv
rm -f outputs/submission_stacking.csv
rm -f outputs/submission_multiseed.csv
rm -f outputs/submission_meta_ensemble.csv
rm -f outputs/submission_dense_ensemble.csv
rm -f outputs/submission_tfidf_ensemble.csv
rm -f outputs/submission_svc_*.csv
rm -f outputs/submission_lr_*.csv
rm -f outputs/submission_majority_*.csv
rm -f outputs/submission_single_best.csv
rm -f outputs/submission_top*.csv
rm -f outputs/submission_0.38.csv
rm -f outputs/submission_v*.csv
rm -f outputs/submission_stage*.csv
rm -f outputs/submission_blend_*.csv
rm -f outputs/submission.csv

# Delete other OOF files (keep only ft_* and s2new_oof.npz)
rm -f outputs/v4_oof_proba.npy outputs/v4_test_proba.npy
rm -f outputs/v5_oof_proba.npy outputs/v5_test_proba.npy
rm -f outputs/stage4_ordinal_*.npy
rm -f outputs/stage5_authors_*.npy
rm -f outputs/stage6_specter_*.npy
rm -f outputs/s4_oof_*.npy
rm -f outputs/s5_*.npy outputs/s5_*.npz

echo ""
echo "Cleanup complete!"
echo ""
echo "Kept files:"
echo "  - 4 raw data files"
echo "  - 5 preprocessed data files"
echo "  - 13 model artifact files"
echo "  - 6 winning submission files (0.71+)"
echo "  - 3 core scripts"
echo ""
echo "You can now regenerate S7 submissions using: python s7_inflation.py"
