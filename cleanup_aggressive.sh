#!/bin/bash
# AGGRESSIVE CLEANUP - Keep only essentials for 0.71+ scores
# This removes ALL unnecessary files including old data files

cd "$(dirname "$0")"

echo "=== AGGRESSIVE CLEANUP ==="
echo "This will keep only 31 essential files"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# ============================================
# DELETE ALL REPORTS AND DOCS
# ============================================
rm -f *.md 2>/dev/null
rm -f DIAGNOSTIC_REPORT.md EXECUTIVE_SUMMARY.md README_STAGE2.md REPORT_BREAKTHROUGH.md
rm -f STAGE3_FINAL_REPORT.md SUBMISSION_SUMMARY.md CLEANUP_PLAN.md

# ============================================
# DELETE ALL SCRIPTS EXCEPT 3 CORE ONES
# ============================================
# Keep: s5_core.py, s7_inflation.py, build_dataset.py
# Delete everything else
for f in *.py; do
    if [[ "$f" != "s5_core.py" && "$f" != "s7_inflation.py" && "$f" != "build_dataset.py" ]]; then
        rm -f "$f"
    fi
done

# ============================================
# CLEAN DATA DIRECTORY
# ============================================
# Keep only: train.csv, public_test.csv, private_test.csv, Test_Submission.csv
cd data/raw
for f in *; do
    if [[ "$f" != "train.csv" && "$f" != "public_test.csv" && "$f" != "private_test.csv" && "$f" != "Test_Submission.csv" ]]; then
        rm -f "$f"
    fi
done
cd ../..

# Keep only: s5_*.parquet, s2_metadata_cache.json, stage2_abstracts_cache.json
cd data/preprocessed
for f in *; do
    if [[ "$f" != "s5_train.parquet" && "$f" != "s5_public.parquet" && "$f" != "s5_private.parquet" &&
          "$f" != "s2_metadata_cache.json" && "$f" != "stage2_abstracts_cache.json" ]]; then
        rm -f "$f"
    fi
done
cd ../..

# ============================================
# CLEAN OUTPUTS DIRECTORY
# ============================================
cd outputs

# Delete ALL subdirectories
rm -rf emb/ stage8_cache/ 2>/dev/null

# Keep only specific files
KEEP_FILES=(
    "s2new_oof.npz"
    "ft_scibert_oof.npy"
    "ft_scibert_test.npy"
    "ft_scibert_ids_test.npy"
    "ft_specter2ft_oof.npy"
    "ft_specter2ft_test.npy"
    "ft_specter2ft_ids_test.npy"
    "ft_scibert_s2_oof.npy"
    "ft_scibert_s2_test.npy"
    "ft_scibert_s2_ids_test.npy"
    "ft_specter2ft_s2_oof.npy"
    "ft_specter2ft_s2_test.npy"
    "ft_specter2ft_s2_ids_test.npy"
    "submission_s3b_ensemble_weighted.csv"
    "submission_s3b_robust.csv"
    "submission_s7_tuned_to_s3.csv"
    "submission_s7_breakthrough.csv"
    "submission_s7_aggressive.csv"
    "submission_s7_moderate.csv"
)

# Delete everything not in keep list
for f in *; do
    keep=0
    for keep_file in "${KEEP_FILES[@]}"; do
        if [[ "$f" == "$keep_file" ]]; then
            keep=1
            break
        fi
    done
    if [[ $keep -eq 0 ]]; then
        rm -f "$f"
    fi
done

cd ..

# ============================================
# DELETE OTHER DIRECTORIES
# ============================================
rm -rf catboost_info/ 2>/dev/null
rm -rf app/ 2>/dev/null

echo ""
echo "=== CLEANUP COMPLETE ==="
echo ""
echo "Remaining structure:"
echo "  data/raw/ (4 files)"
echo "  data/preprocessed/ (5 files)"
echo "  outputs/ (19 files)"
echo "  Root: 3 Python scripts"
echo ""
echo "Total: 31 essential files"
echo ""
echo "To regenerate S7 submissions: python s7_inflation.py"
