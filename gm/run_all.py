"""
END-TO-END PIPELINE RUNNER  (DM252 ASP paper ordinal classification, metric=QWK)
================================================================================
Reproducible from raw data + cached metadata/abstracts to final submission.
Fixed seed (42) throughout. GPU used for transformer fine-tuning.

Run:  env/Scripts/python gm/run_all.py            # full pipeline
      env/Scripts/python gm/run_all.py --skip-ft  # reuse existing fine-tunes

Stages (each is an importable script in gm/):
  0. eda.py / eda2.py / eda3.py / rules.py   -- analysis & sanity (optional to re-run)
  1. emb_extract.py     -- dense embeddings (specter2/bge/scibert/mpnet) title & title+abs
  2. features.py        -- assemble aligned OOF(2494) + TEST(596) base-feature matrix -> gm/_feats.npz
  3. finetune.py        -- fresh regression fine-tunes (venue-prefixed title), 5-fold OOF + test
                           Strong set (4 seeds each): bge / scibert / specter2  venue_title
  4. finalize.py        -- multi-seed Ridge stack; thresholds calibrated to test-venue composition;
                           frontmatter->L1 override; writes submissions + comparison table
  5. robust_final.py    -- repeated 5x StratifiedKFold honest mean+/-std

Key empirical findings driving the design (all from CV, never the LB):
  * Venue is a strong ordinal prior (lics<cav<kr<iclp<lpnmr). iclp absent from test.
  * TITLE is the signal; ABSTRACT dilutes it (true for TF-IDF, frozen emb, AND fine-tuning).
  * Prepending "venue: <v>." to the title + a REGRESSION (MSE) head beats classification heads.
  * Per-model OOF QWK ~0.655-0.664; multi-seed + stacking -> ~0.668 (test-venue, +/-0.002).
  * Frontmatter/proceedings titles -> Label 1 (train precision 62/62 = 100%).
  * This lifts the prior ~0.659 plateau to ~0.668 honest OOF, calibrated (not LB-overfit).
"""
import sys, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, 'env', 'Scripts', 'python')

def run(script, *args):
    cmd = [PY, os.path.join(HERE, script), *args]
    print('\n' + '='*70 + f'\n>>> {script} {" ".join(args)}\n' + '='*70, flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == '__main__':
    skip_ft = '--skip-ft' in sys.argv
    run('emb_extract.py')
    run('features.py')
    if not skip_ft:
        for m in ['bge', 'scibert', 'specter2']:
            run('finetune.py', m, 'venue_title', '4', '0,1,2,3')
    run('finalize.py')
    run('robust_final.py')
    print('\nDONE. Recommended submission: gm/outputs/submission_gm_final.csv')
