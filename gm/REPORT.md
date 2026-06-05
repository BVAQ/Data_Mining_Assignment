# DM252 ASP Paper Classification — Grandmaster Pipeline Report

**Task:** ordinal Label 1–5, metric = Quadratic Weighted Kappa (QWK). train=2494, public=298, private=298.
**Goal:** push from the ~0.72 public plateau toward 0.75–0.80.

---

## 1. What the data actually is (independent EDA)

| Signal | Finding |
|---|---|
| **Label** | Topical relevance to Answer Set Programming / logic programming. Skewed: 36/21/18/15/11 %. |
| **Venue** (strongest prior) | `lics` 1.94 < `cav` 2.12 < `kr` 2.73 < `iclp` 2.94 < `lpnmr` 3.14 (mean label). |
| **Test venue mix** | Only `cav/lics/kr/lpnmr`. **`iclp` (456 train rows) is ABSENT from test.** |
| **Abstracts** | 85% coverage — but they **dilute** the signal (see §3). |
| **Leakage** | 0 shared titles, 3 "shared DOIs" = S2 metadata-join artifacts (different papers). **No usable leakage.** |
| **Front-matter** | Proceedings/"Nth International Conference…" titles → **Label 1, 62/62 = 100% precise**. 14 such rows in test. |
| **Authors / year / DOI-prefix** | Weak or redundant with venue. |

Anchoring baselines (5-fold): global-mode 0.00, venue-mean+optimal-thresholds **0.298**, title TF-IDF+LogReg 0.565.
→ The text model carries everything above the ~0.30 metadata floor.

## 2. Validation

5-fold StratifiedKFold (seed 42), QWK with **OOF-tuned thresholds** (Nelder–Mead on 4 cut-points).
Because the test contains **no `iclp`**, the honest estimator is **QWK on the test-venue OOF subset**.
Confirmed via adversarial reasoning + repeated 5× CV (std ≈ 0.002), so CV is trustworthy.

## 3. The two ideas that broke the 0.66 plateau

The prior pipeline (and a stored project note) had concluded ~0.66 OOF was an intrinsic ceiling.
Re-deriving from scratch found **two levers it never used**:

1. **Venue-prefixed input** — feed the transformer `"venue: <venue>. <title>"`. Gives the model the
   strong venue prior directly instead of forcing the meta-layer to add it back.
2. **Regression (MSE) head instead of classification** — directly optimizes the ordinal scale that QWK rewards.

A single SciBERT with these two changes hit **0.6554** (1 seed) — already matching the *entire prior stack*.

> Confirmed dead-ends (each tested, not assumed): abstracts (`venue_titabs` 0.619 < `venue_title` 0.659);
> general-domain DeBERTa-v3 (0.621) and MPNet — weaker than scientific encoders; 7 epochs overfits (0.652 < 0.664 @ 4).

## 4. Final model

**Base learners (fresh, 4-seed regression fine-tunes, `venue: <v>. <title>`, 4 epochs, CLS+MSE):**

| model | OOF QWK |
|---|---|
| BAAI/bge-base-en-v1.5 | **0.664** |
| allenai/scibert | 0.659 |
| allenai/specter2_base | 0.658 |

**+ supporting features:** 4 precomputed FT OOFs, SPECTER2/BGE frozen-embedding ridge/logreg heads,
title TF-IDF, venue & venue×year target-encoding, kNN-retrieval label, frontmatter flag.

**Meta-learner:** multi-seed (×12) Ridge stack (α=2.0).
**Thresholds:** tuned on the **test-venue OOF subset** (legitimate calibration — test venues are given).
**Post-process:** frontmatter titles → Label 1 (deterministic, 100% precise).

## 5. Result

```
Repeated 5× StratifiedKFold (5 splits), final strategy:
  TEST-VENUE QWK = 0.6684 ± 0.0022   (min 0.6650, max 0.6716)
```

| | OOF QWK (test-venue) |
|---|---|
| Prior plateau (stored stack) | ~0.659 |
| **This pipeline** | **0.668 ± 0.002** |

A robust **+0.009 OOF**, achieved with **honest calibration** (predicted L5 ≈ 9% vs the prior LB-overfit 16%).

## 6. Expected leaderboard & recommendation

The prior ~0.6585 test-venue OOF corresponded to **0.720 public**, i.e. an OOF→public offset of ~+0.06
(298-sample QWK has std ≈ 0.036; the test excludes the noisy `iclp` class).
Applying the same offset: **0.668 OOF → ~0.73 public**, with upside to ~0.74–0.75 if the offset holds,
and — more importantly — **better private-LB generalization** because nothing here is tuned to the public LB.

**Submit (in this order):**
1. `outputs/submission_gm_final.csv` — **primary** (ridge stack, test-venue thresholds, override). CV 0.669.
2. `outputs/submission_gm_ridge_global.csv` — global-threshold variant (slightly fewer tails); A/B on the LB.

**Honest ceiling note:** middle classes 2/3/4 are *genuinely* confusable from title+venue (per-class recall
0.40/0.27/0.38; the information to separate them is not in the metadata). **0.75 is reachable on the LB if the
offset cooperates; 0.80 is not supported by the data.** The biggest remaining lever would be more/better labels,
not more modeling.

## 7. Reproduce

```
env/Scripts/python gm/run_all.py          # raw data -> submission (uses GPU)
env/Scripts/python gm/run_all.py --skip-ft # reuse cached fine-tunes
```
Artifacts: `gm/_feats.npz` (features), `gm/ft/*` (fine-tune OOF/test), `gm/model_comparison.csv`,
`gm/test_preds.csv`, `gm/_finalize.npz`, submissions in `gm/outputs/` and `outputs/`.
