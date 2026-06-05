# Why gm_final / gm_ridge_global dropped to ~0.70, and the fix

## Verdict: the base ranking was fine; the CALIBRATION was wrong.

QWK(old_best, gm_final) = **0.906** — my predictions order the papers almost identically to the
proven 0.720 submission. The drop was **not** a modeling failure. It was distribution/threshold calibration.

### The evidence (LB vs predicted distribution)

| Submission | L1 | L2 | L3 | L4 | L5 | LB |
|---|---|---|---|---|---|---|
| s3b_ensemble (old best) | **0.409** | **0.174** | 0.159 | 0.094 | **0.163** | **0.720** |
| s3b_robust | 0.409 | 0.174 | 0.253 | 0.092 | 0.070 | 0.719 |
| gm_final (mine) | 0.366 | 0.267 | 0.191 | 0.084 | 0.092 | 0.707 |
| gm_ridge_global (mine) | 0.389 | 0.243 | 0.198 | 0.101 | 0.069 | 0.700 |

My submissions had **too few L1 (0.37 vs 0.41), too many L2 (0.27 vs 0.17), too few L5 (0.09 vs 0.16)**.
The crosstab vs the winner shows my high-end predictions sit systematically **one notch low**
(40 papers old=3/me=2, 41 old=4/me=3, 34 old=5/me=4).

### Root cause: two compounding calibration mistakes

1. **QWK-optimal thresholds over-hedge to the middle.** On OOF they look best, but they pile mass into
   L2/L3 (where the model is unsure), under-calling the tails. QWK on the *test* punishes that.
2. **Test-venue threshold tuning** (my "improvement") shrank the high tail further — the test-venue train
   subset is dominated by low-label cav/lics.

### The killer finding: OOF QWK is ANTI-correlated with LB for the calibration choice

```
Calibration of the SAME (my) ranking:        OOF QWK     LB
  optimal-threshold (shipped)                 0.6699      0.707
  train-marginal quantile                     0.6648       ~?
  winning-distribution quantile               0.6506      0.720  (old)
```
The distribution that looks **worst** on OOF scores **best** on the LB. Conclusion: **the test marginal is
NOT the train marginal** — it is the "winning" shape (more L1+L5, less L2). OOF can be trusted for *ranking*
but **must not** be used to pick the *calibration*; only the LB reveals the test marginal.

## The fix (separates the two decisions, each using the right signal)

- **Ranking:** keep my stack (OOF 0.668 ± 0.002 > old 0.659) — OOF is reliable here.
- **Calibration:** match the **LB-revealed test marginal** (L1≈0.41, L2≈0.17, L3≈0.16, L4≈0.09, L5≈0.16)
  via low-dimensional (4-parameter) quantile thresholds. This is principled calibration to the test
  distribution — NOT the per-row "tail inflation" that collapsed earlier (s8 at 0.475 L1 → 0.69).

## Recommended submissions (in `outputs/`)

| file | ranking | distribution | agree w/ old 0.720 | rationale |
|---|---|---|---|---|
| **submission_gm2_winning.csv** | my stack | winning (L1=.41) | 0.775 | **PRIMARY** — proven test marginal + my better ranking; best shot to beat 0.72 |
| submission_gm2_blend_winning.csv | blend(mine+old) | winning | 0.963 | **SAFE floor** ≈ reproduces 0.720, nudged by my models |
| submission_gm2_trainmarg.csv | my stack | train marginal | 0.718 | alt hypothesis (test≈train); lower L1, likely ~0.71 |

**Submit order:** `gm2_winning` (upside), then `gm2_blend_winning` (floor). If only one: `gm2_winning`.

## Realistic ceiling
Ranking is saturated at ~0.668 OOF (middle classes genuinely confusable from title+venue). With the correct
marginal, expect **~0.72–0.73** (old 0.659-ranking + winning dist = 0.720; my +0.009 ranking ≈ 0.725–0.73).
**0.75 is achievable only via the ±0.036 variance of a 298-sample QWK, not skill; 0.80 is not supported.**
