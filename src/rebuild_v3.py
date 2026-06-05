"""
REBUILD v3 (FAST): Streamlined pipeline using existing predictions
==================================================================
Core approach:
  1. Use GM _feats.npz features (16 features, includes fine-tuned models)
  2. Add GM extra fine-tunes (ft2_*)
  3. Add S2 meta OOF
  4. Add fresh TF-IDF Ridge (fast, no dense matrices)
  5. Multi-seed Ridge stacking  
  6. Multiple calibration strategies
  7. Frontmatter override
"""
import os, sys, warnings, re, glob, time
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

BASE = r"D:/HK252/Data Mining/Assignment"
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "outputs")
N_SPLITS = 5
TEST_VENUES = {'cav', 'lics', 'kr', 'lpnmr'}

t0 = time.time()

def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")

def apply_th(p, th):
    th = sorted(th)
    out = np.ones(len(p), dtype=int)
    for i, t in enumerate(th):
        out[p > t] = i + 2
    return out

def coord_ascent_th(yt, p, n_round=5, grid_size=600):
    lo, hi = p.min() - 0.1, p.max() + 0.1
    best_th, best_score = None, -1
    for init_q in [[0.30, 0.50, 0.70, 0.85], [0.35, 0.55, 0.73, 0.88],
                   [0.36, 0.57, 0.74, 0.89], [0.38, 0.60, 0.76, 0.90]]:
        th = list(np.quantile(p, init_q))
        grid = np.linspace(lo, hi, grid_size)
        best = qwk(yt, apply_th(p, th))
        for _ in range(n_round):
            improved = False
            for i in range(4):
                for g in grid:
                    cand = th.copy(); cand[i] = g
                    cs = sorted(cand)
                    if not (cs[0] < cs[1] < cs[2] < cs[3]):
                        continue
                    s = qwk(yt, apply_th(p, cs))
                    if s > best:
                        best, th = s, cs; improved = True
            if not improved: break
        if best > best_score:
            best_score, best_th = best, sorted(th)
    return best_th, best_score

def quantile_calibrate(test_cont, target_dist):
    cum = np.cumsum(target_dist)
    ranks = pd.Series(test_cont).rank(pct=True).values
    labels = np.ones(len(ranks), dtype=int)
    for i in range(len(ranks)):
        labels[i] = 1 + int(np.searchsorted(cum, ranks[i], side="right"))
    return np.clip(labels, 1, 5)

def detect_frontmatter(titles):
    patterns = [r'proceedings?\s+of\s+the', r'foreword', r'preface',
                r'\d+(st|nd|rd|th)\s+international\s+(conference|workshop|symposium)',
                r'editorial', r'table\s+of\s+contents']
    return titles.fillna('').str.lower().str.contains('|'.join(patterns), regex=True, na=False).values

print("=" * 90)
print("REBUILD v3 (FAST): STREAMLINED PIPELINE")
print("=" * 90)

# Load data
tr = pd.read_csv(os.path.join(RAW, "train.csv"))
pub = pd.read_csv(os.path.join(RAW, "public_test.csv"))
pri = pd.read_csv(os.path.join(RAW, "private_test.csv"))
sample_sub = pd.read_csv(os.path.join(RAW, "Test_Submission.csv"))
df_test = pd.concat([pub, pri], ignore_index=True)

for d in [tr, df_test]:
    d['title'] = d['title'].fillna('')
    d['venue'] = d['venue'].fillna('unknown')

y = tr['Label'].values
tv_mask = tr['venue'].isin(TEST_VENUES).values
fm_train = detect_frontmatter(tr['title'])
fm_test = detect_frontmatter(df_test['title'])

print(f"Train: {len(tr)}, Test: {len(df_test)}, Test-venue: {tv_mask.sum()}")
print(f"Frontmatter: train={fm_train.sum()}, test={fm_test.sum()}")

# ============================================================================
# LOAD EXISTING PREDICTIONS
# ============================================================================
print(f"\n[{time.time()-t0:.0f}s] Loading existing predictions...")

# GM features
gm = np.load(os.path.join(BASE, 'gm', '_feats.npz'), allow_pickle=True)
X_oof = gm['X_oof'].copy()  # (2494, 16)
X_te = gm['X_test'].copy()  # (596, 16)
names = list(gm['names'])
print(f"  GM features: {X_oof.shape[1]} ({names})")

# Extra GM fine-tunes
EXCLUDE_FT = {'ft2_scibert_venue_titabs', 'ft2_deberta_venue_title'}
for op in sorted(glob.glob(os.path.join(BASE, 'gm', 'ft', 'ft2_*_oof.npy'))):
    tag = os.path.basename(op).replace('_oof.npy', '')
    tp = os.path.join(BASE, 'gm', 'ft', f'{tag}_test.npy')
    if tag in EXCLUDE_FT or not os.path.exists(tp):
        continue
    oof = np.load(op)
    te = np.load(tp)
    X_oof = np.column_stack([X_oof, oof])
    X_te = np.column_stack([X_te, te])
    names.append(tag)

# S2 meta
s2 = np.load(os.path.join(OUT, 's2new_oof.npz'), allow_pickle=True)
X_oof = np.column_stack([X_oof, s2['meta_oof']])
X_te = np.column_stack([X_te, s2['meta_te']])
names.append('s2_meta')

# S2 individual base models that have test predictions
# Actually the S2 Z matrix has individual model predictions but only OOF
# Let's add them for OOF but we can't use them for test
# Instead, just use S2 meta (which incorporates all S2 base models)

print(f"  Total features: {X_oof.shape[1]}")
print(f"  Names: {names}")

# ============================================================================
# ADD FRESH TF-IDF FEATURES (fast)
# ============================================================================
print(f"\n[{time.time()-t0:.0f}s] Building fresh TF-IDF features...")

folds_42 = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=42).split(tr, y))

# Word TF-IDF Ridge
for alpha in [3.0]:
    oof_w = np.zeros(len(tr))
    te_w = np.zeros(len(df_test))
    for trn_idx, val_idx in folds_42:
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=50000, sublinear_tf=True)
        vec.fit(tr['title'].iloc[trn_idx])
        m = Ridge(alpha=alpha).fit(vec.transform(tr['title'].iloc[trn_idx]), y[trn_idx])
        oof_w[val_idx] = m.predict(vec.transform(tr['title'].iloc[val_idx]))
        te_w += m.predict(vec.transform(df_test['title'])) / N_SPLITS
    X_oof = np.column_stack([X_oof, oof_w])
    X_te = np.column_stack([X_te, te_w])
    names.append(f'tfidf_word_a{alpha}')
    th, sc = coord_ascent_th(y, oof_w)
    print(f"  tfidf_word_a{alpha}: QWK={sc:.4f}")

# Char TF-IDF Ridge
for alpha in [3.0]:
    oof_c = np.zeros(len(tr))
    te_c = np.zeros(len(df_test))
    for trn_idx, val_idx in folds_42:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=3, 
                              max_features=50000, sublinear_tf=True)
        vec.fit(tr['title'].iloc[trn_idx])
        m = Ridge(alpha=alpha).fit(vec.transform(tr['title'].iloc[trn_idx]), y[trn_idx])
        oof_c[val_idx] = m.predict(vec.transform(tr['title'].iloc[val_idx]))
        te_c += m.predict(vec.transform(df_test['title'])) / N_SPLITS
    X_oof = np.column_stack([X_oof, oof_c])
    X_te = np.column_stack([X_te, te_c])
    names.append(f'tfidf_char_a{alpha}')
    th, sc = coord_ascent_th(y, oof_c)
    print(f"  tfidf_char_a{alpha}: QWK={sc:.4f}")

# Combined word+char TF-IDF Ridge
oof_wc = np.zeros(len(tr))
te_wc = np.zeros(len(df_test))
for trn_idx, val_idx in folds_42:
    vec_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=60000, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=3, max_features=60000, sublinear_tf=True)
    vec_w.fit(tr['title'].iloc[trn_idx])
    vec_c.fit(tr['title'].iloc[trn_idx])
    X_trn = sp.hstack([vec_w.transform(tr['title'].iloc[trn_idx]), vec_c.transform(tr['title'].iloc[trn_idx])])
    X_val = sp.hstack([vec_w.transform(tr['title'].iloc[val_idx]), vec_c.transform(tr['title'].iloc[val_idx])])
    X_t = sp.hstack([vec_w.transform(df_test['title']), vec_c.transform(df_test['title'])])
    m = Ridge(alpha=3.0).fit(X_trn, y[trn_idx])
    oof_wc[val_idx] = m.predict(X_val)
    te_wc += m.predict(X_t) / N_SPLITS
X_oof = np.column_stack([X_oof, oof_wc])
X_te = np.column_stack([X_te, te_wc])
names.append('tfidf_combined')
th, sc = coord_ascent_th(y, oof_wc)
print(f"  tfidf_combined: QWK={sc:.4f}")

print(f"\n[{time.time()-t0:.0f}s] Final feature matrix: {X_oof.shape}")

# Replace NaN/inf
X_oof = np.nan_to_num(X_oof, nan=0, posinf=0, neginf=0)
X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

# ============================================================================
# FEATURE ABLATION (quick)
# ============================================================================
print(f"\n{'='*90}")
print("INDIVIDUAL FEATURE QWK (with coord-ascent thresholds)")
print(f"{'='*90}")

for i, name in enumerate(names):
    th, sc = coord_ascent_th(y, X_oof[:, i])
    th_tv, sc_tv = coord_ascent_th(y[tv_mask], X_oof[tv_mask, i])
    print(f"  {name:30s}: all={sc:.4f}, tv={sc_tv:.4f}")

# ============================================================================
# MULTI-SEED STACKING
# ============================================================================
print(f"\n{'='*90}")
print("MULTI-SEED RIDGE STACKING")
print(f"{'='*90}")

configs = []

for alpha in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    seed_oofs = []
    seed_tes = []
    seed_qwks_all = []
    seed_qwks_tv = []
    
    for seed in [42, 123, 456, 789, 1234]:
        folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(tr, y))
        
        oof = np.zeros(len(tr))
        te = np.zeros(len(df_test))
        
        for trn_idx, val_idx in folds:
            sc = StandardScaler().fit(X_oof[trn_idx])
            m = Ridge(alpha=alpha, random_state=seed).fit(
                sc.transform(X_oof[trn_idx]), y[trn_idx])
            oof[val_idx] = m.predict(sc.transform(X_oof[val_idx]))
            te += m.predict(sc.transform(X_te)) / N_SPLITS
        
        th, q_all = coord_ascent_th(y, oof)
        th_tv, q_tv = coord_ascent_th(y[tv_mask], oof[tv_mask])
        
        seed_oofs.append(oof)
        seed_tes.append(te)
        seed_qwks_all.append(q_all)
        seed_qwks_tv.append(q_tv)
    
    # Average across seeds
    avg_oof = np.mean(seed_oofs, axis=0)
    avg_te = np.mean(seed_tes, axis=0)
    
    th_avg, q_avg = coord_ascent_th(y, avg_oof)
    th_tv_avg, q_tv_avg = coord_ascent_th(y[tv_mask], avg_oof[tv_mask])
    
    print(f"\n  alpha={alpha:5.1f}:")
    print(f"    per-seed: all={np.mean(seed_qwks_all):.4f}±{np.std(seed_qwks_all):.4f}, tv={np.mean(seed_qwks_tv):.4f}±{np.std(seed_qwks_tv):.4f}")
    print(f"    avg-oof:  all={q_avg:.4f}, tv={q_tv_avg:.4f}")
    
    configs.append({
        'alpha': alpha,
        'mean_all': np.mean(seed_qwks_all), 'std_all': np.std(seed_qwks_all),
        'mean_tv': np.mean(seed_qwks_tv), 'std_tv': np.std(seed_qwks_tv),
        'q_avg_all': q_avg, 'q_avg_tv': q_tv_avg,
        'avg_oof': avg_oof, 'avg_te': avg_te,
        'th': th_avg, 'th_tv': th_tv_avg,
        'fold_qwks_all': seed_qwks_all, 'fold_qwks_tv': seed_qwks_tv,
    })

# Sort by avg-oof TV QWK
configs.sort(key=lambda x: x['q_avg_tv'], reverse=True)
best = configs[0]
print(f"\nBEST: alpha={best['alpha']}, TV QWK={best['q_avg_tv']:.4f}")

# ============================================================================
# COMPARE WITH BASELINES
# ============================================================================
print(f"\n{'='*90}")
print("COMPARISON WITH BASELINES")
print(f"{'='*90}")

# S2 meta
th_s2, sc_s2 = coord_ascent_th(y, s2['meta_oof'])
th_s2_tv, sc_s2_tv = coord_ascent_th(y[tv_mask], s2['meta_oof'][tv_mask])
print(f"  S2 META:      all={sc_s2:.4f}, tv={sc_s2_tv:.4f}")

# GM finalize (from _finalize.npz)
gm_fin = np.load(os.path.join(BASE, 'gm', '_finalize.npz'), allow_pickle=True)
th_gm, sc_gm = coord_ascent_th(y, gm_fin['oof'])
th_gm_tv, sc_gm_tv = coord_ascent_th(y[tv_mask], gm_fin['oof'][tv_mask])
print(f"  GM FINALIZE:  all={sc_gm:.4f}, tv={sc_gm_tv:.4f}")

print(f"  REBUILD v3:   all={best['q_avg_all']:.4f}, tv={best['q_avg_tv']:.4f}")

diff_s2 = best['q_avg_tv'] - sc_s2_tv
diff_gm = best['q_avg_tv'] - sc_gm_tv
print(f"\n  Improvement over S2: +{diff_s2:.4f}")
print(f"  Improvement over GM: +{diff_gm:.4f}")

# ============================================================================
# CALIBRATION STRATEGIES
# ============================================================================
print(f"\n{'='*90}")
print("CALIBRATION STRATEGIES")
print(f"{'='*90}")

avg_oof = best['avg_oof']
avg_te = best['avg_te']

# Define distributions
train_dist = np.bincount(y.astype(int), minlength=6)[1:] / len(y)
tv_dist = np.bincount(y[tv_mask].astype(int), minlength=6)[1:] / tv_mask.sum()
winning_dist = np.array([0.409, 0.174, 0.159, 0.094, 0.163])

strategies = {}

# 1. OOF-optimal (all)
pred = apply_th(avg_te, best['th']); pred[fm_test] = 1
strategies['opt_all'] = pred

# 2. OOF-optimal (test-venue)
pred = apply_th(avg_te, best['th_tv']); pred[fm_test] = 1
strategies['opt_tv'] = pred

# 3. Train marginal quantile
pred = quantile_calibrate(avg_te, train_dist); pred[fm_test] = 1
strategies['train_q'] = pred

# 4. TV marginal quantile
pred = quantile_calibrate(avg_te, tv_dist); pred[fm_test] = 1
strategies['tv_q'] = pred

# 5. Winning distribution
pred = quantile_calibrate(avg_te, winning_dist); pred[fm_test] = 1
strategies['winning_q'] = pred

# 6. Multiple blends
for alpha_blend in [0.3, 0.5, 0.7]:
    blend = alpha_blend * winning_dist + (1 - alpha_blend) * tv_dist
    blend = blend / blend.sum()
    pred = quantile_calibrate(avg_te, blend); pred[fm_test] = 1
    strategies[f'blend_{alpha_blend}'] = pred

# Evaluate calibrations on OOF
print("\n--- OOF QWK for each calibration ---")
cal_dists = {
    'opt_all': ('th', best['th']),
    'opt_tv': ('th', best['th_tv']),
    'train_q': ('dist', train_dist),
    'tv_q': ('dist', tv_dist),
    'winning_q': ('dist', winning_dist),
}
for cname, (ctype, cparam) in cal_dists.items():
    if ctype == 'th':
        oof_lab = apply_th(avg_oof, cparam)
    else:
        oof_lab = quantile_calibrate(avg_oof, cparam)
    oof_lab[fm_train] = 1
    q_all = qwk(y, oof_lab)
    q_tv = qwk(y[tv_mask], oof_lab[tv_mask])
    print(f"  {cname:15s}: OOF_all={q_all:.4f}, OOF_tv={q_tv:.4f}")

# ============================================================================
# GENERATE SUBMISSIONS
# ============================================================================
print(f"\n{'='*90}")
print("GENERATING SUBMISSIONS")
print(f"{'='*90}")

# Load prev best for comparison
prev_path = os.path.join(OUT, 'submission_s3b_ensemble_weighted.csv')
prev_best = pd.read_csv(prev_path) if os.path.exists(prev_path) else None

for name, preds in strategies.items():
    fname = f'submission_v3_{name}.csv'
    sub = pd.DataFrame({'id': sample_sub['id'], 'Label': preds.astype(int)})
    
    assert sub.shape[0] == 596
    assert (sub['id'] == sample_sub['id']).all()
    assert sub['Label'].between(1, 5).all()
    
    dist = sub['Label'].value_counts(normalize=True).sort_index()
    mean_lab = sub['Label'].mean()
    
    info = f"  {fname:40s} mean={mean_lab:.3f}"
    if prev_best is not None:
        merged = sub.merge(prev_best, on='id', suffixes=('_new', '_old'))
        agree = (merged['Label_new'] == merged['Label_old']).mean()
        kappa = qwk(merged['Label_old'], merged['Label_new'])
        info += f" agree={agree:.3f} QWK_prev={kappa:.3f}"
    
    sub.to_csv(os.path.join(OUT, fname), index=False)
    print(info)
    print(f"    dist={dict(dist.round(3))}")

# Save OOF
np.savez(os.path.join(OUT, 'rebuild_v3_oof.npz'),
         oof=avg_oof, test=avg_te, y=y,
         fm_train=fm_train, fm_test=fm_test, tv_mask=tv_mask,
         sub_ids=sample_sub['id'].values, feature_names=np.array(names),
         alpha=best['alpha'], th=best['th'], th_tv=best['th_tv'])

# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================
print(f"\n{'='*90}")
print("MODEL COMPARISON TABLE")
print(f"{'='*90}")
print(f"{'Strategy':30s} {'QWK_all':>8s} {'QWK_tv':>8s} {'Std_tv':>8s}")
print("-" * 58)
print(f"{'S2 META (baseline)':30s} {sc_s2:8.4f} {sc_s2_tv:8.4f} {'N/A':>8s}")
print(f"{'GM FINALIZE':30s} {sc_gm:8.4f} {sc_gm_tv:8.4f} {'N/A':>8s}")
for cfg in configs[:5]:
    print(f"{'Ridge alpha='+str(cfg['alpha']):30s} {cfg['q_avg_all']:8.4f} {cfg['q_avg_tv']:8.4f} {cfg['std_tv']:8.4f}")

elapsed = time.time() - t0
print(f"\n{'='*90}")
print(f"COMPLETE in {elapsed:.0f}s")
print(f"{'='*90}")
print(f"""
SUBMISSION PRIORITY:
  1. submission_v3_winning_q.csv  — Proven distribution + improved ranking
  2. submission_v3_blend_0.5.csv  — 50/50 blend of winning + TV marginal
  3. submission_v3_opt_tv.csv     — OOF-optimal on test-venue subset
  
RATIONALE:
  - Ranking improved from S2 META ({sc_s2_tv:.4f} tv) to rebuild ({best['q_avg_tv']:.4f} tv)
  - Improvement of +{diff_s2:.4f} in test-venue OOF QWK
  - With winning distribution, expected LB: ~0.72-0.73
  - With improved ranking, possible to exceed 0.72
""")
