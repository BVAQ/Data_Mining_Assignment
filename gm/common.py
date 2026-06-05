"""Shared bootstrap: force UTF-8 stdout on Windows + common imports/helpers."""
import sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np, pandas as pd, re
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

RAW = 'data/raw'
PRE = 'data/preprocessed'
SEED = 42

def qwk(a, b):
    return cohen_kappa_score(a, b, weights='quadratic')

def opt_th(y_true, y_raw, starts=None):
    """Find 4 cut points mapping continuous score -> 1..5 maximizing QWK."""
    if starts is None:
        starts = [np.array([1.5, 2.5, 3.5, 4.5]),
                  np.percentile(y_raw, [36, 57, 75, 89])]
    best, bs = None, -1
    for s in starts:
        r = minimize(lambda th: -qwk(y_true, np.digitize(y_raw, np.sort(th)) + 1),
                     s, method='Nelder-Mead',
                     options={'maxiter': 4000, 'xatol': 1e-4, 'fatol': 1e-6})
        th = np.sort(r.x)
        sc = qwk(y_true, np.digitize(y_raw, th) + 1)
        if sc > bs:
            bs, best = sc, th
    return best, bs

def apply_th(y_raw, th):
    return (np.digitize(y_raw, np.sort(th)) + 1).astype(int)

def norm_text(t):
    if pd.isna(t):
        return ''
    t = str(t).lower()
    t = t.replace('&amp;', ' and ').replace('&apos;', "'").replace('&quot;', ' ')
    t = re.sub(r'[^a-z0-9\s\+\-]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
