"""
Stage 2 (new) — strong stacked ensemble for ASP-relevance ordinal QWK.

Design (evidence-based from EDA):
  * Label is an ASP-relevance ordinal 1..5; title text is the dominant signal,
    venue is a strong prior, iclp is absent from test so we down-weight it.
  * Base models (diverse views), all producing OOF + test continuous scores:
      L1  word TF-IDF (1,2) Ridge regression
      L2  char TF-IDF (3,5) Ridge regression
      L3  word+char TF-IDF LogisticRegression -> expected value
      L4  SPECTER2 title+abs embeddings Ridge
      L5  BGE title+abs embeddings Ridge
      L6  embedding KNN retrieval (weighted neighbour label)
      L7  LightGBM regression on [meta + SVD(text) + emb-SVD]
  * CV-safe venue target encoding as a meta feature.
  * Meta learner: Ridge on stacked OOF -> continuous score.
  * QWK threshold optimization on OOF (global), applied to test.
All randomness seeded; OOF retrieval/target-encoding done strictly within folds.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import lightgbm as lgb

BASE = r"D:/HK252/Data Mining/Assignment"
RAW = os.path.join(BASE, "data", "raw")
PRE = os.path.join(BASE, "data", "preprocessed")
EMB = os.path.join(BASE, "outputs", "emb")
OUT = os.path.join(BASE, "outputs")
SEED = 42
N_SPLITS = 5

# --------------------------------------------------------------------------
def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")

def apply_th(p, th):
    th = sorted(th)
    out = np.ones(len(p), dtype=int)
    for i, t in enumerate(th):
        out[p > t] = i + 2
    return out

def opt_th(yt, p, init=(1.5, 2.5, 3.5, 4.5)):
    best = None
    for s in (init, (1.6, 2.4, 3.3, 4.2), (1.4, 2.6, 3.6, 4.4)):
        r = minimize(lambda th: -qwk(yt, apply_th(p, th)), s, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
        if best is None or -r.fun > best[1]:
            best = (r.x, -r.fun)
    return best[0]

# --------------------------------------------------------------------------
def load():
    tr = pd.read_csv(os.path.join(RAW, "train.csv"))
    pub = pd.read_csv(os.path.join(RAW, "public_test.csv"))
    pri = pd.read_csv(os.path.join(RAW, "private_test.csv"))
    for d in (tr, pub, pri):
        d["title"] = d["title"].fillna("")
        d["venue"] = d["venue"].fillna("na")
        d["authors"] = d["authors"].fillna("")
        d["doi"] = d["doi"].fillna("")
        d["year"] = pd.to_numeric(d["year"], errors="coerce").fillna(2020).astype(int)
    return tr, pub, pri

def get_abstracts(ids):
    ab = json.load(open(os.path.join(PRE, "stage2_abstracts_cache.json"), encoding="utf-8"))
    return [ab.get(str(i), "") if isinstance(ab.get(str(i), ""), str) else "" for i in ids]

# --------------------------------------------------------------------------
def main():
    tr, pub, pri = load()
    y = tr["Label"].values
    n_tr, n_pub, n_pri = len(tr), len(pub), len(pri)
    df = pd.concat([tr, pub, pri], ignore_index=True)
    n_all = len(df)
    abstracts = get_abstracts(df["id"].tolist())
    df["abs"] = abstracts
    df["ta"] = [f"{t}. {a}" if len(a) > 20 else t for t, a in zip(df["title"], df["abs"])]

    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, y))

    title = df["title"]
    ta = df["ta"]
    tr_slice = slice(0, n_tr)
    te_slice = slice(n_tr, n_all)

    base_oof = {}   # name -> oof array (len n_tr)
    base_test = {}  # name -> test array (len n_pub+n_pri)

    # ---- helper to run a sklearn regressor over folds with a feature builder ----
    def run_linear(name, build_fn, model_fn):
        """build_fn(idx_fit) -> (X_all_sparse_or_dense) using only idx_fit to fit vectorizer.
           We refit vectorizer per fold to avoid leakage."""
        oof = np.zeros(n_tr)
        test_acc = np.zeros(n_pub + n_pri)
        for trn, val in folds:
            Xtr, Xval, Xte = build_fn(trn, val)
            m = model_fn()
            m.fit(Xtr, y[trn])
            if hasattr(m, "predict_proba"):
                classes = m.classes_
                ev = lambda P: P @ classes
                oof[val] = ev(m.predict_proba(Xval))
                test_acc += ev(m.predict_proba(Xte))
            else:
                oof[val] = m.predict(Xval)
                test_acc += m.predict(Xte)
        base_oof[name] = oof
        base_test[name] = test_acc / N_SPLITS
        th = opt_th(y, oof)
        print(f"  {name:28s} OOF-QWK thresh={qwk(y, apply_th(oof, th)):.4f}", flush=True)

    # ---------- L1: word tfidf ridge ----------
    def b_word(trn, val):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        v.fit(title.iloc[trn])
        return v.transform(title.iloc[trn]), v.transform(title.iloc[val]), v.transform(title.iloc[te_slice])
    run_linear("word_ridge", b_word, lambda: Ridge(alpha=3.0))

    # ---------- L2: char tfidf ridge ----------
    def b_char(trn, val):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=80000)
        v.fit(title.iloc[trn])
        return v.transform(title.iloc[trn]), v.transform(title.iloc[val]), v.transform(title.iloc[te_slice])
    run_linear("char_ridge", b_char, lambda: Ridge(alpha=3.0))

    # ---------- L3: word+char logreg (expected value) ----------
    def b_wc(trn, val):
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=80000)
        vw.fit(title.iloc[trn]); vc.fit(title.iloc[trn])
        def tx(idx):
            return sp.hstack([vw.transform(title.iloc[idx]), vc.transform(title.iloc[idx])]).tocsr()
        return tx(trn), tx(val), tx(range(n_tr, n_all))
    run_linear("wc_logreg", b_wc, lambda: LogisticRegression(C=3.0, max_iter=2000))

    # ---------- embedding-based ----------
    def load_emb(tag, field):
        fn = os.path.join(EMB, f"{tag}_{field}.npy")
        return np.load(fn) if os.path.exists(fn) else None

    for tag in ("specter2", "bge"):
        E = load_emb(tag, "titleabs")
        if E is None:
            print(f"  [skip] {tag} embeddings not found", flush=True)
            continue
        Etr_all, Ete_all = E[tr_slice], E[te_slice]
        def b_emb(trn, val, Etr_all=Etr_all, Ete_all=Ete_all):
            return Etr_all[trn], Etr_all[val], Ete_all
        run_linear(f"{tag}_ridge", b_emb, lambda: Ridge(alpha=1.0))

        # KNN retrieval (cosine) — leakage-safe within fold
        name = f"{tag}_knn"
        oof = np.zeros(n_tr); test_acc = np.zeros(n_pub + n_pri)
        for trn, val in folds:
            Et = normalize(Etr_all[trn]); Ev = normalize(Etr_all[val]); Ete = normalize(Ete_all)
            ytrn = y[trn]
            sim = Ev @ Et.T
            for k in [10]:
                idx = np.argsort(-sim, axis=1)[:, :k]
                w = np.take_along_axis(sim, idx, axis=1)
                lab = ytrn[idx]
                oof[val] = (w * lab).sum(1) / (w.sum(1) + 1e-9)
            simte = Ete @ Et.T
            idx = np.argsort(-simte, axis=1)[:, :10]
            w = np.take_along_axis(simte, idx, axis=1); lab = ytrn[idx]
            test_acc += (w * lab).sum(1) / (w.sum(1) + 1e-9)
        base_oof[name] = oof; base_test[name] = test_acc / N_SPLITS
        print(f"  {name:28s} OOF-QWK thresh={qwk(y, apply_th(oof, opt_th(y, oof))):.4f}", flush=True)

    # ---------- L7: LightGBM regression on meta + SVD ----------
    # CV-safe venue target encoding + metadata + text-SVD + emb-SVD
    df["n_auth"] = df["authors"].apply(lambda s: 0 if not s else len(str(s).split(",")))
    df["title_len"] = df["title"].str.len()
    df["title_wc"] = df["title"].str.split().apply(len)
    df["has_doi10"] = df["doi"].str.startswith("10.").astype(int)
    df["yearn"] = df["year"] - 2016
    venues = sorted(df["venue"].unique())
    vmap = {v: i for i, v in enumerate(venues)}
    df["venue_id"] = df["venue"].map(vmap)
    kw = ["answer set", "asp", "logic program", "neural", "deep", "explain",
          "probabilist", "reasoning", "constraint", "ontolog", "argument", "planning"]
    for w in kw:
        df[f"kw_{w[:6]}"] = df["title"].str.lower().str.contains(w).astype(int)
    meta_cols = ["n_auth", "title_len", "title_wc", "has_doi10", "yearn", "venue_id"] + \
                [f"kw_{w[:6]}" for w in kw]

    # text SVD (fit on train only per global - acceptable, unsupervised) but to be safe fit per fold-ish:
    # use a single unsupervised SVD on all titles (no label leakage)
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit(title)
    Xw = vw.transform(title)
    svd = TruncatedSVD(64, random_state=SEED).fit(Xw)
    Tsvd = svd.transform(Xw)
    emb_specter = load_emb("specter2", "titleabs")
    if emb_specter is not None:
        esvd = TruncatedSVD(32, random_state=SEED).fit(emb_specter[tr_slice])
        Esvd = esvd.transform(emb_specter)
    else:
        Esvd = np.zeros((n_all, 0))

    Mall = np.hstack([df[meta_cols].values.astype(float), Tsvd, Esvd])
    Mtr, Mte = Mall[tr_slice], Mall[te_slice]

    # CV-safe venue mean-target encoding appended
    def venue_te(trn, val):
        gm = y[trn].mean()
        means = pd.Series(y[trn]).groupby(df["venue"].iloc[trn].values).mean()
        enc_val = df["venue"].iloc[val].map(means).fillna(gm).values.reshape(-1, 1)
        enc_te = df["venue"].iloc[te_slice].map(means).fillna(gm).values.reshape(-1, 1)
        enc_trn = df["venue"].iloc[trn].map(means).fillna(gm).values.reshape(-1, 1)
        return enc_trn, enc_val, enc_te

    oof = np.zeros(n_tr); test_acc = np.zeros(n_pub + n_pri)
    for trn, val in folds:
        et, ev, ete = venue_te(trn, val)
        Xt = np.hstack([Mtr[trn], et]); Xv = np.hstack([Mtr[val], ev]); Xte = np.hstack([Mte, ete])
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                              reg_lambda=1.0, random_state=SEED, verbose=-1)
        m.fit(Xt, y[trn])
        oof[val] = m.predict(Xv); test_acc += m.predict(Xte)
    base_oof["lgbm_meta"] = oof; base_test["lgbm_meta"] = test_acc / N_SPLITS
    print(f"  {'lgbm_meta':28s} OOF-QWK thresh={qwk(y, apply_th(oof, opt_th(y, oof))):.4f}", flush=True)

    # ---------- fine-tuned transformer OOF (strongest single view) ----------
    # Use single best seed per model (diversity > variance reduction in ensemble)
    for tag in ("scibert", "specter2ft", "deberta"):
        fo = os.path.join(OUT, f"ft_{tag}_oof.npy")
        ftt = os.path.join(OUT, f"ft_{tag}_test.npy")
        if os.path.exists(fo) and os.path.exists(ftt):
            base_oof[f"ft_{tag}"] = np.load(fo)
            base_test[f"ft_{tag}"] = np.load(ftt)
            print(f"  ft_{tag:25s} OOF-QWK thresh={qwk(y, apply_th(base_oof[f'ft_{tag}'], opt_th(y, base_oof[f'ft_{tag}']))):.4f}", flush=True)

    # ---------- META STACK ----------
    names = list(base_oof.keys())
    Z = np.column_stack([base_oof[n] for n in names])
    Zte = np.column_stack([base_test[n] for n in names])
    print("\nBase models:", names)
    # correlation
    print("OOF corr matrix:")
    print(np.round(np.corrcoef(Z.T), 2))

    meta_oof = np.zeros(n_tr); meta_te = np.zeros(n_pub + n_pri)
    for trn, val in folds:
        mm = Ridge(alpha=1.0).fit(Z[trn], y[trn])
        meta_oof[val] = mm.predict(Z[val])
    # final test: fit on all
    meta_final = Ridge(alpha=1.0).fit(Z, y)
    meta_te = meta_final.predict(Zte)

    th = opt_th(y, meta_oof)
    print(f"\n=== META STACK OOF-QWK thresh={qwk(y, apply_th(meta_oof, th)):.4f} (round={qwk(y, np.clip(np.round(meta_oof),1,5)):.4f}) ===")
    print("thresholds", np.round(th, 4))
    pred_lab = apply_th(meta_oof, th)
    print("OOF confusion:\n", confusion_matrix(y, pred_lab))

    # also a simple average blend for comparison
    avg_oof = Z.mean(1); avg_te = Zte.mean(1)
    tha = opt_th(y, avg_oof)
    print(f"Simple-avg blend OOF-QWK thresh={qwk(y, apply_th(avg_oof, tha)):.4f}")

    # ---------- save OOF/test artifacts ----------
    np.savez(os.path.join(OUT, "s2new_oof.npz"),
             names=np.array(names), Z=Z, Zte=Zte, y=y,
             meta_oof=meta_oof, meta_te=meta_te, th=th,
             n_pub=n_pub, n_pri=n_pri,
             ids_test=df["id"].iloc[te_slice].values)
    print("saved s2new_oof.npz")

    # ---------- write submission (meta stack) ----------
    test_lab = apply_th(meta_te, th)
    ids_test = df["id"].iloc[te_slice].values
    sub = pd.DataFrame({"id": ids_test, "Label": test_lab})
    sub.to_csv(os.path.join(OUT, "submission_s2new_stack.csv"), index=False)
    print("test pred dist:", dict(pd.Series(test_lab).value_counts().sort_index()))
    print("wrote submission_s2new_stack.csv")


if __name__ == "__main__":
    main()
