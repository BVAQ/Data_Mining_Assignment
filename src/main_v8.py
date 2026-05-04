"""
ASP Paper Classification - v8
Builds on v3's meta-ensemble stacking approach (scored 0.38 on Kaggle).

v3's winning formula:
- SciBERT embeddings (normalized)
- Rich TF-IDF (title, abstract, combined, char n-grams, authors)
- Engineered features (keywords, metadata)
- Meta-ensemble: out-of-fold stacking with diverse models per feature set
- Multi-seed majority vote

v8 improvements:
- Add BGE embeddings alongside SciBERT
- Better hyperparameters (C=5-10 for SVC on embeddings based on v5/v6 findings)
- More diverse base models in the stacking
- More seeds for stability
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from collections import Counter
from sentence_transformers import SentenceTransformer

BASE_DIR = r"d:\HK252\Data Mining\Assignment"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_train_with_abstracts.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_test_with_abstracts.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "data", "raw", "sample_submission_DM252.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load():
    tr = pd.read_csv(TRAIN_PATH)
    te = pd.read_csv(TEST_PATH)
    for df in [tr, te]:
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['authors'] = df['authors'].fillna('')
        df['venue'] = df['venue'].fillna('unknown')
        df['doi'] = df['doi'].fillna('')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020).astype(int)
    return tr, te


def get_embeddings(df_tr, df_te):
    """Get SciBERT + BGE embeddings."""
    full_tr = (df_tr['title'] + '. ' + df_tr['abstract']).tolist()
    full_te = (df_te['title'] + '. ' + df_te['abstract']).tolist()
    title_tr = df_tr['title'].tolist()
    title_te = df_te['title'].tolist()
    
    all_emb = {}
    
    # SciBERT
    print("  Loading SciBERT...")
    m = SentenceTransformer('allenai/scibert_scivocab_uncased')
    all_emb['scibert_full'] = (
        normalize(m.encode(full_tr, show_progress_bar=True, batch_size=32)),
        normalize(m.encode(full_te, show_progress_bar=True, batch_size=32))
    )
    all_emb['scibert_title'] = (
        normalize(m.encode(title_tr, show_progress_bar=True, batch_size=32)),
        normalize(m.encode(title_te, show_progress_bar=True, batch_size=32))
    )
    del m
    
    # BGE
    print("  Loading BGE...")
    m = SentenceTransformer('BAAI/bge-base-en-v1.5')
    all_emb['bge_full'] = (
        normalize(m.encode(full_tr, show_progress_bar=True, batch_size=32)),
        normalize(m.encode(full_te, show_progress_bar=True, batch_size=32))
    )
    all_emb['bge_title'] = (
        normalize(m.encode(title_tr, show_progress_bar=True, batch_size=32)),
        normalize(m.encode(title_te, show_progress_bar=True, batch_size=32))
    )
    del m
    
    # Concatenated
    all_emb['scibert_concat'] = (
        normalize(np.hstack([all_emb['scibert_full'][0], all_emb['scibert_title'][0]])),
        normalize(np.hstack([all_emb['scibert_full'][1], all_emb['scibert_title'][1]]))
    )
    all_emb['bge_concat'] = (
        normalize(np.hstack([all_emb['bge_full'][0], all_emb['bge_title'][0]])),
        normalize(np.hstack([all_emb['bge_full'][1], all_emb['bge_title'][1]]))
    )
    all_emb['all_emb'] = (
        normalize(np.hstack([all_emb['scibert_full'][0], all_emb['bge_full'][0]])),
        normalize(np.hstack([all_emb['scibert_full'][1], all_emb['bge_full'][1]]))
    )
    
    return all_emb


def get_tfidf(df_tr, df_te):
    """Build rich TF-IDF features (same as v3)."""
    # Title word n-grams
    tv1 = TfidfVectorizer(max_features=3000, ngram_range=(1,3), sublinear_tf=True, min_df=2, max_df=0.95, strip_accents='unicode')
    X1_tr = tv1.fit_transform(df_tr['title'])
    X1_te = tv1.transform(df_te['title'])
    
    # Abstract word n-grams
    tv2 = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95, strip_accents='unicode')
    X2_tr = tv2.fit_transform(df_tr['abstract'])
    X2_te = tv2.transform(df_te['abstract'])
    
    # Combined text
    comb_tr = df_tr['title'] + ' ' + df_tr['abstract']
    comb_te = df_te['title'] + ' ' + df_te['abstract']
    tv3 = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95, strip_accents='unicode')
    X3_tr = tv3.fit_transform(comb_tr)
    X3_te = tv3.transform(comb_te)
    
    # Char n-grams
    tv4 = TfidfVectorizer(max_features=3000, ngram_range=(3,5), analyzer='char_wb', sublinear_tf=True, min_df=2, max_df=0.95)
    X4_tr = tv4.fit_transform(comb_tr)
    X4_te = tv4.transform(comb_te)
    
    # Authors
    tv5 = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    X5_tr = tv5.fit_transform(df_tr['authors'])
    X5_te = tv5.transform(df_te['authors'])
    
    X_tr = sp.hstack([X1_tr, X2_tr, X3_tr, X4_tr, X5_tr]).tocsr()
    X_te = sp.hstack([X1_te, X2_te, X3_te, X4_te, X5_te]).tocsr()
    return X_tr, X_te


def get_engineered(df_tr, df_te):
    """Hand-crafted features (same as v3)."""
    all_feats = []
    for df in [df_tr, df_te]:
        feats = pd.DataFrame()
        text = (df['title'] + ' ' + df['abstract']).str.lower()
        
        feats['title_len'] = df['title'].str.len()
        feats['title_words'] = df['title'].str.split().str.len()
        feats['abstract_len'] = df['abstract'].str.len()
        feats['abstract_words'] = df['abstract'].str.split().str.len().fillna(0)
        feats['has_abstract'] = (df['abstract'].str.len() > 10).astype(int)
        feats['venue_iclp'] = (df['venue'] == 'iclp').astype(int)
        feats['year_norm'] = (df['year'] - 2016) / 10
        feats['num_authors'] = df['authors'].apply(lambda x: len(str(x).split(',')) if x else 0)
        
        kw = {
            'asp_core': r'answer set|stable model|logic program|clingo|dlv',
            'planning': r'planning|action|scheduling|temporal|path finding',
            'knowledge': r'knowledge|ontolog|description logic',
            'reasoning': r'reasoning|inference|non-monotonic',
            'constraint': r'constraint|satisfiab|sat solver|optimization',
            'learning': r'learning|neural|deep learning|machine learning',
            'explain': r'explain|explanation|interpretab|xai',
            'uncertain': r'probabilistic|uncertain|belief|bayesian',
            'agent': r'multi-agent|agent|game|strategic',
            'complexity': r'complexity|decidab|expressive|tractab',
            'debug': r'debug|repair|diagnosis|inconsisten',
            'argumentation': r'argument|argumentation|dialogue',
            'ground': r'grounding|instantiation',
            'robot': r'robot|autonomous|navigation',
        }
        for name, pattern in kw.items():
            feats[f'kw_{name}'] = text.str.contains(pattern, case=False, na=False).astype(int)
        
        all_feats.append(feats.values)
    
    sc = StandardScaler()
    return sc.fit_transform(all_feats[0]), sc.transform(all_feats[1])


def meta_stacking(feature_sets, y, n_seeds=15):
    """
    Meta-ensemble stacking across multiple feature sets.
    For each seed:
      1. For each (feature_set, model): get OOF probabilities
      2. Stack all OOF probabilities as meta-features
      3. Train LR meta-learner
      4. Predict test
    Final: majority vote across seeds.
    """
    n_train = len(y)
    n_classes = len(np.unique(y))
    n_test = feature_sets[0]['test'].shape[0]
    
    seeds = list(range(n_seeds))
    all_final_preds = []
    all_cv_scores = []
    
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 37 + 42)
        
        meta_train = np.zeros((n_train, 0))
        meta_test = np.zeros((n_test, 0))
        
        for fs in feature_sets:
            X_tr = fs['train']
            X_te = fs['test']
            models = fs['models']
            
            for model in models:
                oof_proba = np.zeros((n_train, n_classes))
                test_proba = np.zeros((n_test, n_classes))
                
                for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(
                    X_tr.toarray() if sp.issparse(X_tr) else X_tr, y)):
                    
                    if sp.issparse(X_tr):
                        Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]
                    else:
                        Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]
                    
                    m = clone(model)
                    m.fit(Xf_tr, y[tr_idx])
                    
                    if hasattr(m, 'predict_proba'):
                        oof_proba[val_idx] = m.predict_proba(Xf_val)
                        test_proba += m.predict_proba(X_te) / 5
                    else:
                        pred_val = m.predict(Xf_val)
                        for i, p in zip(val_idx, pred_val):
                            oof_proba[i, p - 1] = 1.0
                        pred_te = m.predict(X_te)
                        for j in range(n_test):
                            test_proba[j, pred_te[j] - 1] += 1.0 / 5
                
                meta_train = np.hstack([meta_train, oof_proba])
                meta_test = np.hstack([meta_test, test_proba])
        
        # Meta-learner CV evaluation
        meta_cv_scores = []
        for tr_idx, val_idx in cv.split(meta_train, y):
            meta_m = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)
            meta_m.fit(meta_train[tr_idx], y[tr_idx])
            preds = meta_m.predict(meta_train[val_idx])
            meta_cv_scores.append(f1_score(y[val_idx], preds, average='macro'))
        
        seed_score = np.mean(meta_cv_scores)
        all_cv_scores.append(seed_score)
        
        # Final prediction
        meta_final = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)
        meta_final.fit(meta_train, y)
        all_final_preds.append(meta_final.predict(meta_test))
        
        if seed < 10:
            print(f"    Seed {seed:3d}: Meta-CV F1 = {seed_score:.4f}")
    
    # Majority vote
    final = np.array([Counter([p[i] for p in all_final_preds]).most_common(1)[0][0] for i in range(n_test)])
    avg = np.mean(all_cv_scores)
    print(f"\n    Average Meta-CV F1 ({n_seeds} seeds): {avg:.4f} (+/- {np.std(all_cv_scores):.4f})")
    
    return final, avg, all_final_preds, all_cv_scores


def main():
    print("=" * 60)
    print("  ASP Paper Classification v8")
    print("  (Enhanced Meta-Ensemble based on v3)")
    print("=" * 60)
    
    df_tr, df_te = load()
    y = df_tr['Label'].values
    
    # === Features ===
    print("\n--- Phase 1: Feature Extraction ---")
    
    print("  Embeddings...")
    emb = get_embeddings(df_tr, df_te)
    for k, (tr, te) in emb.items():
        print(f"    {k}: {tr.shape}")
    
    print("  TF-IDF...")
    tfidf_tr, tfidf_te = get_tfidf(df_tr, df_te)
    print(f"    tfidf: {tfidf_tr.shape}")
    
    print("  Engineered...")
    eng_tr, eng_te = get_engineered(df_tr, df_te)
    print(f"    eng: {eng_tr.shape}")
    
    # Combined dense features
    emb_eng_tr = np.hstack([emb['scibert_full'][0], eng_tr])
    emb_eng_te = np.hstack([emb['scibert_full'][1], eng_te])
    
    # === Strategy 1: v3-style meta-stacking (baseline to reproduce 0.38) ===
    print("\n--- Strategy 1: v3-style Meta-Stacking ---")
    feature_sets_v3 = [
        {
            'train': emb['scibert_full'][0], 'test': emb['scibert_full'][1],
            'models': [
                SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
                SVC(C=0.5, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
            ]
        },
        {
            'train': tfidf_tr, 'test': tfidf_te,
            'models': [
                CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3),
                LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
                CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3),
            ]
        },
        {
            'train': emb_eng_tr, 'test': emb_eng_te,
            'models': [
                SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
            ]
        },
    ]
    preds_v3, score_v3, _, _ = meta_stacking(feature_sets_v3, y, n_seeds=10)
    
    # === Strategy 2: Enhanced meta-stacking with BGE + better hyperparams ===
    print("\n--- Strategy 2: Enhanced Meta-Stacking ---")
    
    bge_eng_tr = np.hstack([emb['bge_full'][0], eng_tr])
    bge_eng_te = np.hstack([emb['bge_full'][1], eng_te])
    
    feature_sets_v8 = [
        # SciBERT full+abstract
        {
            'train': emb['scibert_full'][0], 'test': emb['scibert_full'][1],
            'models': [
                SVC(C=5.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=10.0, max_iter=5000, class_weight='balanced'),
                SVC(C=7.0, kernel='linear', class_weight='balanced', probability=True),
            ]
        },
        # BGE full
        {
            'train': emb['bge_full'][0], 'test': emb['bge_full'][1],
            'models': [
                SVC(C=7.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=5.0, max_iter=5000, class_weight='balanced'),
                SVC(C=5.0, kernel='linear', class_weight='balanced', probability=True),
            ]
        },
        # SciBERT title only
        {
            'train': emb['scibert_title'][0], 'test': emb['scibert_title'][1],
            'models': [
                SVC(C=10.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=50.0, max_iter=5000, class_weight='balanced'),
            ]
        },
        # BGE title
        {
            'train': emb['bge_title'][0], 'test': emb['bge_title'][1],
            'models': [
                SVC(C=7.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=5.0, max_iter=5000, class_weight='balanced'),
            ]
        },
        # TF-IDF (sparse)
        {
            'train': tfidf_tr, 'test': tfidf_te,
            'models': [
                CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3),
                LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
                CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3),
                SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced'),
            ]
        },
        # SciBERT + engineered
        {
            'train': emb_eng_tr, 'test': emb_eng_te,
            'models': [
                SVC(C=5.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=10.0, max_iter=5000, class_weight='balanced'),
            ]
        },
        # BGE + engineered
        {
            'train': bge_eng_tr, 'test': bge_eng_te,
            'models': [
                SVC(C=5.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=20.0, max_iter=5000, class_weight='balanced'),
            ]
        },
    ]
    preds_v8, score_v8, all_preds_v8, all_scores_v8 = meta_stacking(feature_sets_v8, y, n_seeds=15)
    
    # === Strategy 3: Combine v3 + v8 predictions via another majority vote ===
    print("\n--- Strategy 3: Combined v3+v8 Meta-Stacking ---")
    feature_sets_combined = feature_sets_v3 + [
        {
            'train': emb['bge_full'][0], 'test': emb['bge_full'][1],
            'models': [
                SVC(C=7.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=5.0, max_iter=5000, class_weight='balanced'),
            ]
        },
        {
            'train': emb['scibert_title'][0], 'test': emb['scibert_title'][1],
            'models': [
                SVC(C=10.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
            ]
        },
        {
            'train': emb['bge_title'][0], 'test': emb['bge_title'][1],
            'models': [
                SVC(C=7.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
                LogisticRegression(C=5.0, max_iter=5000, class_weight='balanced'),
            ]
        },
        {
            'train': bge_eng_tr, 'test': bge_eng_te,
            'models': [
                LogisticRegression(C=20.0, max_iter=5000, class_weight='balanced'),
            ]
        },
    ]
    preds_comb, score_comb, _, _ = meta_stacking(feature_sets_combined, y, n_seeds=15)
    
    # === Save ===
    print("\n--- Saving ---")
    sample = pd.read_csv(SAMPLE_SUB)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        'v3_repro': (preds_v3, score_v3),
        'v8_enhanced': (preds_v8, score_v8),
        'v8_combined': (preds_comb, score_comb),
    }
    
    best_name = max(results, key=lambda k: results[k][1])
    
    for name, (preds, score) in results.items():
        sub = sample.copy()
        sub['Label'] = preds.astype(int)
        sub.to_csv(os.path.join(OUTPUT_DIR, f'submission_{name}.csv'), index=False)
        dist = dict(zip(*np.unique(preds, return_counts=True)))
        marker = " <<<< BEST" if name == best_name else ""
        print(f"  {name:20s}: CV={score:.4f} dist={dist}{marker}")
    
    # Main submission = best
    sub = sample.copy()
    sub['Label'] = results[best_name][0].astype(int)
    sub.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"\n  >>> submission.csv = {best_name}")
    
    # Also compare with 0.38 submission
    try:
        s038 = pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_0.38.csv'))
        for name, (preds, score) in results.items():
            sub_df = pd.DataFrame({'id': sample['id'], 'Label': preds.astype(int)})
            merged = s038.merge(sub_df, on='id', suffixes=('_038','_new'))
            agree = (merged['Label_038'] == merged['Label_new']).sum()
            print(f"  Agreement with 0.38: {name} = {agree}/86 ({agree/86*100:.1f}%)")
    except:
        pass
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
