"""
ASP Paper Classification - v4 Focused Pipeline
Key changes from v3:
- Try multiple embedding models
- Better hyperparameter tuning with GridSearchCV  
- Smarter feature combination
- Focus on what actually works
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
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

def get_text(df):
    return (df['title'] + '. ' + df['abstract']).tolist()

def encode(model_name, texts_tr, texts_te):
    print(f"  Encoding with {model_name}...")
    model = SentenceTransformer(model_name)
    emb_tr = model.encode(texts_tr, show_progress_bar=True, batch_size=32)
    emb_te = model.encode(texts_te, show_progress_bar=True, batch_size=32)
    return normalize(emb_tr), normalize(emb_te)

def tfidf_features(df_tr, df_te):
    # Word TF-IDF on combined text
    combined_tr = df_tr['title'] + ' ' + df_tr['abstract']
    combined_te = df_te['title'] + ' ' + df_te['abstract']
    
    tv = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True,
                         min_df=2, max_df=0.95, strip_accents='unicode')
    X_tr = tv.fit_transform(combined_tr)
    X_te = tv.transform(combined_te)
    
    # Reduce to dense with SVD
    svd = TruncatedSVD(n_components=150, random_state=42)
    X_tr_d = svd.fit_transform(X_tr)
    X_te_d = svd.transform(X_te)
    return normalize(X_tr_d), normalize(X_te_d)

def eval_cv(X, y, name=""):
    """Evaluate with grid search over SVC params."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    best_score = 0
    best_params = {}
    
    for C in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for gamma in ['scale', 'auto']:
            model = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced')
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            mean = scores.mean()
            if mean > best_score:
                best_score = mean
                best_params = {'C': C, 'gamma': gamma}
    
    print(f"  {name:40s}: {best_score:.4f} (C={best_params['C']}, gamma={best_params['gamma']})")
    return best_score, best_params

def main():
    print("=" * 60)
    print("  ASP Paper Classification v4")
    print("=" * 60)
    
    df_tr, df_te = load()
    y = df_tr['Label'].values
    texts_tr, texts_te = get_text(df_tr), get_text(df_te)
    
    # Try multiple embedding models
    embedding_models = [
        'all-MiniLM-L6-v2',
        'allenai/scibert_scivocab_uncased',
        'all-mpnet-base-v2',
    ]
    
    all_embs = {}
    for model_name in embedding_models:
        try:
            emb_tr, emb_te = encode(model_name, texts_tr, texts_te)
            all_embs[model_name] = (emb_tr, emb_te)
            score, params = eval_cv(emb_tr, y, model_name)
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")
    
    # TF-IDF + SVD features
    print("\n  TF-IDF + SVD features...")
    tfidf_tr, tfidf_te = tfidf_features(df_tr, df_te)
    eval_cv(tfidf_tr, y, "TF-IDF+SVD")
    
    # Combine all embeddings + TF-IDF
    print("\n  === Combined Features ===")
    
    # Try each embedding + tfidf combo
    best_combo_score = 0
    best_combo_name = ""
    best_combo_params = {}
    combo_data = {}
    
    for name, (emb_tr, emb_te) in all_embs.items():
        combined_tr = np.hstack([emb_tr, tfidf_tr])
        combined_te = np.hstack([emb_te, tfidf_te])
        combo_name = f"{name}+tfidf"
        score, params = eval_cv(combined_tr, y, combo_name)
        combo_data[combo_name] = (combined_tr, combined_te, params)
        if score > best_combo_score:
            best_combo_score = score
            best_combo_name = combo_name
            best_combo_params = params
    
    # Combine ALL embeddings together
    if len(all_embs) > 1:
        all_emb_tr = np.hstack([v[0] for v in all_embs.values()] + [tfidf_tr])
        all_emb_te = np.hstack([v[1] for v in all_embs.values()] + [tfidf_te])
        score, params = eval_cv(all_emb_tr, y, "ALL_COMBINED")
        combo_data["ALL_COMBINED"] = (all_emb_tr, all_emb_te, params)
        if score > best_combo_score:
            best_combo_score = score
            best_combo_name = "ALL_COMBINED"
            best_combo_params = params
    
    print(f"\n  >>> Best combo: {best_combo_name} = {best_combo_score:.4f}")
    
    # Multi-seed ensemble on best combo
    print(f"\n  === Multi-seed prediction on {best_combo_name} ===")
    best_tr, best_te, _ = combo_data[best_combo_name]
    
    all_preds = []
    all_scores = []
    
    for seed in [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        model = SVC(C=best_combo_params['C'], kernel='rbf', gamma=best_combo_params['gamma'],
                    class_weight='balanced', probability=True, random_state=seed)
        
        scores = cross_val_score(model, best_tr, y, cv=cv, 
                                scoring=make_scorer(f1_score, average='macro'))
        all_scores.append(scores.mean())
        
        model.fit(best_tr, y)
        all_preds.append(model.predict(best_te))
        print(f"    Seed {seed:5d}: {scores.mean():.4f}")
    
    # Majority vote
    n = best_te.shape[0]
    final_preds = np.array([Counter([p[i] for p in all_preds]).most_common(1)[0][0] for i in range(n)])
    print(f"\n    Avg CV: {np.mean(all_scores):.4f} (+/- {np.std(all_scores):.4f})")
    
    # Also try LR ensemble on best combo
    print(f"\n  === LR Multi-seed on {best_combo_name} ===")
    lr_preds_list = []
    lr_scores = []
    for seed in [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for C in [0.1, 0.5, 1.0]:
            model = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=seed)
            scores = cross_val_score(model, best_tr, y, cv=cv,
                                    scoring=make_scorer(f1_score, average='macro'))
            lr_scores.append((C, seed, scores.mean()))
    
    best_lr_C = max(lr_scores, key=lambda x: x[2])[0]
    print(f"    Best LR C={best_lr_C}")
    
    lr_all_preds = []
    lr_all_scores = []
    for seed in [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = LogisticRegression(C=best_lr_C, max_iter=5000, class_weight='balanced', random_state=seed)
        scores = cross_val_score(model, best_tr, y, cv=cv,
                                scoring=make_scorer(f1_score, average='macro'))
        lr_all_scores.append(scores.mean())
        model.fit(best_tr, y)
        lr_all_preds.append(model.predict(best_te))
        print(f"    Seed {seed:5d}: {scores.mean():.4f}")
    
    lr_final = np.array([Counter([p[i] for p in lr_all_preds]).most_common(1)[0][0] for i in range(n)])
    print(f"    LR Avg CV: {np.mean(lr_all_scores):.4f}")
    
    # Save submissions
    print("\n--- Saving ---")
    sample = pd.read_csv(SAMPLE_SUB)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    svc_score = np.mean(all_scores)
    lr_score = np.mean(lr_all_scores)
    
    results = {
        'svc_multiseed': (final_preds, svc_score),
        'lr_multiseed': (lr_final, lr_score),
    }
    
    # Also do per-model predictions
    for name, (emb_tr, emb_te) in all_embs.items():
        model = SVC(C=best_combo_params['C'], kernel='rbf', gamma=best_combo_params['gamma'],
                    class_weight='balanced', random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, emb_tr, y, cv=cv, scoring=make_scorer(f1_score, average='macro'))
        model.fit(emb_tr, y)
        preds = model.predict(emb_te)
        sname = name.replace('/', '_')
        results[f'svc_{sname}'] = (preds, scores.mean())
    
    best_name = max(results, key=lambda k: results[k][1])
    
    for name, (preds, score) in results.items():
        sub = sample.copy()
        sub['Label'] = preds.astype(int)
        sub.to_csv(os.path.join(OUTPUT_DIR, f'submission_{name}.csv'), index=False)
        dist = dict(zip(*np.unique(preds, return_counts=True)))
        marker = " <<<< BEST" if name == best_name else ""
        print(f"  {name:35s}: CV={score:.4f} dist={dist}{marker}")
    
    best_preds = results[best_name][0]
    sub = sample.copy()
    sub['Label'] = best_preds.astype(int)
    sub.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"\n  >>> submission.csv = {best_name} (CV={results[best_name][1]:.4f})")
    print("DONE!")

if __name__ == "__main__":
    main()
