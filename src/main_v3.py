"""
ASP Paper Classification Pipeline v3 - Definitive version.

Strategy:
1. Extract SciBERT embeddings for semantic understanding
2. Build TF-IDF features for lexical matching  
3. Train separate models on each feature set
4. Combine predictions via meta-ensemble (stacking)
5. Multi-seed majority vote for robustness

Key insight: With only 510 training samples and 5 classes, we need to:
- Use pretrained embeddings to leverage transfer learning
- Avoid overfitting with proper regularization
- Ensemble diverse models for robust predictions
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from collections import Counter

from sentence_transformers import SentenceTransformer

# ==========================================
# PATHS
# ==========================================
BASE_DIR = r"d:\HK252\Data Mining\Assignment"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_train_with_abstracts.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_test_with_abstracts.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "data", "raw", "sample_submission_DM252.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    for df in [df_train, df_test]:
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['authors'] = df['authors'].fillna('')
        df['venue'] = df['venue'].fillna('unknown')
        df['doi'] = df['doi'].fillna('')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020).astype(int)
    
    return df_train, df_test


def get_embeddings(df_train, df_test, model_name='allenai/scibert_scivocab_uncased'):
    """Get sentence embeddings from a pretrained model."""
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Combine title + abstract for better semantic representation
    texts_train = (df_train['title'] + '. ' + df_train['abstract']).tolist()
    texts_test = (df_test['title'] + '. ' + df_test['abstract']).tolist()
    
    print(f"  Encoding {len(texts_train)} train texts...")
    emb_train = model.encode(texts_train, show_progress_bar=True, batch_size=32)
    print(f"  Encoding {len(texts_test)} test texts...")
    emb_test = model.encode(texts_test, show_progress_bar=True, batch_size=32)
    
    return emb_train, emb_test


def get_tfidf_features(df_train, df_test):
    """Build TF-IDF feature matrices."""
    
    # Title TF-IDF (word n-grams)
    tfidf_title = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 3), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode'
    )
    X_train_title = tfidf_title.fit_transform(df_train['title'])
    X_test_title = tfidf_title.transform(df_test['title'])
    
    # Abstract TF-IDF
    tfidf_abstract = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode'
    )
    X_train_abstract = tfidf_abstract.fit_transform(df_train['abstract'])
    X_test_abstract = tfidf_abstract.transform(df_test['abstract'])
    
    # Combined text TF-IDF
    combined_train = df_train['title'] + ' ' + df_train['abstract']
    combined_test = df_test['title'] + ' ' + df_test['abstract']
    
    tfidf_combined = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode'
    )
    X_train_comb = tfidf_combined.fit_transform(combined_train)
    X_test_comb = tfidf_combined.transform(combined_test)
    
    # Character n-grams (captures morphological patterns)
    tfidf_char = TfidfVectorizer(
        max_features=3000, ngram_range=(3, 5), analyzer='char_wb',
        sublinear_tf=True, min_df=2, max_df=0.95
    )
    X_train_char = tfidf_char.fit_transform(df_train['title'] + ' ' + df_train['abstract'])
    X_test_char = tfidf_char.transform(df_test['title'] + ' ' + df_test['abstract'])
    
    # Author TF-IDF
    tfidf_auth = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_auth = tfidf_auth.fit_transform(df_train['authors'])
    X_test_auth = tfidf_auth.transform(df_test['authors'])
    
    X_train = sp.hstack([X_train_title, X_train_abstract, X_train_comb, X_train_char, X_train_auth]).tocsr()
    X_test = sp.hstack([X_test_title, X_test_abstract, X_test_comb, X_test_char, X_test_auth]).tocsr()
    
    return X_train, X_test


def get_engineered_features(df_train, df_test):
    """Build hand-crafted features."""
    all_feats = []
    for df in [df_train, df_test]:
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
        
        # Domain keywords
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
    
    scaler = StandardScaler()
    X_train_eng = scaler.fit_transform(all_feats[0])
    X_test_eng = scaler.transform(all_feats[1])
    
    return X_train_eng, X_test_eng


def cv_evaluate(model, X, y, name="Model"):
    """Quick 5-fold CV evaluation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        if sp.issparse(X):
            X_tr, X_val = X[train_idx], X[val_idx]
        else:
            X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    
    mean_score = np.mean(scores)
    print(f"  {name:40s}: {mean_score:.4f} (+/- {np.std(scores):.4f})")
    return mean_score


def meta_ensemble_predict(X_train_emb, X_train_tfidf, X_train_eng, y_train,
                           X_test_emb, X_test_tfidf, X_test_eng, n_seeds=10):
    """
    Meta-ensemble approach:
    1. Train specialized models on each feature set
    2. Get out-of-fold predictions
    3. Stack them with a meta-learner
    4. Multi-seed for robustness
    """
    
    print("\n  === Meta-Ensemble: Cross-validated stacking ===")
    
    all_final_preds = []
    all_cv_scores = []
    
    seeds = [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555][:n_seeds]
    
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # Define base models for each feature set
        models_emb = [
            ('svc_emb', SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True, random_state=seed)),
            ('lr_emb', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)),
            ('svc_emb2', SVC(C=0.5, kernel='rbf', class_weight='balanced', gamma='scale', probability=True, random_state=seed)),
        ]
        
        models_tfidf = [
            ('svc_tfidf', CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
            ('lr_tfidf', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)),
            ('svc_tfidf2', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
        ]
        
        # Combined features (embeddings + engineered)
        X_train_combined = np.hstack([X_train_emb, X_train_eng])
        X_test_combined = np.hstack([X_test_emb, X_test_eng])
        
        models_combined = [
            ('svc_comb', SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True, random_state=seed)),
            ('lr_comb', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)),
        ]
        
        # Get out-of-fold predictions for meta-features
        n_train = len(y_train)
        n_test = X_test_emb.shape[0]
        n_classes = 5
        
        meta_train = np.zeros((n_train, 0))
        meta_test = np.zeros((n_test, 0))
        
        # Process each (feature_set, model) combination
        all_model_configs = [
            (X_train_emb, X_test_emb, models_emb),
            (X_train_tfidf, X_test_tfidf, models_tfidf),
            (X_train_combined, X_test_combined, models_combined),
        ]
        
        for X_tr_feat, X_te_feat, model_list in all_model_configs:
            for model_name, model in model_list:
                # Out-of-fold predictions (probabilities)
                oof_proba = np.zeros((n_train, n_classes))
                test_proba = np.zeros((n_test, n_classes))
                
                for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_tr_feat if not sp.issparse(X_tr_feat) else X_tr_feat.toarray(), y_train)):
                    if sp.issparse(X_tr_feat):
                        X_fold_tr = X_tr_feat[tr_idx]
                        X_fold_val = X_tr_feat[val_idx]
                    else:
                        X_fold_tr = X_tr_feat[tr_idx]
                        X_fold_val = X_tr_feat[val_idx]
                    
                    y_fold_tr = y_train[tr_idx]
                    
                    model_clone = _clone_model(model)
                    model_clone.fit(X_fold_tr, y_fold_tr)
                    
                    if hasattr(model_clone, 'predict_proba'):
                        oof_proba[val_idx] = model_clone.predict_proba(X_fold_val)
                        test_proba += model_clone.predict_proba(X_te_feat) / 5
                    else:
                        pred = model_clone.predict(X_fold_val)
                        for i, p in zip(val_idx, pred):
                            oof_proba[i, p - 1] = 1.0  # one-hot
                        pred_test = model_clone.predict(X_te_feat)
                        for p in pred_test:
                            test_proba[:, p - 1] += 1.0 / 5
                
                meta_train = np.hstack([meta_train, oof_proba])
                meta_test = np.hstack([meta_test, test_proba])
        
        # Train meta-learner on stacked predictions
        meta_model = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)
        
        # Evaluate meta-model
        meta_cv_scores = []
        for tr_idx, val_idx in cv.split(meta_train, y_train):
            meta_model_fold = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)
            meta_model_fold.fit(meta_train[tr_idx], y_train[tr_idx])
            preds = meta_model_fold.predict(meta_train[val_idx])
            meta_cv_scores.append(f1_score(y_train[val_idx], preds, average='macro'))
        
        seed_score = np.mean(meta_cv_scores)
        all_cv_scores.append(seed_score)
        
        # Final prediction
        meta_model.fit(meta_train, y_train)
        final_preds = meta_model.predict(meta_test)
        all_final_preds.append(final_preds)
        
        print(f"    Seed {seed:5d}: Meta-CV F1 = {seed_score:.4f}")
    
    # Majority vote across seeds
    final = np.zeros(X_test_emb.shape[0], dtype=int)
    for i in range(X_test_emb.shape[0]):
        votes = [p[i] for p in all_final_preds]
        final[i] = Counter(votes).most_common(1)[0][0]
    
    avg_score = np.mean(all_cv_scores)
    print(f"\n    Average Meta-CV F1: {avg_score:.4f} (+/- {np.std(all_cv_scores):.4f})")
    
    return final, avg_score


def _clone_model(model):
    """Simple model cloning."""
    from sklearn.base import clone
    return clone(model)


def simple_ensemble_predict(X_train, y_train, X_test, n_seeds=10):
    """Simple soft-voting ensemble with multiple seeds."""
    
    all_preds = []
    all_scores = []
    seeds = [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555][:n_seeds]
    
    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        ensemble = VotingClassifier(
            estimators=[
                ('svc1', CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                ('svc2', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                ('lr1', LogisticRegression(C=0.5, max_iter=5000, class_weight='balanced', random_state=seed)),
                ('lr2', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)),
                ('sgd', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=seed)),
            ],
            voting='soft', n_jobs=-1
        )
        
        # CV
        fold_scores = []
        for tr_idx, val_idx in cv.split(X_train if not sp.issparse(X_train) else X_train.toarray(), y_train):
            if sp.issparse(X_train):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            else:
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            
            ensemble_fold = VotingClassifier(
                estimators=[
                    ('svc1', CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                    ('svc2', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                    ('lr1', LogisticRegression(C=0.5, max_iter=5000, class_weight='balanced', random_state=seed)),
                    ('lr2', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', random_state=seed)),
                    ('sgd', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=seed)),
                ],
                voting='soft', n_jobs=-1
            )
            ensemble_fold.fit(X_tr, y_train[tr_idx])
            preds = ensemble_fold.predict(X_val)
            fold_scores.append(f1_score(y_train[val_idx], preds, average='macro'))
        
        seed_score = np.mean(fold_scores)
        all_scores.append(seed_score)
        
        ensemble.fit(X_train, y_train)
        all_preds.append(ensemble.predict(X_test))
    
    # Majority vote
    n_test = X_test.shape[0]
    final = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        votes = [p[i] for p in all_preds]
        final[i] = Counter(votes).most_common(1)[0][0]
    
    return final, np.mean(all_scores)


def main():
    print("=" * 60)
    print("  ASP Paper Classification Pipeline v3")
    print("  (SciBERT Embeddings + TF-IDF + Meta-Ensemble)")
    print("=" * 60)
    
    # Load data
    print("\n--- Phase 1: Loading Data ---")
    df_train, df_test = load_data()
    y_train = df_train['Label'].values
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  Labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Extract features
    print("\n--- Phase 2: Extracting Features ---")
    
    print("  [A] SciBERT Embeddings...")
    emb_train, emb_test = get_embeddings(df_train, df_test)
    print(f"      Shape: {emb_train.shape}")
    
    # Normalize embeddings
    from sklearn.preprocessing import normalize
    emb_train = normalize(emb_train)
    emb_test = normalize(emb_test)
    
    print("  [B] TF-IDF Features...")
    tfidf_train, tfidf_test = get_tfidf_features(df_train, df_test)
    print(f"      Shape: {tfidf_train.shape}")
    
    print("  [C] Engineered Features...")
    eng_train, eng_test = get_engineered_features(df_train, df_test)
    print(f"      Shape: {eng_train.shape}")
    
    # Evaluate individual approaches
    print("\n--- Phase 3: Individual Model Evaluation ---")
    
    print("\n  [A] Models on SciBERT Embeddings:")
    cv_evaluate(SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale'), emb_train, y_train, "SVC(rbf, C=1.0) on emb")
    cv_evaluate(SVC(C=0.5, kernel='rbf', class_weight='balanced', gamma='scale'), emb_train, y_train, "SVC(rbf, C=0.5) on emb")
    cv_evaluate(SVC(C=2.0, kernel='rbf', class_weight='balanced', gamma='scale'), emb_train, y_train, "SVC(rbf, C=2.0) on emb")
    cv_evaluate(LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'), emb_train, y_train, "LR(C=1.0) on emb")
    cv_evaluate(LogisticRegression(C=0.5, max_iter=5000, class_weight='balanced'), emb_train, y_train, "LR(C=0.5) on emb")
    
    print("\n  [B] Models on TF-IDF:")
    cv_evaluate(CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3), tfidf_train, y_train, "LinearSVC(C=0.5) on tfidf")
    cv_evaluate(CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3), tfidf_train, y_train, "LinearSVC(C=1.0) on tfidf")
    cv_evaluate(LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'), tfidf_train, y_train, "LR(C=1.0) on tfidf")
    
    print("\n  [C] Models on Embeddings + Engineered:")
    X_emb_eng_train = np.hstack([emb_train, eng_train])
    X_emb_eng_test = np.hstack([emb_test, eng_test])
    cv_evaluate(SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale'), X_emb_eng_train, y_train, "SVC(rbf, C=1.0) on emb+eng")
    cv_evaluate(LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'), X_emb_eng_train, y_train, "LR(C=1.0) on emb+eng")
    
    # LSA reduced TF-IDF + embeddings
    print("\n  [D] Models on LSA(TF-IDF) + Embeddings:")
    svd = TruncatedSVD(n_components=100, random_state=42)
    tfidf_lsa_train = svd.fit_transform(tfidf_train)
    tfidf_lsa_test = svd.transform(tfidf_test)
    
    X_all_dense_train = np.hstack([emb_train, tfidf_lsa_train, eng_train])
    X_all_dense_test = np.hstack([emb_test, tfidf_lsa_test, eng_test])
    cv_evaluate(SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale'), X_all_dense_train, y_train, "SVC(rbf) on emb+lsa+eng")
    cv_evaluate(LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'), X_all_dense_train, y_train, "LR on emb+lsa+eng")
    
    # Meta-ensemble
    print("\n--- Phase 4: Meta-Ensemble Prediction ---")
    preds_meta, score_meta = meta_ensemble_predict(
        emb_train, tfidf_train, eng_train, y_train,
        emb_test, tfidf_test, eng_test, n_seeds=10
    )
    
    # Simple ensemble on best feature set
    print("\n--- Phase 5: Simple Ensemble on Combined Dense Features ---")
    preds_dense, score_dense = simple_ensemble_predict(X_all_dense_train, y_train, X_all_dense_test, n_seeds=10)
    print(f"  Dense ensemble CV: {score_dense:.4f}")
    
    # Simple ensemble on TF-IDF
    print("\n--- Phase 6: Simple Ensemble on TF-IDF ---")
    preds_tfidf, score_tfidf = simple_ensemble_predict(tfidf_train, y_train, tfidf_test, n_seeds=10)
    print(f"  TF-IDF ensemble CV: {score_tfidf:.4f}")
    
    # Save all submissions
    print("\n--- Phase 7: Saving Submissions ---")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        'meta_ensemble': (preds_meta, score_meta),
        'dense_ensemble': (preds_dense, score_dense),
        'tfidf_ensemble': (preds_tfidf, score_tfidf),
    }
    
    best_name = max(results, key=lambda k: results[k][1])
    
    for name, (preds, score) in results.items():
        sub = sample_sub.copy()
        sub['Label'] = preds.astype(int)
        out_path = os.path.join(OUTPUT_DIR, f'submission_{name}.csv')
        sub.to_csv(out_path, index=False)
        dist = dict(zip(*np.unique(preds, return_counts=True)))
        marker = " <<<< BEST" if name == best_name else ""
        print(f"  {name:20s}: CV={score:.4f}, dist={dist}{marker}")
    
    # Save best as main submission
    best_preds = results[best_name][0]
    sub = sample_sub.copy()
    sub['Label'] = best_preds.astype(int)
    sub.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"\n  Main submission: outputs/submission.csv (strategy: {best_name})")
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
