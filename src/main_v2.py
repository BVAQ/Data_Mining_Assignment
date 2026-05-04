"""
High-performance research paper classifier for ASP domain.
Optimized for Macro F1-score with:
1. TF-IDF on title + abstract (most important features for text classification)
2. Careful feature engineering on metadata (DOI, venue, year, authors)
3. Ensemble of strong linear models (best for high-dim sparse data)
4. Multi-seed aggregation for robust predictions
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
import scipy.sparse as sp

# ==========================================
# PATHS
# ==========================================
BASE_DIR = r"d:\HK252\Data Mining\Assignment"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_train_with_abstracts.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "Stage_1_test_with_abstracts.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "data", "raw", "sample_submission_DM252.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    """Load preprocessed data with abstracts."""
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


def build_features(df_train, df_test):
    """Build feature matrices using TF-IDF and engineered features."""
    
    # === TEXT FEATURES ===
    
    # 1. TF-IDF on title (unigrams + bigrams + trigrams)
    print("  [1/6] TF-IDF on title...")
    tfidf_title = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 3), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode',
        token_pattern=r'(?u)\b\w+\b'
    )
    X_train_title = tfidf_title.fit_transform(df_train['title'])
    X_test_title = tfidf_title.transform(df_test['title'])
    
    # 2. TF-IDF on abstract
    print("  [2/6] TF-IDF on abstract...")
    tfidf_abstract = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode',
        token_pattern=r'(?u)\b\w+\b'
    )
    X_train_abstract = tfidf_abstract.fit_transform(df_train['abstract'])
    X_test_abstract = tfidf_abstract.transform(df_test['abstract'])
    
    # 3. TF-IDF on combined title + abstract (captures cross-field patterns)
    print("  [3/6] TF-IDF on combined text...")
    combined_train = (df_train['title'] + ' ' + df_train['abstract'])
    combined_test = (df_test['title'] + ' ' + df_test['abstract'])
    
    tfidf_combined = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=2, max_df=0.95, strip_accents='unicode'
    )
    X_train_combined = tfidf_combined.fit_transform(combined_train)
    X_test_combined = tfidf_combined.transform(combined_test)
    
    # 4. Character n-grams on title (captures morphological patterns)
    print("  [4/6] Character n-grams on title...")
    tfidf_char = TfidfVectorizer(
        max_features=3000, ngram_range=(3, 5), analyzer='char_wb',
        sublinear_tf=True, min_df=2, max_df=0.95
    )
    X_train_char = tfidf_char.fit_transform(df_train['title'])
    X_test_char = tfidf_char.transform(df_test['title'])
    
    # 5. TF-IDF on authors
    print("  [5/6] TF-IDF on authors...")
    tfidf_authors = TfidfVectorizer(
        max_features=1000, ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )
    X_train_authors = tfidf_authors.fit_transform(df_train['authors'])
    X_test_authors = tfidf_authors.transform(df_test['authors'])
    
    # === ENGINEERED FEATURES ===
    print("  [6/6] Engineered features...")
    
    eng_features = []
    for df in [df_train, df_test]:
        feats = pd.DataFrame()
        title = df['title']
        abstract = df['abstract']
        text = (title + ' ' + abstract).str.lower()
        
        # Length features
        feats['title_len'] = title.str.len()
        feats['title_words'] = title.str.split().str.len()
        feats['abstract_len'] = abstract.str.len()
        feats['abstract_words'] = abstract.str.split().str.len().fillna(0)
        feats['has_abstract'] = (abstract.str.len() > 10).astype(int)
        
        # Venue
        feats['venue_iclp'] = (df['venue'] == 'iclp').astype(int)
        feats['venue_kr'] = (df['venue'] == 'kr').astype(int)
        
        # Year
        feats['year'] = df['year']
        feats['year_norm'] = (df['year'] - 2016) / 10
        feats['year_recent'] = (df['year'] >= 2023).astype(int)
        
        # Authors
        feats['num_authors'] = df['authors'].apply(lambda x: len(str(x).split(',')) if x else 0)
        feats['has_authors'] = (df['authors'].str.len() > 1).astype(int)
        
        # DOI-based features
        doi = df['doi']
        feats['doi_kr'] = doi.str.contains('kr\\.20', na=False).astype(int)
        feats['doi_eptcs'] = doi.str.contains('eptcs', case=False, na=False).astype(int)
        feats['doi_springer'] = doi.str.contains('10\\.1007', na=False).astype(int)
        feats['doi_elsevier'] = doi.str.contains('10\\.1016', na=False).astype(int)
        feats['doi_aaai'] = doi.str.contains('aaai', case=False, na=False).astype(int)
        feats['doi_acm'] = doi.str.contains('10\\.1145', na=False).astype(int)
        feats['doi_ieee'] = doi.str.contains('10\\.1109', na=False).astype(int)
        
        # Keyword indicators (domain-specific topics)
        keyword_groups = {
            'asp_core': r'answer set|stable model|logic program|clingo|dlv|gringo',
            'planning': r'planning|action|scheduling|temporal|path finding',
            'knowledge': r'knowledge represent|ontolog|knowledge graph|description logic',
            'reasoning': r'reasoning|inference|deduction|non-monotonic',
            'constraint': r'constraint|satisfiab|sat solver|csp|optimization',
            'learning': r'learning|neural|deep learning|machine learning|inductive',
            'explain': r'explain|explanation|interpretab|xai|justif|transparent',
            'uncertain': r'probabilistic|uncertain|belief|bayesian|stochastic',
            'agent': r'multi-agent|agent|negotiation|game|strategic|cooperation',
            'complexity': r'complexity|decidab|expressive|tractab|np-hard|computational',
            'debug': r'debug|repair|diagnosis|inconsisten',
            'verify': r'verification|model checking|correctness|formal method',
            'argumentation': r'argument|argumentation|debate|dialogue|framework',
            'update': r'update|revision|dynamic|evolv|stream',
            'preference': r'preference|weak constraint|optimal|pareto',
            'abduction': r'abduction|abductive|hypothesis',
            'aggregate': r'aggregate|recursive|fixpoint|stratif',
            'ground': r'grounding|instantiation|ground program',
            'robot': r'robot|autonomous|navigation|control system',
            'nlp': r'natural language|text mining|nlp|language model|sentiment',
        }
        
        for name, pattern in keyword_groups.items():
            feats[f'kw_{name}'] = text.str.contains(pattern, case=False, na=False).astype(int)
        
        # Title patterns
        feats['has_colon'] = title.str.contains(':', na=False).astype(int)
        feats['has_question'] = title.str.contains(r'\?', na=False).astype(int)
        feats['is_workshop'] = title.str.contains('workshop|proceedings|editorial', case=False, na=False).astype(int)
        feats['is_survey'] = title.str.contains('survey|review|overview|tutorial', case=False, na=False).astype(int)
        
        eng_features.append(feats)
    
    # Scale engineered features using MaxAbsScaler (preserves sparsity, no negatives for NB)
    scaler = MaxAbsScaler()
    eng_train_scaled = scaler.fit_transform(eng_features[0].values)
    eng_test_scaled = scaler.transform(eng_features[1].values)
    
    # DOI TF-IDF
    tfidf_doi = TfidfVectorizer(
        max_features=200, ngram_range=(1, 1),
        token_pattern=r'[^\s/]+', min_df=1
    )
    X_train_doi = tfidf_doi.fit_transform(df_train['doi'])
    X_test_doi = tfidf_doi.transform(df_test['doi'])
    
    # === COMBINE ALL ===
    X_train = sp.hstack([
        X_train_title,       # word n-grams from title
        X_train_abstract,    # word n-grams from abstract
        X_train_combined,    # word n-grams from combined
        X_train_char,        # char n-grams from title
        X_train_authors,     # author TF-IDF
        X_train_doi,         # DOI tokens
        sp.csr_matrix(eng_train_scaled),  # engineered features
    ]).tocsr()
    
    X_test = sp.hstack([
        X_test_title,
        X_test_abstract,
        X_test_combined,
        X_test_char,
        X_test_authors,
        X_test_doi,
        sp.csr_matrix(eng_test_scaled),
    ]).tocsr()
    
    print(f"  Total features: {X_train.shape[1]}")
    return X_train, X_test


def evaluate_models(X_train, y_train):
    """Evaluate individual models to find best configurations."""
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'LinearSVC_C01': CalibratedClassifierCV(
            LinearSVC(C=0.1, max_iter=10000, class_weight='balanced'), cv=3),
        'LinearSVC_C03': CalibratedClassifierCV(
            LinearSVC(C=0.3, max_iter=10000, class_weight='balanced'), cv=3),
        'LinearSVC_C05': CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3),
        'LinearSVC_C1': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3),
        'LinearSVC_C2': CalibratedClassifierCV(
            LinearSVC(C=2.0, max_iter=10000, class_weight='balanced'), cv=3),
        'LinearSVC_C5': CalibratedClassifierCV(
            LinearSVC(C=5.0, max_iter=10000, class_weight='balanced'), cv=3),
        'LR_C01': LogisticRegression(C=0.1, max_iter=5000, solver='lbfgs', class_weight='balanced'),
        'LR_C05': LogisticRegression(C=0.5, max_iter=5000, solver='lbfgs', class_weight='balanced'),
        'LR_C1': LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', class_weight='balanced'),
        'LR_C5': LogisticRegression(C=5.0, max_iter=5000, solver='lbfgs', class_weight='balanced'),
        'SGD_hinge': CalibratedClassifierCV(
            SGDClassifier(loss='hinge', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=42), cv=3),
        'SGD_log': SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=42),
        'SVC_rbf': SVC(C=1.0, kernel='rbf', class_weight='balanced', gamma='scale', probability=True),
    }
    
    results = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
            results[name] = (scores.mean(), scores.std())
            print(f"  {name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"  {name:25s}: ERROR - {str(e)[:80]}")
    
    # Find best
    best_name = max(results, key=lambda x: results[x][0])
    print(f"\n  >>> Best model: {best_name} with F1={results[best_name][0]:.4f}")
    
    return results


def build_ensemble_and_predict(X_train, y_train, X_test):
    """Build robust ensemble with multiple strategies."""
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Strategy 1: Soft Voting Ensemble of diverse linear models
    print("\n  Strategy 1: Soft Voting Ensemble")
    ensemble = VotingClassifier(
        estimators=[
            ('svc01', CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=10000, class_weight='balanced'), cv=3)),
            ('svc03', CalibratedClassifierCV(LinearSVC(C=0.3, max_iter=10000, class_weight='balanced'), cv=3)),
            ('svc05', CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3)),
            ('svc1', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3)),
            ('svc2', CalibratedClassifierCV(LinearSVC(C=2.0, max_iter=10000, class_weight='balanced'), cv=3)),
            ('lr01', LogisticRegression(C=0.1, max_iter=5000, solver='lbfgs', class_weight='balanced')),
            ('lr05', LogisticRegression(C=0.5, max_iter=5000, solver='lbfgs', class_weight='balanced')),
            ('lr1', LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', class_weight='balanced')),
            ('sgd', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=42)),
        ],
        voting='soft', n_jobs=-1
    )
    scores_ens = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
    print(f"    CV Macro F1: {scores_ens.mean():.4f} (+/- {scores_ens.std():.4f})")
    
    ensemble.fit(X_train, y_train)
    preds_ens = ensemble.predict(X_test)
    
    # Strategy 2: Stacking with linear meta-learner
    print("\n  Strategy 2: Stacking Ensemble")
    stacking = StackingClassifier(
        estimators=[
            ('svc05', CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000, class_weight='balanced'), cv=3)),
            ('svc1', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000, class_weight='balanced'), cv=3)),
            ('lr1', LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', class_weight='balanced')),
            ('sgd', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=42)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
        cv=5, n_jobs=-1, passthrough=False
    )
    scores_stack = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
    print(f"    CV Macro F1: {scores_stack.mean():.4f} (+/- {scores_stack.std():.4f})")
    
    stacking.fit(X_train, y_train)
    preds_stack = stacking.predict(X_test)
    
    # Strategy 3: Multi-seed majority vote for maximum robustness
    print("\n  Strategy 3: Multi-Seed Majority Vote")
    from collections import Counter
    
    all_preds = []
    all_scores = []
    
    for seed in [42, 123, 456, 789, 2024, 7, 13, 99, 314, 555]:
        model = VotingClassifier(
            estimators=[
                ('svc05', CalibratedClassifierCV(
                    LinearSVC(C=0.5, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                ('svc1', CalibratedClassifierCV(
                    LinearSVC(C=1.0, max_iter=10000, class_weight='balanced', random_state=seed), cv=3)),
                ('lr', LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', class_weight='balanced', random_state=seed)),
                ('sgd', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=5000, class_weight='balanced', random_state=seed)),
            ],
            voting='soft', n_jobs=-1
        )
        
        cv_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=cv_seed, scoring='f1_macro', n_jobs=1)
        all_scores.append(scores.mean())
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds.append(preds)
    
    # Majority vote
    n_test = X_test.shape[0]
    preds_multi = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        votes = [p[i] for p in all_preds]
        preds_multi[i] = Counter(votes).most_common(1)[0][0]
    
    print(f"    Avg CV across seeds: {np.mean(all_scores):.4f} (+/- {np.std(all_scores):.4f})")
    for i, (seed, score) in enumerate(zip([42, 123, 456, 789, 2024, 7, 13, 99, 314, 555], all_scores)):
        print(f"      Seed {seed:5d}: {score:.4f}")
    
    return {
        'ensemble': (preds_ens, scores_ens.mean()),
        'stacking': (preds_stack, scores_stack.mean()),
        'multiseed': (preds_multi, np.mean(all_scores)),
    }


def main():
    print("=" * 60)
    print("  ASP Paper Classification Pipeline v2")
    print("=" * 60)
    
    # Load
    print("\n--- Phase 1: Loading Data ---")
    df_train, df_test = load_data()
    y_train = df_train['Label'].values
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}, Classes: {sorted(np.unique(y_train))}")
    print(f"  Distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Features
    print("\n--- Phase 2: Feature Engineering ---")
    X_train, X_test = build_features(df_train, df_test)
    
    # Evaluate individual models
    print("\n--- Phase 3: Individual Model Evaluation ---")
    results = evaluate_models(X_train, y_train)
    
    # Ensemble
    print("\n--- Phase 4: Ensemble Strategies ---")
    predictions = build_ensemble_and_predict(X_train, y_train, X_test)
    
    # Save all submissions
    print("\n--- Phase 5: Saving Submissions ---")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_name = max(predictions, key=lambda k: predictions[k][1])
    
    for name, (preds, score) in predictions.items():
        sub = sample_sub.copy()
        sub['Label'] = preds.astype(int)
        out_path = os.path.join(OUTPUT_DIR, f'submission_{name}.csv')
        sub.to_csv(out_path, index=False)
        dist = dict(zip(*np.unique(preds, return_counts=True)))
        marker = " <<<< BEST" if name == best_name else ""
        print(f"  {name:15s}: CV={score:.4f}, dist={dist}{marker}")
    
    # Save best as main submission
    best_preds = predictions[best_name][0]
    sub = sample_sub.copy()
    sub['Label'] = best_preds.astype(int)
    sub.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"\n  Main submission saved as outputs/submission.csv (strategy: {best_name})")
    
    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
