import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Vui lòng chạy: pip install sentence-transformers")
    exit()

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("Vui lòng chạy: pip install catboost")
    exit()

# Paths
TRAIN_PATH = r"data\preprocessed\Stage_1_train_with_abstracts.csv"
TEST_PATH = r"data\preprocessed\Stage_1_test_with_abstracts.csv"
SAMPLE_SUB_PATH = r"data\raw\sample_submission_DM252.csv"
OUTPUT_DIR = r"outputs"

def load_data():
    base_dir = r"d:\HK252\Data Mining\Assignment"
    train_file = os.path.join(base_dir, TRAIN_PATH)
    test_file = os.path.join(base_dir, TEST_PATH)
    sub_file = os.path.join(base_dir, SAMPLE_SUB_PATH)
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Xử lý NAs
    for df in [df_train, df_test]:
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['authors'] = df['authors'].fillna('Unknown').str.replace(',', ' ')
        df['venue'] = df['venue'].fillna('Unknown')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2000)
        
    return df_train, df_test, sub_file, base_dir

def main():
    print("--- Phase 1: Loading Data ---")
    df_train, df_test, sub_file, base_dir = load_data()
    y_train = df_train['Label'].values

    print("\n--- Phase 2: Extracting Text Embeddings with SciBERT ---")
    # Sử dụng SciBERT để mã hóa ngữ nghĩa chuyên sâu cho Title + Abstract
    text_train = (df_train['title'] + ". " + df_train['abstract']).tolist()
    text_test = (df_test['title'] + ". " + df_test['abstract']).tolist()
    
    model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    X_train_text = model.encode(text_train, show_progress_bar=True)
    X_test_text = model.encode(text_test, show_progress_bar=True)
    
    print("\n--- Phase 3: Extracting Tabular Features ---")
    # Mã hóa các trường dữ liệu dạng Tabular (Tác giả, Nơi xuất bản, Năm)
    tabular_prep = ColumnTransformer([
        ('venue_ohe', OneHotEncoder(handle_unknown='ignore'), ['venue']),
        ('year_scaler', StandardScaler(), ['year']),
        ('authors_tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1,2)), 'authors')
    ])
    
    X_train_tab = tabular_prep.fit_transform(df_train)
    X_test_tab = tabular_prep.transform(df_test)
    
    # Kết hợp SciBERT embeddings (Dense) với Tabular features (Sparse)
    X_train_final = sp.hstack([X_train_text, X_train_tab])
    X_test_final = sp.hstack([X_test_text, X_test_tab])
    
    print("\n--- Phase 4: Training CatBoost Model (Ensemble Approach) ---")
    clf = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        auto_class_weights='Balanced',
        random_seed=42
    )
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    
    print("Running 5-Fold Cross Validation...")
    X_train_csr = X_train_final.tocsr()
    for i, (train_idx, val_idx) in enumerate(cv.split(X_train_csr, y_train)):
        X_tr, X_val = X_train_csr[train_idx], X_train_csr[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        preds = clf.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        f1_scores.append(score)
        print(f"Fold {i+1} Macro F1: {score:.4f}")
    
    mean_f1 = np.mean(f1_scores)
    print(f"\n=> Ensemble (SciBERT + CatBoost) Macro F1-score: {mean_f1:.4f} (+/- {np.std(f1_scores):.4f})")
    
    print("\n--- Phase 5: Predicting and Generating Submission ---")
    clf.fit(X_train_csr, y_train, verbose=0)
    final_preds = clf.predict(X_test_final.tocsr()).flatten()
    
    submission = pd.read_csv(sub_file)
    submission['Label'] = final_preds
    
    out_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "submission.csv")
    submission.to_csv(out_file, index=False)
    print(f"Successfully saved predictions to {out_file}")
    print("Sẵn sàng để nộp lên Kaggle!")

if __name__ == "__main__":
    main()
