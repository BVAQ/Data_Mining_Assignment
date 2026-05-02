import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

TRAIN_PATH = r"..\data\raw\Stage_1_publcitrain.csv"

def run_topic_modeling():
    print("--- Phase 5: Phân tích Chủ đề (Topic Modeling) ---")
    
    # Handle local path execution context
    import os
    path = TRAIN_PATH if os.path.exists(TRAIN_PATH) else r"data\raw\Stage_1_publcitrain.csv"
    
    df = pd.read_csv(path)
    
    # Lấy text feature (title)
    texts = df['title'].fillna("").values
    
    # Count Vectorizer
    tf_vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    tf = tf_vectorizer.fit_transform(texts)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    # LDA
    n_topics = 5 # Số cụm ta muốn tìm, bằng với số label (1-5) để dễ so sánh
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online')
    lda.fit(tf)
    
    print("\n1. Các chủ đề được phát hiện (Top từ khóa):")
    n_top_words = 10
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [tf_feature_names[i] for i in top_words_idx]
        print(f"Chủ đề {topic_idx + 1}: {', '.join(top_words)}")
        
    # Tính toán Topic cho từng bài báo
    topic_distributions = lda.transform(tf)
    df['Dominant_Topic'] = topic_distributions.argmax(axis=1) + 1
    
    print("\n2. Mối liên hệ giữa Chủ đề phát hiện (Dominant Topic) và Nhãn gốc (Label):")
    # Bảng chéo (Cross-tabulation)
    cross_tab = pd.crosstab(df['Dominant_Topic'], df['Label'])
    print(cross_tab)
    
    print("\nNhận xét (Insight):")
    print("- Dựa vào bảng phân phối, ta có thể đánh giá thủ công (gắn nhãn) các Chủ đề (1-5) dựa trên nhãn nào chiếm đa số trong chủ đề đó.")
    print("- Ví dụ, nếu Chủ đề 1 chứa nhiều bài báo thuộc Label 3, có thể suy ra từ khóa của Chủ đề 1 đại diện mạnh mẽ cho lĩnh vực của Label 3.")
    
if __name__ == "__main__":
    run_topic_modeling()
