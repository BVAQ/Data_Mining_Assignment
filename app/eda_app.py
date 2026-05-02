import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Set page config
st.set_page_config(page_title="EDA: Paper Classification", layout="wide")

# Paths
TRAIN_PATH = r"..\data\raw\Stage_1_publcitrain.csv"

st.title("📊 Khám phá Dữ liệu (EDA) - Phân loại Bài báo Khoa học")

@st.cache_data
def load_data():
    if os.path.exists(TRAIN_PATH):
        return pd.read_csv(TRAIN_PATH)
    else:
        # Fallback to local path if running from root
        local_path = r"data\raw\Stage_1_publcitrain.csv"
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return None

df = load_data()

if df is None:
    st.error(f"Không tìm thấy file dữ liệu tại: {TRAIN_PATH}. Vui lòng kiểm tra lại đường dẫn!")
    st.stop()

st.sidebar.header("Tổng quan Dữ liệu")
st.sidebar.write(f"Số lượng mẫu: {df.shape[0]}")
st.sidebar.write(f"Số lượng đặc trưng: {df.shape[1]}")

st.subheader("1. Mẫu dữ liệu")
st.dataframe(df.head())

st.subheader("2. Phân phối của Nhãn (Label)")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Label', palette='viridis', ax=ax)
    ax.set_title("Số lượng bài báo theo từng Label")
    st.pyplot(fig)

with col2:
    st.write("Bảng thống kê số lượng Label:")
    st.write(df['Label'].value_counts().reset_index().rename(columns={'index': 'Label', 'Label': 'Count'}))
    st.info("Nhận xét: Kiểm tra xem dữ liệu có bị mất cân bằng (imbalanced data) hay không để chọn chiến lược huấn luyện phù hợp.")

st.subheader("3. Top Hội nghị (Venue) Phổ biến")
fig2, ax2 = plt.subplots(figsize=(10, 5))
top_venues = df['venue'].value_counts().head(10)
sns.barplot(y=top_venues.index, x=top_venues.values, palette='magma', ax=ax2)
ax2.set_title("Top 10 Hội nghị có nhiều bài báo nhất")
ax2.set_xlabel("Số lượng bài báo")
ax2.set_ylabel("Hội nghị")
st.pyplot(fig2)

st.subheader("4. Xu hướng xuất bản qua các năm")
fig3, ax3 = plt.subplots(figsize=(10, 5))
year_counts = df['year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', ax=ax3)
ax3.set_title("Số lượng bài báo được xuất bản theo năm")
ax3.set_xlabel("Năm")
ax3.set_ylabel("Số lượng bài báo")
st.pyplot(fig3)

st.subheader("5. Word Cloud cho Tiêu đề (Title)")
text = " ".join(title for title in df['title'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.imshow(wordcloud, interpolation='bilinear')
ax4.axis("off")
st.pyplot(fig4)

st.success("Hoàn thành Giai đoạn 2: EDA!")
