# Project: Academic Paper Classification 

Dự án này thực hiện việc phân loại các bài báo khoa học dựa trên workflow 6 giai đoạn.

## Cấu trúc thư mục hiện tại:
- `app/eda_app.py`: Ứng dụng Streamlit trực quan hóa dữ liệu (Giai đoạn 2).
- `data/raw/`: Chứa các file CSV dữ liệu thô.
- `src/main.py`: Mã nguồn chính thực hiện Tiền xử lý, Huấn luyện mô hình (Logistic Regression & XGBoost), và Xuất file dự đoán (Giai đoạn 3, 4, 6).
- `src/topic_modeling.py`: Mã nguồn mở rộng thực hiện Phân tích Chủ đề (LDA) để tìm Insight (Giai đoạn 5).
- `outputs/`: Nơi lưu trữ file `submission.csv` cuối cùng.
- `requirements.txt`: Chứa danh sách các thư viện cần thiết.
- `setup.ps1`: Script tự động cài đặt môi trường.

## Hướng dẫn chạy dự án (Các bước thực thi)

### Bước 1: Khởi tạo môi trường (Phase 1)
Mở Terminal (PowerShell) tại thư mục gốc của dự án và chạy:
```powershell
.\setup.ps1
```
*(Nếu gặp lỗi phân quyền Execution Policy, hãy chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` trước).*

Sau đó kích hoạt môi trường ảo:
```powershell
.\venv\Scripts\activate
```

### Bước 2: Khám phá Dữ liệu (Phase 2 - EDA)
Chạy ứng dụng Streamlit để xem biểu đồ trực quan hóa dữ liệu:
```powershell
streamlit run app/eda_app.py
```
Ứng dụng sẽ mở ra trên trình duyệt (thường là http://localhost:8501).

### Bước 3: Huấn luyện và Đóng gói (Phase 3, 4, 6)
Để chạy toàn bộ quá trình tiền xử lý, huấn luyện mô hình, và tạo file dự đoán (submission):
```powershell
python src/main.py
```
File dự đoán sẽ được lưu tại `outputs/submission.csv`.

### Bước 4: Chạy Phân tích Chủ đề (Phase 5 - Insights)
Để xem thuật toán phân cụm tự động phân loại các chủ đề như thế nào:
```powershell
python src/topic_modeling.py
```
Kết quả hiển thị trực tiếp trên Terminal, thể hiện sự tương quan giữa các Chủ đề và Label thực tế.
