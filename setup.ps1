# Khởi tạo môi trường ảo (Virtual Environment)
Write-Host "1. Tao moi truong ao env..." -ForegroundColor Green
python -m venv env

# Kích hoạt môi trường ảo và cài đặt thư viện
Write-Host "2. Cai dat cac thu vien tu requirements.txt..." -ForegroundColor Green
.\env\Scripts\activate
pip install -r requirements.txt

Write-Host "Hoan thanh thiet lap moi truong! Hay chay lenh '.\env\Scripts\activate' de kich hoat." -ForegroundColor Cyan
