# Hướng dẫn dùng bản chia sẻ FACENET_DACN

## 1) Gói này dùng để làm gì?
- Đây là bản `share` đã được cập nhật lại từ source hiện tại của project.
- Bạn nhận file chỉ cần giải nén, mở thư mục và chạy `RUN_APP.bat`.
- Script sẽ tự tạo `.venv`, cài thư viện từ `requirements.txt`, rồi mở app desktop.

## 2) Cách gửi cho bạn của bạn
1. Gửi file nén mới được tạo từ thư mục `share`.
2. Bạn của bạn giải nén ra đường dẫn **không dấu, không quá dài**.
   - Ví dụ tốt: `D:\FACENET_SHARE`
   - Tránh để trong đường dẫn quá dài hoặc có ký tự lạ.

## 3) Cách chạy trên máy bạn của bạn
1. Cài Python 3.10+.
2. Khi cài Python nhớ tick **Add Python to PATH**.
3. Mở thư mục đã giải nén.
4. Double-click `RUN_APP.bat`.
5. Lần đầu chạy sẽ cài dependencies, có thể mất vài phút.
6. Sau khi cài xong, app desktop sẽ tự mở.

## 4) Dataset cho tab "Tìm kiếm trong Dataset"
- Đặt dataset vào thư mục: `dataset\`
- Cấu trúc khuyến nghị:
  - `dataset\TenNguoi1\img1.jpg`
  - `dataset\TenNguoi1\img2.jpg`
  - `dataset\TenNguoi2\img1.jpg`
- Sau khi chép dataset xong:
  1. Mở app.
  2. Vào tab **Tìm kiếm**.
  3. Bấm **Rebuild cache model đang chọn** hoặc **Rebuild cache tất cả model**.

## 5) Các thành phần đã đóng gói sẵn
- App desktop: `app\app_desktop.py` và các file hỗ trợ trong `app\`.
- Database/khuôn mặt đăng ký hiện tại: `app\faces_db`, `app\registered_faces`.
- Source xử lý chính: `src\`.
- Model/cache: `models\`.
- Báo cáo/ngưỡng/metric tham chiếu: `reports\`.
- Dependencies: `requirements.txt`.

## 6) Lưu ý lỗi thường gặp
- Lần chạy đầu cần mạng để cài thư viện Python.
- Nếu cài thư viện lỗi, mở lại `RUN_APP.bat` hoặc kiểm tra Python/PATH.
- Nếu camera không mở được, đóng app khác đang dùng camera rồi chạy lại.
- Nếu app bị Windows chặn, chọn **More info** > **Run anyway**.
