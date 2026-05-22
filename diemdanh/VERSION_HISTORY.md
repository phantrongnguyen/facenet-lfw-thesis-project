# Version History

## 0.0.10-dev

- Camera không còn tự bật; người dùng phải bấm `Bật camera` và có thể tắt lại để bảo vệ riêng tư.
- Layout đăng nhập/đăng ký sinh viên chia 2 vùng: form bên trái và camera panel bên phải.
- Preview ảnh sau khi chụp giữ cùng khung hiển thị với camera để tránh cảm giác ảnh bị thu nhỏ.
- Cổng sinh viên tách tab `Test nhận diện`, `Điểm danh`, và `Đổi ảnh`.
- Test nhận diện hiển thị thông tin sinh viên, confidence và ảnh đăng ký nếu database có lưu ảnh gốc.
- Thêm workflow sinh viên gửi yêu cầu đổi ảnh đăng ký; giảng viên/Admin xem lịch sử, duyệt hoặc từ chối.
- Backend bổ sung bảng `FacePhotoChangeRequest`, router `/api/face-changes/*`, và helper kích hoạt embedding mới sau khi duyệt.
- Cập nhật frontend/backend metadata lên version `0.0.10-dev`.

## 0.0.8

- Siết mã sinh viên theo rule hiện tại `SV000` - `SV999` ở đăng ký và CRUD sinh viên.
- Giữ mật khẩu sinh viên tối thiểu 6 ký tự và chuẩn hóa lỗi đăng nhập tiếng Việt.
- Thêm endpoint `POST /api/face/test` cho sinh viên tự test nhận diện khuôn mặt.
- Test nhận diện lưu ảnh vào `storage/face_tests`, trả confidence/threshold/model và không tạo bản ghi điểm danh.
- Bổ sung card `Test nhận diện` trong cổng sinh viên để kiểm tra trước khi điểm danh thật.
- Cập nhật frontend/backend metadata lên version `0.0.8`.

## 0.0.7

- Siết vòng đời buổi học theo luồng `scheduled -> open -> closed`.
- Chặn chỉnh sửa buổi đã đóng, chặn mở lại buổi đã đóng và tự đóng buổi đang mở cùng lớp khi mở buổi mới.
- Chuẩn hóa thông báo điểm danh tiếng Việt cho buổi chưa mở/đã đóng, điểm danh trùng và cần giảng viên duyệt.
- Cập nhật response điểm danh thành payload rõ ràng gồm trạng thái, nhãn trạng thái, confidence, threshold và metadata model.
- Bổ sung báo cáo JSON/CSV theo buổi học với thống kê đủ sĩ số, có mặt, trễ, vắng và cần duyệt.
- Cập nhật frontend/backend metadata lên version `0.0.7`.

## 0.0.6

- Benchmark 6 pipeline embedding từ `models/precomputed` trên 10.000 cặp ảnh LFW.
- Chọn `Facenet512_retinaface` làm pipeline nhận diện chính cho web.
- Cập nhật threshold nhận diện mặc định sang `0.407157` theo benchmark.
- Giữ luồng production: đăng ký tạo embedding một lần, lưu database, cache RAM và điểm danh bằng so sánh vector.
- Bổ sung metadata `model_key`, `model_label`, `threshold` trong phản hồi đăng ký/reload/điểm danh.
- Cập nhật frontend/backend metadata lên version `0.0.6`.

## 0.0.5

- Kiểm tra lại `0.0.4`: luồng đăng ký khuôn mặt và điểm danh camera đã đúng demo.
- Chuyển CRUD Admin từ `localStorage` sang API backend/database thật cho sinh viên, lớp học và buổi học.
- Bổ sung sửa/xóa dữ liệu ở backend, kiểm tra trùng mã và chặn xóa lớp/buổi học khi còn dữ liệu liên quan.
- Cập nhật giao diện Admin sang nhãn `Dữ liệu database` thay cho `CRUD demo`.
- Cập nhật frontend/backend metadata lên version `0.0.5`.

## 0.0.4

- Căn giữa màn hình đăng nhập/đăng ký và bỏ khung giới thiệu lớn ở trang đầu.
- Thêm đăng ký sinh viên thật bằng mã thẻ sinh viên, họ tên, mật khẩu và ảnh khuôn mặt.
- Mật khẩu được mã hóa ở backend; ảnh khuôn mặt đăng ký được lưu trong `storage/faces/<student_id>/`.
- Điểm danh sinh viên chuyển sang chụp trực tiếp bằng camera, không còn chọn file ở bước điểm danh.
- Cập nhật frontend/backend metadata lên version `0.0.4`.

## 0.0.3

- Chuyển giao diện chính sang tiếng Việt có dấu, ưu tiên UTF-8 để tránh lỗi ký tự `?`.
- Thêm các tab quản trị: Tổng quan, Sinh viên, Lớp học, Buổi học và Báo cáo.
- Thêm CRUD demo cho sinh viên, lớp học và buổi học bằng state/localStorage ở frontend.
- Giữ Cổng sinh viên tối giản: xem thẻ sinh viên, buổi học đang mở và điểm danh khuôn mặt.
- Cập nhật metadata frontend/backend lên version `0.0.3`.

## 0.0.2

- Changed the interface to a light blue education-tech theme.
- Added Student and Teacher/Admin login choices.
- Added virtual student card registration for demo use.
- Added a minimal Student Portal focused on face check-in.
- Added a Teacher/Admin Portal shell for dashboard, CRUD, sessions, and reports.
- Updated frontend and backend metadata to version `0.0.2`.

## 0.0.1

- Rebuilt the project from a Streamlit prototype into React + Vite and FastAPI.
- Added the initial database model structure for users, students, classes, sessions, and attendance records.
- Added the initial FaceService using RAM-cached embeddings and vectorized cosine similarity.
- Verified that the frontend and backend could run as a first technical MVP.
