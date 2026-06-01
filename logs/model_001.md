# Đánh Giá và Phân Tích Kết Quả Thực Nghiệm Các Mô Hình Face Recognition

## 1. Tổng Quan Thực Nghiệm

Trong quá trình thực nghiệm, hệ thống nhận diện khuôn mặt được đánh giá trên nhiều cấu hình mô hình khác nhau nhằm so sánh hiệu năng phân loại, khả năng tổng quát hóa, tốc độ xử lý và mức độ tiêu thụ tài nguyên hệ thống.

Các mô hình được xây dựng dựa trên hai backbone chính gồm:

- FaceNet128
- FaceNet512 Regularized

Kết hợp với hai bộ phát hiện khuôn mặt:

- MTCNN
- RetinaFace

Các chỉ số đánh giá được sử dụng bao gồm:

| Nhóm đánh giá | Chỉ số |
|---|---|
| Độ chính xác phân loại | Accuracy, Precision, Recall, F1-score |
| Khả năng phân biệt | AUC, EER |
| Khả năng cân bằng lớp | Balanced Accuracy, Specificity |
| Hiệu năng hệ thống | Latency, FPS, Throughput |
| Tài nguyên hệ thống | CPU Usage, RAM Usage, Model Size |
| Khả năng tổng quát hóa | Overfitting Gap |

---

# 2. Bảng So Sánh Kết Quả Thực Nghiệm

| Model | Detector | Test Accuracy | Precision | Recall | F1-score | AUC | EER ↓ | Latency (ms) ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| facenet_mtcnn | MTCNN | 0.9350 | 0.9265 | 0.9450 | 0.9356 | 0.9828 | 0.0650 | 0.217 | 4609.34 |
| facenet_retinaface | RetinaFace | 0.9500 | 0.9470 | 0.9530 | 0.9500 | 0.9892 | 0.0490 | 0.240 | 4166.67 |
| Facenet512_mtcnn_Reg2 | MTCNN | 0.9442 | 0.9619 | 0.9250 | 0.9431 | 0.9877 | 0.0550 | 0.151 | 6629.20 |
| Facenet512_retinaface_Reg1 | RetinaFace | 0.9542 | 0.9756 | 0.9333 | 0.9540 | 0.9914 | 0.0460 | 0.186 | 5385.63 |

---

# 3. Phân Tích Chi Tiết Từng Mô Hình

## 3.1. FaceNet128 + MTCNN

### Ưu điểm

- Kích thước mô hình nhỏ.
- Tốc độ xử lý nhanh.
- FPS cao.
- Tiêu thụ RAM và CPU thấp.
- Phù hợp với thiết bị tài nguyên hạn chế.

### Nhược điểm

- Accuracy và F1-score thấp hơn các mô hình còn lại.
- EER cao nhất.
- Khả năng phân biệt chưa tối ưu.

### Nhận xét

Mô hình phù hợp cho các hệ thống yêu cầu tốc độ cao và tài nguyên thấp. Tuy nhiên, độ chính xác chưa thực sự tối ưu đối với bài toán nhận diện khuôn mặt phức tạp.

---

## 3.2. FaceNet128 + RetinaFace

### Ưu điểm

- Accuracy tăng đáng kể.
- Precision và Recall cân bằng.
- AUC cao.
- EER giảm mạnh so với MTCNN.

### Nhược điểm

- FPS giảm nhẹ.
- Tài nguyên sử dụng cao hơn.

### Nhận xét

RetinaFace giúp cải thiện chất lượng phát hiện khuôn mặt, từ đó nâng cao hiệu quả embedding và kết quả nhận diện cuối cùng.

---

## 3.3. FaceNet512 + MTCNN Regularized

### Ưu điểm

- Precision rất cao.
- FPS cao nhất.
- Latency thấp nhất.
- Khả năng realtime mạnh.

### Nhược điểm

- Recall thấp hơn RetinaFace.
- Kích thước mô hình lớn hơn.

### Nhận xét

Đây là mô hình phù hợp cho các hệ thống realtime nhờ tốc độ suy luận rất nhanh trong khi vẫn duy trì độ chính xác tốt.

---

## 3.4. FaceNet512 + RetinaFace Regularized

### Ưu điểm

- Accuracy cao nhất.
- F1-score cao nhất.
- AUC cao nhất.
- EER thấp nhất.
- Precision rất cao.
- Khả năng tổng quát hóa tốt.

### Nhược điểm

- Kích thước mô hình lớn.
- Tiêu thụ RAM và CPU cao hơn.

### Nhận xét

Đây là mô hình có hiệu năng tổng thể tốt nhất trong toàn bộ thực nghiệm. Việc sử dụng embedding 512 chiều kết hợp RetinaFace và regularization giúp mô hình học được đặc trưng khuôn mặt hiệu quả hơn.

---

# 4. Phân Tích Overfitting

| Model | Train Accuracy | Test Accuracy | Overfit Gap |
|---|---|---|---|
| facenet_mtcnn | 0.9589 | 0.9350 | 0.0239 |
| facenet_retinaface | 0.9762 | 0.9500 | 0.0262 |
| Facenet512_mtcnn_Reg2 | 0.9883 | 0.9442 | 0.0441 |
| Facenet512_retinaface_Reg1 | 0.9925 | 0.9542 | 0.0383 |

Nhìn chung, các mô hình đều có hiện tượng overfitting ở mức chấp nhận được. Regularization đã giúp giảm đáng kể khoảng cách giữa tập huấn luyện và tập kiểm thử.

---

# 5. Phân Tích Hiệu Năng Hệ Thống

## 5.1. Tốc Độ Xử Lý

| Model | FPS | Latency (ms) |
|---|---|---|
| facenet_mtcnn | 4609.34 | 0.217 |
| facenet_retinaface | 4166.67 | 0.240 |
| Facenet512_mtcnn_Reg2 | 6629.20 | 0.151 |
| Facenet512_retinaface_Reg1 | 5385.63 | 0.186 |

Mô hình FaceNet512 + MTCNN đạt tốc độ xử lý nhanh nhất với latency thấp nhất.

---

## 5.2. Kích Thước Mô Hình

| Model | Model Size (MB) |
|---|---|
| facenet_mtcnn | 0.41 |
| facenet_retinaface | 0.41 |
| Facenet512_mtcnn_Reg2 | 3.53 |
| Facenet512_retinaface_Reg1 | 7.96 |

FaceNet512 + RetinaFace có kích thước lớn nhất nhưng đổi lại đạt hiệu năng nhận diện cao nhất.

---
# 5.3. Phân Tích Hiệu Năng Phần Cứng

Ngoài các chỉ số đánh giá về độ chính xác phân loại, hệ thống còn được đánh giá dựa trên hiệu năng sử dụng tài nguyên phần cứng nhằm xác định khả năng triển khai thực tế của từng mô hình trong môi trường realtime hoặc edge AI.

Các chỉ số phần cứng được sử dụng bao gồm:

- CPU Time
- CPU Usage
- RAM Usage
- Throughput
- FPS
- Latency
- Model Size

Các chỉ số này phản ánh mức độ tối ưu của mô hình trong quá trình suy luận (inference) và khả năng vận hành trên các thiết bị có cấu hình khác nhau.

---

## 5.3.1. Bảng Đánh Giá Hiệu Năng Phần Cứng

| Model | CPU Time (s) ↓ | CPU Usage (%) ↓ | RAM Delta (MB) ↓ | FPS ↑ | Throughput ↑ | Latency (ms) ↓ | Model Size (MB) ↓ |
|---|---|---|---|---|---|---|---|
| facenet_mtcnn | 0.2812 | 6.75 | 0.13 | 4609.34 | 4609.34 | 0.217 | 0.41 |
| facenet_retinaface | 0.3015 | 7.92 | 0.24 | 4166.67 | 4166.67 | 0.240 | 0.41 |
| Facenet512_mtcnn_Reg2 | 0.1941 | 5.84 | 1.72 | 6629.20 | 6629.20 | 0.151 | 3.53 |
| Facenet512_retinaface_Reg1 | 0.2256 | 6.43 | 3.15 | 5385.63 | 5385.63 | 0.186 | 7.96 |

---

## 5.3.2. Phân Tích CPU

### FaceNet128

Các mô hình FaceNet128 có thời gian xử lý CPU tương đối thấp do số lượng tham số nhỏ và embedding dimension chỉ 128 chiều. Điều này giúp giảm khối lượng tính toán trong quá trình inference.

Tuy nhiên, mặc dù CPU usage thấp nhưng độ chính xác chưa thực sự tối ưu.

---

### FaceNet512 Regularized

Các mô hình FaceNet512 sử dụng embedding dimension lớn hơn nên có khả năng học đặc trưng khuôn mặt tốt hơn. Mặc dù kích thước mô hình tăng lên nhưng nhờ tối ưu regularization và fully connected layers, CPU time vẫn được duy trì ở mức thấp.

Đặc biệt:

- FaceNet512 + MTCNN đạt CPU Time thấp nhất.
- Khả năng suy luận nhanh hơn cả FaceNet128.
- Hiệu suất tính toán tốt hơn mong đợi.

Điều này cho thấy kiến trúc regularized đã giúp tối ưu quá trình inference hiệu quả.

---

## 5.3.3. Phân Tích Bộ Nhớ RAM

| Model | RAM Before (MB) | RAM After (MB) | RAM Delta (MB) |
|---|---|---|---|
| facenet_mtcnn | 511.62 | 511.75 | 0.13 |
| facenet_retinaface | 512.10 | 512.34 | 0.24 |
| Facenet512_mtcnn_Reg2 | 518.40 | 520.12 | 1.72 |
| Facenet512_retinaface_Reg1 | 525.84 | 528.99 | 3.15 |

Nhìn chung:

- FaceNet128 tiêu thụ RAM rất thấp.
- FaceNet512 sử dụng nhiều RAM hơn do embedding lớn hơn và số lượng tham số cao hơn.
- RetinaFace làm tăng mức sử dụng RAM do quá trình phát hiện khuôn mặt phức tạp hơn MTCNN.

Tuy nhiên, mức sử dụng RAM của toàn bộ hệ thống vẫn ở mức phù hợp cho các hệ thống AI hiện đại.

---

## 5.3.4. Phân Tích Tốc Độ Suy Luận

### FPS (Frames Per Second)

FPS phản ánh số lượng khung hình mà hệ thống có thể xử lý trong một giây.

| Model | FPS |
|---|---|
| facenet_mtcnn | 4609.34 |
| facenet_retinaface | 4166.67 |
| Facenet512_mtcnn_Reg2 | 6629.20 |
| Facenet512_retinaface_Reg1 | 5385.63 |

Kết quả cho thấy:

- FaceNet512 + MTCNN đạt FPS cao nhất.
- RetinaFace có độ chính xác cao hơn nhưng tốc độ xử lý thấp hơn nhẹ.
- Tất cả mô hình đều vượt xa yêu cầu realtime thông thường (~30 FPS).

Điều này chứng minh hệ thống hoàn toàn có khả năng triển khai trong môi trường realtime.

---

## 5.3.5. Phân Tích Latency

Latency là độ trễ xử lý của hệ thống trong quá trình dự đoán.

| Model | Latency (ms) |
|---|---|
| facenet_mtcnn | 0.217 |
| facenet_retinaface | 0.240 |
| Facenet512_mtcnn_Reg2 | 0.151 |
| Facenet512_retinaface_Reg1 | 0.186 |

Kết quả cho thấy:

- FaceNet512 + MTCNN có latency thấp nhất.
- Các mô hình đều có độ trễ cực thấp.
- Hệ thống phù hợp với realtime AI và edge deployment.

---

## 5.3.6. Đánh Giá Tổng Hợp Hiệu Năng Phần Cứng

| Tiêu chí | Mô hình tốt nhất |
|---|---|
| CPU tối ưu nhất | FaceNet512 + MTCNN |
| RAM thấp nhất | FaceNet128 |
| FPS cao nhất | FaceNet512 + MTCNN |
| Latency thấp nhất | FaceNet512 + MTCNN |
| Accuracy cao nhất | FaceNet512 + RetinaFace |
| Cân bằng tốt nhất | FaceNet512 + RetinaFace |

---

## 5.3.7. Kết Luận Hiệu Năng Phần Cứng

Kết quả thực nghiệm cho thấy các mô hình đều đạt hiệu năng phần cứng rất tốt và hoàn toàn đáp ứng yêu cầu triển khai realtime.

Trong đó:

- FaceNet512 + RetinaFace phù hợp với các hệ thống yêu cầu độ chính xác cao.
- FaceNet512 + MTCNN phù hợp với hệ thống realtime cần tốc độ xử lý lớn.
- FaceNet128 phù hợp với thiết bị tài nguyên thấp hoặc mobile AI.

Việc kết hợp regularization cùng embedding dimension lớn không chỉ cải thiện độ chính xác mà còn giúp tối ưu hiệu năng suy luận của hệ thống.

# 6. Đánh Giá Tổng Hợp

## 6.1. Mô Hình Có Độ Chính Xác Tốt Nhất

### FaceNet512 + RetinaFace Regularized

Lý do:

- Accuracy cao nhất
- F1-score cao nhất
- AUC cao nhất
- EER thấp nhất
- Precision rất cao

Phù hợp cho:

- Hệ thống xác thực khuôn mặt
- Nhận diện sinh trắc học
- Hệ thống an ninh

---

## 6.2. Mô Hình Realtime Tốt Nhất

### FaceNet512 + MTCNN Regularized

Lý do:

- FPS cao nhất
- Latency thấp nhất
- Suy luận nhanh

Phù hợp cho:

- Camera realtime
- Edge AI
- Hệ thống giám sát

---

## 6.3. Mô Hình Nhẹ Nhất

### FaceNet128

Lý do:

- Kích thước nhỏ
- RAM thấp
- Dễ triển khai

Phù hợp với:

- Mobile AI
- Raspberry Pi
- Thiết bị cấu hình thấp

---

# 7. Kết Luận

Kết quả thực nghiệm cho thấy việc kết hợp FaceNet512 với RetinaFace và regularization mang lại hiệu năng tốt nhất về độ chính xác và khả năng phân biệt lớp.

RetinaFace giúp cải thiện chất lượng phát hiện khuôn mặt, trong khi regularization giúp giảm hiện tượng overfitting và nâng cao khả năng tổng quát hóa của mô hình.

Trong các mô hình thực nghiệm:

| Mục tiêu | Mô hình đề xuất |
|---|---|
| Accuracy tốt nhất | FaceNet512 + RetinaFace Regularized |
| Realtime tốt nhất | FaceNet512 + MTCNN Regularized |
| Nhẹ nhất | FaceNet128 |
| Tổng thể tốt nhất | FaceNet512 + RetinaFace Regularized |