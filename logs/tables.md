# Bảng Đánh Giá Hiệu Năng Các Mô Hình

| Model | Backbone | Detector | Train Acc | Test Acc | Precision | Recall | Specificity | F1-score | AUC | FAR ↓ | FRR ↓ | EER ↓ | Balanced Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facenet_mtcnn | FaceNet128 | MTCNN | 0.9589 | 0.9350 | 0.9265 | 0.9450 | 0.9250 | 0.9356 | 0.9828 | 0.0750 | 0.0550 | 0.0650 | 0.9350 |
| facenet_retinaface | FaceNet128 | RetinaFace | 0.9762 | 0.9500 | 0.9470 | 0.9530 | 0.9470 | 0.9500 | 0.9892 | 0.0530 | 0.0470 | 0.0490 | 0.9500 |
| Facenet512_mtcnn_Reg2 | FaceNet512 | MTCNN | 0.9883 | 0.9442 | 0.9619 | 0.9250 | 0.9633 | 0.9431 | 0.9877 | 0.0367 | 0.0750 | 0.0550 | 0.9442 |
| Facenet512_retinaface_Reg1 | FaceNet512 | RetinaFace | 0.9925 | 0.9542 | 0.9756 | 0.9333 | 0.9750 | 0.9540 | 0.9914 | 0.0250 | 0.0667 | 0.0460 | 0.9542 |

---

# Bảng Đánh Giá Hiệu Năng Phần Cứng

| Model | Latency (ms) ↓ | FPS ↑ | Throughput ↑ | CPU Time (s) ↓ | CPU Usage (%) ↓ | RAM Before (MB) | RAM After (MB) | RAM Delta (MB) ↓ | Parameters | Model Size (MB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| facenet_mtcnn | 0.217 | 4609.34 | 4609.34 | 0.2812 | 6.75 | 511.62 | 511.75 | 0.13 | 108,545 | 0.41 |
| facenet_retinaface | 0.240 | 4166.67 | 4166.67 | 0.3015 | 7.92 | 512.10 | 512.34 | 0.24 | 108,545 | 0.41 |
| Facenet512_mtcnn_Reg2 | 0.151 | 6629.20 | 6629.20 | 0.1941 | 5.84 | 518.40 | 520.12 | 1.72 | 305,153 | 3.53 |
| Facenet512_retinaface_Reg1 | 0.186 | 5385.63 | 5385.63 | 0.2256 | 6.43 | 525.84 | 528.99 | 3.15 | 692,225 | 7.96 |

---

# Nhận Xét Tổng Quan

| Tiêu chí | Mô hình tốt nhất |
|---|---|
| Accuracy cao nhất | FaceNet512 + RetinaFace |
| Precision cao nhất | FaceNet512 + RetinaFace |
| AUC cao nhất | FaceNet512 + RetinaFace |
| EER thấp nhất | FaceNet512 + RetinaFace |
| FPS cao nhất | FaceNet512 + MTCNN |
| Latency thấp nhất | FaceNet512 + MTCNN |
| RAM thấp nhất | FaceNet128 |
| Model nhỏ nhất | FaceNet128 |
| Tổng thể tốt nhất | FaceNet512 + RetinaFace |