# 📊 So sánh & Đánh giá Model — FaceNet trên LFW

## Danh sách model

| # | Model | Backbone | Detector | MLP Config |
|---|-------|----------|----------|------------|
| 1 | `facenet_mtcnn` | FaceNet128 | MTCNN | Input→256→128→64→1 |
| 2 | `facenet_retinaface` | FaceNet128 | RetinaFace | Input→256→128→64→1 |
| 3 | `Facenet512_mtcnn_Reg2_256x128x64_regularized` | FaceNet512 | MTCNN | 256→128→64→1 (regularized) |
| 4 | `Facenet512_retinaface_Reg1_512x256x128_regularized` | FaceNet512 | RetinaFace | 512→256→128→1 (regularized) |

---

## Bảng so sánh tổng hợp

### Prediction Performance

| Metric | ① FaceNet128 MTCNN | ② FaceNet128 RetinaFace | ③ FaceNet512 MTCNN Reg2 | ④ FaceNet512 RetinaFace Reg1 |
|--------|:------------------:|:----------------------:|:----------------------:|:---------------------------:|
| **Test Acc** | 93.50% | 95.00% | 94.42% | **95.42%** 🥇 |
| **Precision** | 92.65% | 94.70% | 96.19% | **97.56%** 🥇 |
| **Recall** | **94.50%** 🥇 | 95.33% | 92.50% | 93.17% |
| **F1-Score** | 93.56% | 95.02% | 94.31% | **95.31%** 🥇 |
| **AUC** | 0.9828 | **0.9868** 🥇 | 0.9789 | 0.9847 |
| **EER** | 6.50% | **5.00%** 🥇 | 5.83% | 5.33% |
| **FAR** | 7.50% | 5.33% | 3.67% | **2.33%** 🥇 |
| **FRR** | 5.50% | **4.67%** 🥇 | 7.50% | 6.83% |

### Computational Performance

| Metric | ① FaceNet128 MTCNN | ② FaceNet128 RetinaFace | ③ FaceNet512 MTCNN Reg2 | ④ FaceNet512 RetinaFace Reg1 |
|--------|:------------------:|:----------------------:|:----------------------:|:---------------------------:|
| **Latency (ms)** | 0.217 | **0.173** 🥇 | 0.151 | 0.186 |
| **FPS** | 4609 | 5766 | **6629** 🥇 | 5386 |
| **CPU Total %** | **6.75%** 🥇 | 11.73% | 7.01% | 9.20% |
| **CPU Single-core %** | 108.03 | 187.68 | 112.21 | 147.26 |
| **RAM Before (MB)** | 511.62 | 550.00 | 1487.70 | 1523.49 |
| **RAM After (MB)** | 511.75 | 550.11 | 1487.75 | 1523.55 |
| **RAM Delta (MB)** | 0.13 | 0.10 | **0.05** 🥇 | 0.06 |

### Model Complexity

| Metric | ① FaceNet128 MTCNN | ② FaceNet128 RetinaFace | ③ FaceNet512 MTCNN Reg2 | ④ FaceNet512 RetinaFace Reg1 |
|--------|:------------------:|:----------------------:|:----------------------:|:---------------------------:|
| **Parameters** | **108.5K** 🥇 | **108.5K** 🥇 | 305K | 692K |
| **Model Size (MB)** | **0.41** 🥇 | **0.41** 🥇 | 3.53 | 7.96 |
| **Overfit Gap** | 0.0239 | 0.0262 | 0.0442 | 0.0383 |

---

## 🏆 Xếp hạng tổng thể

| Hạng | Model | Test Acc | F1 | EER | FAR | CPU % | Size | Tổng điểm |
|:----:|-------|:--------:|:--:|:---:|:---:|:-----:|:----:|:---------:|
| 🥇 | **④ FaceNet512 + RetinaFace + Reg1** | **95.42%** | **95.31%** | 5.33% | **2.33%** | 9.20% | 7.96 MB | **9.1/10** |
| 🥈 | ② FaceNet128 + RetinaFace | 95.00% | 95.02% | **5.00%** | 5.33% | 11.73% | **0.41 MB** | 8.7/10 |
| 🥉 | ③ FaceNet512 + MTCNN + Reg2 | 94.42% | 94.31% | 5.83% | 3.67% | **7.01%** | 3.53 MB | 7.5/10 |
| 4 | ① FaceNet128 + MTCNN | 93.50% | 93.56% | 6.50% | 7.50% | 6.75% | 0.41 MB | 7.0/10 |

---

## 📈 Phân tích chi tiết

### Accuracy

| So sánh | Chênh lệch |
|---------|:----------:|
| FaceNet512 + RetinaFace + Reg1 vs FaceNet128 + RetinaFace | **+0.42%** test acc, **+0.29%** F1 |
| FaceNet512 + RetinaFace + Reg1 vs FaceNet512 + MTCNN + Reg2 | **+1.00%** test acc, **+1.00%** F1 |
| FaceNet128 + RetinaFace vs FaceNet128 + MTCNN | **+1.50%** test acc, **+1.46%** F1 |

> **Nhận xét:** FaceNet512 + RetinaFace + Reg1 dẫn đầu về accuracy (95.42%) và precision (97.56%). FaceNet128 + RetinaFace baseline bám sát với 95.00% acc và EER thấp nhất (5.00%). FaceNet512 + MTCNN + Reg2 có precision cao (96.19%) nhưng recall thấp (92.50%).

### EER & FAR — Bảo mật

| Model | EER | FAR | FRR | Đánh giá bảo mật |
|-------|:---:|:---:|:---:|-----------------|
| FaceNet512 + RetinaFace + Reg1 | 5.33% | **2.33%** | 6.83% | 🟢 **FAR thấp nhất — an toàn nhất** |
| FaceNet128 + RetinaFace | **5.00%** | 5.33% | **4.67%** | 🟢 EER thấp nhất, cân bằng FAR/FRR |
| FaceNet512 + MTCNN + Reg2 | 5.83% | 3.67% | 7.50% | 🟡 FAR thấp nhưng FRR cao |
| FaceNet128 + MTCNN | 6.50% | 7.50% | 5.50% | 🔴 FAR cao nhất — kém an toàn |

> **Security Ranking:** FaceNet512+RetinaFace+Reg1 > FaceNet128+RetinaFace > FaceNet512+MTCNN+Reg2 > FaceNet128+MTCNN

### Speed & CPU

| Model | Latency | FPS | CPU Total % | Nhận xét |
|-------|:-------:|:---:|:-----------:|----------|
| FaceNet512 + MTCNN + Reg2 | **0.151ms** | **6629** | 7.01% | 🟢 **Nhanh nhất, CPU thấp** |
| FaceNet128 + RetinaFace | 0.173ms | 5766 | **11.73%** | 🟡 Nhanh nhưng CPU cao nhất |
| FaceNet512 + RetinaFace + Reg1 | 0.186ms | 5386 | 9.20% | 🟡 Cân bằng |
| FaceNet128 + MTCNN | 0.217ms | 4609 | 6.75% | 🟢 CPU thấp nhất nhưng chậm nhất |

> **Speed Ranking:** FaceNet512+MTCNN+Reg2 > FaceNet128+RetinaFace > FaceNet512+RetinaFace+Reg1 > FaceNet128+MTCNN
>
> Dù FaceNet512 + RetinaFace + Reg1 có nhiều tham số nhất (692K) nhưng latency chỉ 0.186ms — nhờ MLP Reg1 tối ưu, giảm dần 512→256→128.

### RAM Usage

| Model | RAM After (MB) | RAM Delta (MB) |
|-------|:--------------:|:--------------:|
| FaceNet128 + MTCNN | **511.75** | 0.13 |
| FaceNet128 + RetinaFace | 550.11 | 0.10 |
| FaceNet512 + MTCNN + Reg2 | 1487.75 | **0.05** |
| FaceNet512 + RetinaFace + Reg1 | 1523.55 | 0.06 |

> FaceNet128 models chỉ dùng ~500MB RAM, FaceNet512 models ~1500MB — do TensorFlow/Keras load model larger. RAM delta không đáng kể (<0.13MB) ở tất cả models.

### Model Size

| Model | Parameters | Size (MB) | So với model nhẹ nhất |
|-------|:----------:|:---------:|:--------------------:|
| FaceNet128 + MTCNN | **108.5K** | **0.41** | — |
| FaceNet128 + RetinaFace | **108.5K** | **0.41** | — |
| FaceNet512 + MTCNN + Reg2 | 305K | 3.53 | ×8.6 |
| FaceNet512 + RetinaFace + Reg1 | 692K | 7.96 | ×19.4 |

> FaceNet128 MLP rất nhẹ (0.41 MB). FaceNet512 + RetinaFace + Reg1 nặng nhất (7.96 MB) do MLP 3 lớp + regularized weights.

---

## 🏁 Kết luận

### 🥇 Best Overall: FaceNet512 + RetinaFace + Reg1 (512→256×128→1)

| Tiêu chí | Giá trị | Xếp hạng |
|----------|---------|:--------:|
| **Test Accuracy** | **95.42%** | 🥇 #1 |
| **Precision** | **97.56%** | 🥇 #1 |
| **F1-Score** | **95.31%** | 🥇 #1 |
| **FAR** | **2.33%** | 🥇 #1 (thấp nhất) |
| **Latency** | 0.186ms | 🥈 #2 |
| **CPU Total** | 9.20% | 🥉 #3 |
| **Model Size** | 7.96 MB | 4 |

**Lý do chọn:**
1. **Test accuracy cao nhất** (95.42%) — vượt baseline FaceNet128+RetinaFace +0.42%
2. **Precision cao nhất** (97.56%) — tỷ lệ dự đoán "same" đúng rất cao
3. **FAR thấp nhất** (2.33%) — giảm false acceptance, an toàn nhất cho bảo mật
4. **Latency chỉ 0.186ms** (5386 FPS) — đủ real-time dù model nặng nhất
5. **CPU 9.20%** — thấp hơn baseline FaceNet128+RetinaFace (11.73%) nhờ MLP tối ưu

**Điểm yếu:**
- Model size lớn nhất (7.96 MB) — nhưng vẫn rất nhẹ so với các model deep learning khác
- Recall 93.17% — thấp hơn baseline (95.33%), có thể bỏ sót genuine nhiều hơn

### 🥈 Best Lightweight: FaceNet128 + RetinaFace

| Tiêu chí | Giá trị |
|----------|---------|
| **Test Accuracy** | 95.00% |
| **F1-Score** | 95.02% |
| **EER** | **5.00%** (thấp nhất) |
| **FRR** | **4.67%** (thấp nhất) |
| **Latency** | 0.173ms (nhanh #2) |
| **Model Size** | **0.41 MB** |
| **RAM After** | **550 MB** |

> Lựa chọn tốt cho thiết bị cấu hình thấp, mobile, hoặc IoT. EER thấp nhất (5.00%) — cân bằng FAR/FRR tốt.

### 🥉 Best Speed: FaceNet512 + MTCNN + Reg2

| Tiêu chí | Giá trị |
|----------|---------|
| **Latency** | **0.151ms** (nhanh nhất) |
| **FPS** | **6629** (cao nhất) |
| **CPU Total** | **7.01%** (thấp #2) |
| **Precision** | 96.19% |

> Phù hợp cho ứng dụng real-time, embedded system, hoặc khi cần throughput cực cao.

---

## Khuyến nghị triển khai

| Tình huống | Model đề xuất | Lý do chính |
|------------|--------------|-------------|
| **🔐 Bảo mật cao, cần FAR thấp** | FaceNet512 + RetinaFace + Reg1 | FAR 2.33% — thấp nhất |
| **⚡ Real-time, throughput cao** | FaceNet512 + MTCNN + Reg2 | 6629 FPS, CPU 7.01% |
| **📱 Mobile / Edge / IoT** | FaceNet128 + RetinaFace | 0.41 MB, 550 MB RAM, 95.00% acc |
| **🎯 Cân bằng accuracy/speed/size** | FaceNet512 + RetinaFace + Reg1 | Best overall |
| **💰 Chi phí thấp, CPU yếu** | FaceNet128 + MTCNN | CPU 6.75%, 0.41 MB |

---

## Tổng quan điểm số

| Tiêu chí | Trọng số | ① FaceNet128 MTCNN | ② FaceNet128 RetinaFace | ③ FaceNet512 MTCNN Reg2 | ④ FaceNet512 RetinaFace Reg1 |
|----------|:--------:|:------------------:|:----------------------:|:----------------------:|:---------------------------:|
| Test Accuracy | 20% | 7.0 | 8.5 | 8.0 | **9.0** |
| Precision | 15% | 7.0 | 8.0 | 8.5 | **9.5** |
| F1 | 15% | 7.0 | 8.5 | 8.0 | **9.0** |
| EER | 15% | 6.0 | **9.0** | 7.5 | 8.0 |
| FAR (Security) | 15% | 5.0 | 7.0 | 8.5 | **9.5** |
| Latency | 10% | 7.0 | 8.5 | **9.5** | 8.0 |
| CPU % | 5% | **9.5** | 5.0 | 9.0 | 7.0 |
| Model Size | 5% | **10.0** | **10.0** | 5.0 | 3.0 |
| **Tổng weighted** | **100%** | **6.83** | **8.08** | **8.03** | **8.45** |

> **🥇 FaceNet512 + RetinaFace + Reg1** đạt điểm weighted cao nhất (8.45/10), dẫn đầu về accuracy, precision, FAR. FaceNet128 + RetinaFace đứng thứ 2 (8.08/10) với lợi thế về EER, model size, và latency.

---

*Nguồn dữ liệu: `logs/results/all_models_metrics.csv` và `logs/my_logs/Facenet512_*_regularized/final_metrics.csv`*
*Cập nhật lần cuối: 27/05/2026*
