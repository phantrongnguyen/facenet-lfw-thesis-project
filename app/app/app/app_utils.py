"""
Các hàm tiện ích cho ứng dụng Demo Streamlit.
Wrap lại logic đã có sẵn trong src/ để phục vụ giao diện.
"""

import os
import time
import pickle
import numpy as np
from PIL import Image


def _get_embedding(image_path, model_name):
    """Lazy wrapper để import deepface chỉ khi cần."""
    from src.embedding_extractor import get_embedding
    return get_embedding(image_path, model_name)


def _cosine_similarity(a, b):
    """Lazy wrapper cho cosine similarity."""
    from src.similarity import cosine_similarity
    return cosine_similarity(a, b)


def _get_threshold():
    """Lazy load thresholds."""
    from src.config import THRESHOLD
    return THRESHOLD


def verify_faces(img_path1, img_path2, model_name):
    """
    So sánh 2 ảnh khuôn mặt, trả về kết quả nhận diện.
    
    Returns:
        dict: {
            similarity: float,
            is_match: bool,
            confidence: float (0-100%),
            threshold: float,
            inference_time: float (seconds)
        }
    """
    threshold = _get_threshold()[model_name]
    
    start = time.time()
    emb1 = _get_embedding(img_path1, model_name)
    emb2 = _get_embedding(img_path2, model_name)
    elapsed = time.time() - start
    
    sim = _cosine_similarity(emb1, emb2)
    is_match = sim >= threshold
    
    # Tính confidence: normalize similarity score thành %
    # Nếu match: confidence = mức vượt threshold
    # Nếu không match: confidence = mức dưới threshold
    if is_match:
        confidence = min(100.0, (sim / threshold) * 100.0)
    else:
        confidence = max(0.0, (sim / threshold) * 100.0)
    
    return {
        "similarity": float(sim),
        "is_match": bool(is_match),
        "confidence": round(confidence, 1),
        "threshold": threshold,
        "inference_time": round(elapsed, 2),
        "embedding_dim": len(emb1),
    }


def search_face(query_img_path, model_name, top_k=5):
    """
    Tìm kiếm khuôn mặt giống nhất trong database LFW đã precompute.
    
    Returns:
        list[dict]: Top-K kết quả [{path, similarity, name}, ...]
    """
    # Trích xuất embedding cho ảnh truy vấn
    query_emb = _get_embedding(query_img_path, model_name)
    
    # Load precomputed embeddings
    pkl_path = f"models/precomputed/{model_name}_embeddings.pkl"
    if not os.path.exists(pkl_path):
        return []
    
    with open(pkl_path, "rb") as f:
        db_embs = pickle.load(f)
    
    # So sánh với toàn bộ database
    results = []
    for img_path, emb in db_embs.items():
        if emb is not None and np.any(emb):  # bỏ qua vector 0
            sim = _cosine_similarity(query_emb, emb)

            # Trích xuất tên người từ đường dẫn
            name = os.path.basename(os.path.dirname(img_path))
            results.append({
                "path": img_path,
                "similarity": float(sim),
                "name": name,
            })
    
    # Sắp xếp theo similarity giảm dần
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results[:top_k]


def load_evaluation_metrics():
    """
    Trả về kết quả đánh giá tốt nhất (Lần 3) dưới dạng dict.
    """
    return {
        "Facenet": {
            "embedding_dim": 128,
            "accuracy": 0.9351,
            "precision": 0.9745,
            "recall": 1 - 0.1064,  # 1 - FRR
            "f1": 0.9323,
            "far": 0.0233,
            "frr": 0.1064,
            "threshold": 0.40,
            "inference_time": 0.6,
            "ram_mb": 362,
        },
        "Facenet512": {
            "embedding_dim": 512,
            "accuracy": 0.9396,
            "precision": 0.9523,
            "recall": 1 - 0.0744,  # 1 - FRR
            "f1": 0.9388,
            "far": 0.0464,
            "frr": 0.0744,
            "threshold": 0.35,
            "inference_time": 0.6,
            "ram_mb": 175,
        },
    }


def save_uploaded_file(uploaded_file, save_dir="app/temp_uploads"):
    """Lưu file upload tạm thời, trả về đường dẫn."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ===========================================================
#  FACE REGISTRATION & WEBCAM RECOGNITION
# ===========================================================

FACES_DB_DIR = "app/faces_db"


def get_faces_db_dir():
    """Trả về đường dẫn tới thư mục chứa khuôn mặt đã đăng ký."""
    os.makedirs(FACES_DB_DIR, exist_ok=True)
    return FACES_DB_DIR


def is_valid_name(name):
    """
    Kiểm tra tên hợp lệ: không dấu tiếng Việt, chỉ chứa chữ cái, số, khoảng trắng.
    Returns: (bool, str) - (hợp lệ?, thông báo lỗi)
    """
    import re

    if len(name.strip()) < 2:
        return False, "Tên phải có ít nhất 2 ký tự!"

    # Danh sách ký tự có dấu tiếng Việt
    vn = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    vn += vn.upper() + "Đ"
    for c in name:
        if c in vn:
            return False, "Tên không được chứa dấu tiếng Việt! Ví dụ: 'Gia Bao' thay vì 'Gia Bảo'."

    if not re.match(r'^[A-Za-z0-9 ]+$', name):
        return False, "Tên chỉ được chứa chữ cái (không dấu), số và khoảng trắng!"

    return True, "OK"


def is_name_duplicate(name):
    """Kiểm tra tên đã tồn tại trong database chưa."""
    db_dir = get_faces_db_dir()
    if not os.path.exists(db_dir):
        return False
    existing = [d.lower() for d in os.listdir(db_dir) 
                if os.path.isdir(os.path.join(db_dir, d))]
    return name.strip().lower() in existing


def register_face_image(name, uploaded_file):
    """
    Lưu ảnh khuôn mặt vào thư mục của người dùng.
    Returns: đường dẫn file đã lưu
    """
    db_dir = get_faces_db_dir()
    person_dir = os.path.join(db_dir, name.strip())
    os.makedirs(person_dir, exist_ok=True)

    # Đếm file hiện có để đặt tên không trùng
    existing = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    filename = f"face_{len(existing) + 1:03d}.jpg"
    filepath = os.path.join(person_dir, filename)

    # Lưu ảnh dưới dạng JPEG
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img.save(filepath, "JPEG", quality=95)

    return filepath


def list_registered_people():
    """
    Liệt kê tất cả người đã đăng ký khuôn mặt.
    Returns: list[dict] - [{name, photo_count, dir, first_photo}, ...]
    """
    db_dir = get_faces_db_dir()
    if not os.path.exists(db_dir):
        return []

    people = []
    for name in sorted(os.listdir(db_dir)):
        person_dir = os.path.join(db_dir, name)
        if not os.path.isdir(person_dir):
            continue
        photos = sorted([
            f for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if photos:
            people.append({
                "name": name,
                "photo_count": len(photos),
                "dir": person_dir,
                "first_photo": os.path.join(person_dir, photos[0]),
            })
    return people


def recognize_from_camera(camera_image_path, model_name):
    """
    So sánh ảnh chụp từ webcam với tất cả khuôn mặt đã đăng ký.
    Returns: list[dict] - kết quả sắp xếp theo similarity giảm dần
    """
    db_dir = get_faces_db_dir()
    if not os.path.exists(db_dir):
        return []

    # 1. Trích xuất embedding từ ảnh webcam
    try:
        query_emb = _get_embedding(camera_image_path, model_name)
    except Exception as e:
        return [{"error": str(e)}]

    threshold = _get_threshold()[model_name]
    results = []

    # 2. So sánh với từng người trong database
    for person_name in os.listdir(db_dir):
        person_dir = os.path.join(db_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        best_sim = -1.0
        best_photo = None

        for photo_file in os.listdir(person_dir):
            if not photo_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            photo_path = os.path.join(person_dir, photo_file)
            try:
                person_emb = _get_embedding(photo_path, model_name)
                sim = _cosine_similarity(query_emb, person_emb)
                if sim > best_sim:
                    best_sim = float(sim)
                    best_photo = photo_path
            except Exception:
                continue

        if best_photo is not None:
            results.append({
                "name": person_name,
                "similarity": best_sim,
                "is_match": best_sim >= threshold,
                "confidence": round(min(100.0, (best_sim / threshold) * 100.0), 1),
                "photo_path": best_photo,
            })

    # 3. Sắp xếp theo similarity giảm dần
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
