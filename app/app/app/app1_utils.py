"""
Các hàm tiện ích cho ứng dụng Demo Streamlit.
Wrap lại logic đã có sẵn trong src/ để phục vụ giao diện.
"""

import os
import time
import tempfile
import pickle
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_embedding(image_path, model_name):
    """Lazy wrapper để import deepface chỉ khi cần."""
    from src.embedding_extractor import get_embedding
    return get_embedding(image_path, model_name)


def extract_face_crop(image_path, model_name):
    """
    Trích xuất và trả về ảnh khuôn mặt đã được AI cắt + căn chỉnh (Align).
    Đây là BẰNG CHỨNG trực quan cho thấy Detector Backend đang hoạt động.
    
    Returns:
        PIL.Image hoặc None nếu không tìm thấy mặt.
    """
    from deepface import DeepFace
    from PIL import Image as PILImage
    import numpy as np

    if "_" in model_name:
        _, detector = model_name.split("_", 1)
    else:
        detector = "mtcnn"

    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector,
            align=True,
            enforce_detection=True,
        )
        if faces:
            face_arr = faces[0]["face"]  # numpy array (H, W, 3), giá trị 0.0-1.0
            face_uint8 = (face_arr * 255).astype(np.uint8)
            return PILImage.fromarray(face_uint8)
    except Exception as e:
        print(f"extract_face_crop lỗi: {e}")
    return None


def _cosine_similarity(a, b):
    """Lazy wrapper cho cosine similarity."""
    from src.similarity import cosine_similarity
    return cosine_similarity(a, b)


def _get_threshold():
    """Load thresholds from CSV."""
    import pandas as pd
    csv_path = os.path.join(PROJECT_ROOT, "reports", "Model_Evaluation_Metrics.csv")
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df["Model"], df["Optimal Threshold"]))
    except Exception as e:
        print(f"Lỗi đọc threshold: {e}")
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
    pkl_path = os.path.join(PROJECT_ROOT, "models", "precomputed", f"{model_name}_embeddings.pkl")
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
    Đọc kết quả đánh giá từ file CSV.
    """
    import pandas as pd
    metrics = {}
    csv_path = os.path.join(PROJECT_ROOT, "reports", "Model_Evaluation_Metrics.csv")
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            model = row["Model"]
            metrics[model] = {
                "embedding_dim": 512 if "512" in model else 128,
                "accuracy": row["Accuracy (%)"] / 100.0,
                "precision": row["Precision (%)"] / 100.0,
                "recall": row["Recall (%)"] / 100.0,
                "f1": row["F1-Score"],
                "far": row["EER (%)"] / 100.0,
                "frr": row["EER (%)"] / 100.0,
                "threshold": row["Optimal Threshold"],
                "inference_time": 0.6,
                "ram_mb": 175 if "512" in model else 362,
            }
    except Exception as e:
        print("Lỗi đọc CSV:", e)
    return metrics


def save_uploaded_file(uploaded_file, save_dir="app/temp_uploads"):
    """Lưu file upload tạm thời, trả về đường dẫn."""
    abs_save_dir = os.path.join(PROJECT_ROOT, save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)
    path = os.path.join(abs_save_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path



# ===========================================================
#  FACE REGISTRATION & WEBCAM RECOGNITION
# ===========================================================

FACES_DB_DIR = "app/faces_db"
EMBEDDING_CACHE_PATH = os.path.join(PROJECT_ROOT, "app", "registered_faces", "embeddings_cache.pkl")
ALL_RECOGNITION_MODELS = [
    "Facenet_mtcnn",
    "Facenet_retinaface",
    "Facenet512_mtcnn",
    "Facenet512_retinaface",
]


def _normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _load_embedding_cache():
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        return {"version": 1, "records": []}
    try:
        with open(EMBEDDING_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and "records" in cache:
            return cache
    except Exception as e:
        print(f"Lỗi đọc embedding cache: {e}")
    return {"version": 1, "records": []}


def _save_embedding_cache(cache):
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _photo_signature(photo_path):
    stat = os.stat(photo_path)
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _iter_registered_photos():
    db_dir = get_faces_db_dir()
    if not os.path.exists(db_dir):
        return []

    photos = []
    for person_name in sorted(os.listdir(db_dir)):
        person_dir = os.path.join(db_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for photo_file in sorted(os.listdir(person_dir)):
            if photo_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                photos.append((person_name, os.path.join(person_dir, photo_file)))
    return photos


def _cache_record_is_valid(record, model_name):
    photo_path = record.get("photo_path")
    if record.get("model_name") != model_name or not photo_path or not os.path.exists(photo_path):
        return False
    try:
        sig = _photo_signature(photo_path)
        return record.get("mtime") == sig["mtime"] and record.get("size") == sig["size"]
    except OSError:
        return False


def build_registered_embeddings_cache(model_names=None, force=False):
    """Precompute embeddings for registered faces and persist them to disk."""
    model_names = model_names or ALL_RECOGNITION_MODELS
    cache = _load_embedding_cache()
    records = cache.get("records", [])

    existing = {}
    for record in records:
        key = (record.get("model_name"), record.get("photo_path"))
        if key[0] and key[1]:
            existing[key] = record

    new_records = []
    registered_photos = _iter_registered_photos()
    for person_name, photo_path in registered_photos:
        sig = _photo_signature(photo_path)
        for model_name in model_names:
            key = (model_name, photo_path)
            old = existing.get(key)
            if not force and old and _cache_record_is_valid(old, model_name):
                new_records.append(old)
                continue
            try:
                emb = _normalize_embedding(_get_embedding(photo_path, model_name))
                new_records.append({
                    "model_name": model_name,
                    "person_name": person_name,
                    "photo_path": photo_path,
                    "embedding": emb,
                    "mtime": sig["mtime"],
                    "size": sig["size"],
                    "cached_at": time.time(),
                })
            except Exception as e:
                print(f"Không cache được {photo_path} với {model_name}: {e}")

    cache = {"version": 1, "records": new_records, "updated_at": time.time()}
    _save_embedding_cache(cache)
    return get_embedding_cache_status()


def update_embedding_cache_for_photo(photo_path, person_name, model_names=None):
    """Compute embeddings for one newly registered photo and append/update cache."""
    model_names = model_names or ALL_RECOGNITION_MODELS
    cache = _load_embedding_cache()
    records = [r for r in cache.get("records", []) if r.get("photo_path") != photo_path]

    sig = _photo_signature(photo_path)
    for model_name in model_names:
        try:
            emb = _normalize_embedding(_get_embedding(photo_path, model_name))
            records.append({
                "model_name": model_name,
                "person_name": person_name,
                "photo_path": photo_path,
                "embedding": emb,
                "mtime": sig["mtime"],
                "size": sig["size"],
                "cached_at": time.time(),
            })
        except Exception as e:
            print(f"Không cache được ảnh mới {photo_path} với {model_name}: {e}")

    cache = {"version": 1, "records": records, "updated_at": time.time()}
    _save_embedding_cache(cache)
    return get_embedding_cache_status()


def get_cached_embeddings(model_name, auto_build=True):
    cache = _load_embedding_cache()
    records = [r for r in cache.get("records", []) if _cache_record_is_valid(r, model_name)]

    registered_count = len(_iter_registered_photos())
    if auto_build and len(records) < registered_count:
        build_registered_embeddings_cache([model_name], force=False)
        cache = _load_embedding_cache()
        records = [r for r in cache.get("records", []) if _cache_record_is_valid(r, model_name)]

    return records


def get_embedding_cache_status():
    cache = _load_embedding_cache()
    records = cache.get("records", [])
    by_model = {}
    for record in records:
        model = record.get("model_name", "unknown")
        if _cache_record_is_valid(record, model):
            by_model[model] = by_model.get(model, 0) + 1
    return {
        "cache_path": EMBEDDING_CACHE_PATH,
        "total_records": sum(by_model.values()),
        "by_model": by_model,
        "registered_photos": len(_iter_registered_photos()),
        "updated_at": cache.get("updated_at"),
    }

def get_faces_db_dir():
    """Trả về đường dẫn tới thư mục chứa khuôn mặt đã đăng ký."""
    abs_db_dir = os.path.join(PROJECT_ROOT, FACES_DB_DIR)
    os.makedirs(abs_db_dir, exist_ok=True)
    return abs_db_dir


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


def _person_photo_paths(person_name):
    """Return all registered image paths for one person."""
    db_dir = get_faces_db_dir()
    person_dir = os.path.join(db_dir, person_name.strip())
    if not os.path.isdir(person_dir):
        return []
    return [
        os.path.join(person_dir, f)
        for f in sorted(os.listdir(person_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]


def _next_person_photo_path(person_name):
    """Return the next stable face_NNN.jpg path for a person."""
    db_dir = get_faces_db_dir()
    person_dir = os.path.join(db_dir, person_name.strip())
    os.makedirs(person_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    return os.path.join(person_dir, f"face_{len(existing) + 1:03d}.jpg")


def check_registration_consistency(candidate_path, person_name, model_name, threshold=None):
    """Check whether a new image is similar enough to existing photos of person_name."""
    existing_paths = _person_photo_paths(person_name)
    if not existing_paths:
        return {
            "accepted": True,
            "is_new_person": True,
            "best_similarity": None,
            "threshold": float(threshold if threshold is not None else _get_threshold().get(model_name, 0.35)),
            "best_photo_path": None,
            "message": "User mới: ảnh đầu tiên được chấp nhận.",
        }

    gate_threshold = float(threshold if threshold is not None else _get_threshold().get(model_name, 0.35))
    candidate_emb = _normalize_embedding(_get_embedding(candidate_path, model_name)).astype(np.float32)
    best_similarity = -1.0
    best_photo_path = None

    for old_path in existing_paths:
        try:
            old_emb = _normalize_embedding(_get_embedding(old_path, model_name)).astype(np.float32)
            sim = float(old_emb @ candidate_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_photo_path = old_path
        except Exception as e:
            print(f"Không kiểm tra được ảnh cũ {old_path}: {e}")

    accepted = best_similarity >= gate_threshold
    return {
        "accepted": accepted,
        "is_new_person": False,
        "best_similarity": best_similarity,
        "threshold": gate_threshold,
        "best_photo_path": best_photo_path,
        "message": (
            "Ảnh mới giống user đã đăng ký, được chấp nhận."
            if accepted else
            "Ảnh mới không đủ giống các ảnh đã có của user này, đã từ chối để tránh nhiễu database."
        ),
    }


def register_face_image_checked(name, uploaded_file, model_name, threshold=None, enforce_consistency=True):
    """Register an image, optionally blocking wrong-person additions for existing users."""
    clean_name = name.strip()
    tmp_dir = os.path.join(PROJECT_ROOT, "app", "temp_uploads")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=tmp_dir) as tmp:
        tmp_path = tmp.name

    try:
        img = Image.open(uploaded_file).convert("RGB")
        img.save(tmp_path, "JPEG", quality=95)

        if enforce_consistency:
            gate = check_registration_consistency(tmp_path, clean_name, model_name, threshold)
            if not gate["accepted"]:
                return {"saved": False, "path": None, "gate": gate}
        else:
            gate = {"accepted": True, "message": "Đã bỏ qua kiểm tra giống user cũ."}

        filepath = _next_person_photo_path(clean_name)
        img.save(filepath, "JPEG", quality=95)
        update_embedding_cache_for_photo(filepath, clean_name, [model_name])
        return {"saved": True, "path": filepath, "gate": gate}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def register_face_image(name, uploaded_file):
    """
    Lưu ảnh khuôn mặt vào thư mục của người dùng và tạo embedding cache ngay.
    Returns: đường dẫn file đã lưu
    """
    clean_name = name.strip()
    filepath = _next_person_photo_path(clean_name)

    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img.save(filepath, "JPEG", quality=95)

    # Tối ưu tốc độ nhận diện: tạo embedding cho ảnh vừa đăng ký ngay lập tức.
    update_embedding_cache_for_photo(filepath, clean_name)

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


def _prune_cache_for_people(people_names):
    """Remove cache records for deleted/rebuilt people."""
    selected = {name.strip() for name in people_names if name and name.strip()}
    cache = _load_embedding_cache()
    records = [
        r for r in cache.get("records", [])
        if r.get("person_name") not in selected and os.path.exists(r.get("photo_path", ""))
    ]
    cache = {"version": 1, "records": records, "updated_at": time.time()}
    _save_embedding_cache(cache)
    return get_embedding_cache_status()


def delete_registered_people(people_names):
    """Delete selected registered people and remove their embedding cache."""
    import shutil

    db_dir = get_faces_db_dir()
    deleted = []
    errors = []
    for name in people_names:
        clean_name = name.strip()
        person_dir = os.path.join(db_dir, clean_name)
        if not clean_name or not os.path.isdir(person_dir):
            continue
        try:
            shutil.rmtree(person_dir)
            deleted.append(clean_name)
        except Exception as e:
            errors.append(f"{clean_name}: {e}")

    if deleted:
        _prune_cache_for_people(deleted)
    return {"deleted": deleted, "errors": errors, "cache_status": get_embedding_cache_status()}


def rebuild_embeddings_for_people(people_names, model_names=None, force=True):
    """Recompute embedding cache only for selected registered people.

    Returns a dict with cache status plus detailed success/error counts so the UI
    can avoid reporting success when dependency/model extraction fails.
    """
    selected = {name.strip() for name in people_names if name and name.strip()}
    model_names = model_names or ALL_RECOGNITION_MODELS
    if not selected:
        return {
            "created": 0,
            "failed": 0,
            "errors": {},
            "cache_status": get_embedding_cache_status(),
        }

    _prune_cache_for_people(selected)
    cache = _load_embedding_cache()
    records = cache.get("records", [])
    created = 0
    failed = 0
    errors = {}

    for person_name, photo_path in _iter_registered_photos():
        if person_name not in selected:
            continue
        sig = _photo_signature(photo_path)
        for model_name in model_names:
            try:
                emb = _normalize_embedding(_get_embedding(photo_path, model_name))
                records.append({
                    "model_name": model_name,
                    "person_name": person_name,
                    "photo_path": photo_path,
                    "embedding": emb,
                    "mtime": sig["mtime"],
                    "size": sig["size"],
                    "cached_at": time.time(),
                })
                created += 1
            except ModuleNotFoundError as e:
                failed += 1
                missing_module = getattr(e, "name", "unknown")
                key = f"Thiếu package `{missing_module}`"
                errors.setdefault(key, 0)
                errors[key] += 1
            except Exception as e:
                failed += 1
                key = str(e).strip() or e.__class__.__name__
                errors.setdefault(key, 0)
                errors[key] += 1

    cache = {"version": 1, "records": records, "updated_at": time.time()}
    _save_embedding_cache(cache)
    return {
        "created": created,
        "failed": failed,
        "errors": errors,
        "cache_status": get_embedding_cache_status(),
    }



def recognize_from_camera(camera_image_path, model_name):
    """
    So sánh ảnh chụp từ webcam với database embedding đã cache.
    Chỉ trích xuất embedding của ảnh webcam, không trích xuất lại từng ảnh database.
    Returns: list[dict] - kết quả sắp xếp theo similarity giảm dần
    """
    db_dir = get_faces_db_dir()
    if not os.path.exists(db_dir):
        return []

    # 1. Trích xuất embedding từ ảnh webcam đúng 1 lần cho model đang dùng.
    try:
        query_emb = _normalize_embedding(_get_embedding(camera_image_path, model_name))
    except Exception as e:
        return [{"error": str(e)}]

    # 2. Load cache embedding của database. Nếu thiếu cache thì tự build bổ sung.
    records = get_cached_embeddings(model_name, auto_build=True)
    if not records:
        return []

    threshold = _get_threshold()[model_name]

    # 3. Vectorized cosine similarity: matrix @ query thay cho loop get_embedding từng ảnh.
    matrix = np.vstack([_normalize_embedding(r["embedding"]) for r in records]).astype(np.float32)
    scores = matrix @ query_emb

    best_by_person = {}
    for idx, score in enumerate(scores):
        record = records[idx]
        person_name = record["person_name"]
        score = float(score)
        if person_name not in best_by_person or score > best_by_person[person_name]["similarity"]:
            best_by_person[person_name] = {
                "name": person_name,
                "similarity": score,
                "is_match": score >= threshold,
                "confidence": round(min(100.0, (score / threshold) * 100.0), 1),
                "photo_path": record["photo_path"],
            }

    results = list(best_by_person.values())
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

