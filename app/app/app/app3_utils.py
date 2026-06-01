"""
Utilities for App3: white-theme realtime face recognition.
Reuses the optimized registration embedding cache from app1_utils and adds
frame-based recognition helpers for webcam video streams.
"""

import os
import tempfile
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

from app1_utils import (  # re-export stable app2 helpers
    ALL_RECOGNITION_MODELS,
    PROJECT_ROOT,
    build_registered_embeddings_cache,
    delete_registered_people,
    extract_face_crop,
    get_cached_embeddings,
    get_embedding_cache_status,
    get_faces_db_dir,
    is_name_duplicate,
    is_valid_name,
    list_registered_people,
    rebuild_embeddings_for_people,
    register_face_image,
    register_face_image_checked,
    save_uploaded_file,
    update_embedding_cache_for_photo,
    _get_embedding,
    _get_threshold,
    _normalize_embedding,
)


def _split_model_name(model_name: str) -> Tuple[str, str]:
    """Return (deepface_model, detector_backend) from names like Facenet512_mtcnn."""
    if "_" not in model_name:
        return model_name, "mtcnn"
    model, detector = model_name.split("_", 1)
    return model, detector


def get_available_thresholds() -> Dict[str, float]:
    """Load recognition thresholds, falling back to common defaults."""
    thresholds = _get_threshold()
    defaults = {
        "Facenet_mtcnn": 0.40,
        "Facenet_retinaface": 0.40,
        "Facenet512_mtcnn": 0.35,
        "Facenet512_retinaface": 0.35,
    }
    defaults.update(thresholds)
    return defaults


def recognize_image_path_cached(image_path: str, model_name: str, threshold: Optional[float] = None, top_k: int = 5, auto_build: bool = False):
    """Recognize one image by extracting only the query embedding and comparing to cache."""
    started = time.time()
    records = get_cached_embeddings(model_name, auto_build=auto_build)
    if not records:
        return {
            "error": (
                f"Cache embedding cho model {model_name} chưa sẵn hoặc đang thiếu. "
                "Hãy vào tab Database và bấm rebuild cache trước khi nhận diện để tránh chờ lâu."
            ),
            "cache_missing": True,
            "elapsed": round(time.time() - started, 3),
        }

    query_emb = _normalize_embedding(_get_embedding(image_path, model_name)).astype(np.float32)
    return _match_query_embedding(query_emb, records, model_name, threshold, top_k, started=started)


def _match_query_embedding(query_emb: np.ndarray, records, model_name: str, threshold: Optional[float], top_k: int = 5, started: Optional[float] = None):
    """Compare one normalized query embedding with cached database embeddings."""
    matrix = np.vstack([_normalize_embedding(r["embedding"]) for r in records]).astype(np.float32)
    scores = matrix @ query_emb

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_record = records[best_idx]
    threshold = float(threshold if threshold is not None else get_available_thresholds().get(model_name, 0.35))
    is_match = best_score >= threshold

    ordered = np.argsort(scores)[::-1][:max(1, int(top_k))]
    top_matches = []
    for rank, idx in enumerate(ordered, start=1):
        record = records[int(idx)]
        top_matches.append({
            "rank": rank,
            "person_name": record["person_name"],
            "photo_path": record["photo_path"],
            "similarity": float(scores[int(idx)]),
        })

    return {
        "name": best_record["person_name"] if is_match else "Không xác định",
        "raw_name": best_record["person_name"],
        "similarity": best_score,
        "threshold": threshold,
        "is_match": is_match,
        "confidence": round(min(100.0, max(0.0, best_score / max(threshold, 1e-6) * 100.0)), 1),
        "photo_path": best_record["photo_path"],
        "elapsed": round(time.time() - started, 3) if started else 0.0,
        "embedding_dim": int(query_emb.size),
        "query_embedding": query_emb.tolist(),
        "top_matches": top_matches,
    }


def _enhance_realtime_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Improve dark webcam frames before realtime recognition/display."""
    import cv2

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(enhanced, alpha=1.12, beta=12)


def _resize_frame(frame_bgr: np.ndarray, max_width: int = 720) -> np.ndarray:
    """Resize large webcam frames to keep Streamlit realtime rendering light."""
    import cv2

    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    return cv2.resize(frame_bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _center_face_roi(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Fallback crop used only when a detector cannot find a face."""
    h, w = frame_bgr.shape[:2]
    box_w, box_h = int(w * 0.55), int(h * 0.68)
    x, y = (w - box_w) // 2, (h - box_h) // 2
    return frame_bgr[y:y + box_h, x:x + box_w], (x, y, box_w, box_h)


def detect_face_rois(frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Detect visible faces and return padded face ROIs for recognition.

    This lightweight OpenCV detector keeps realtime responsive and avoids the
    old fixed center crop, which failed whenever the face was not perfectly
    inside the guide area.
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return []

    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(72, 72))
    h, w = frame_bgr.shape[:2]
    rois = []
    for x, y, bw, bh in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
        pad_x = int(bw * 0.32)
        pad_y = int(bh * 0.42)
        x1 = max(0, int(x) - pad_x)
        y1 = max(0, int(y) - pad_y)
        x2 = min(w, int(x + bw) + pad_x)
        y2 = min(h, int(y + bh) + pad_y)
        if x2 > x1 and y2 > y1:
            rois.append((frame_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)))
    return rois


def prepare_realtime_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Return natural display frame, detected face ROI, and detected overlay box."""
    # Keep the displayed frame visually consistent with the static/compare camera
    # previews. The enhanced copy is only used internally to help face detection
    # in dim lighting without making Realtime look darker or overprocessed.
    display_frame = _resize_frame(frame_bgr)
    detection_frame = _enhance_realtime_frame(display_frame)
    rois = detect_face_rois(detection_frame)
    if not rois:
        return display_frame, None, None
    _, box = rois[0]
    x, y, w, h = box
    roi = display_frame[y:y + h, x:x + w]
    return display_frame, roi, box


def recognize_frame_cached(frame_bgr: np.ndarray, model_name: str, threshold: Optional[float] = None):
    """Recognize a BGR OpenCV face ROI using cached registered embeddings."""
    import cv2

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Chưa detect được khuôn mặt trong frame camera.")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=os.path.join(PROJECT_ROOT, "app", "temp_uploads")) as tmp:
        temp_path = tmp.name
    try:
        Image.fromarray(rgb).save(temp_path, "JPEG", quality=90)
        return recognize_image_path_cached(temp_path, model_name, threshold)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def safe_recognize_frame_cached(frame_bgr: np.ndarray, model_name: str, threshold: Optional[float] = None):
    """Recognize a frame without letting DeepFace/TensorFlow errors break realtime UI."""
    try:
        return recognize_frame_cached(frame_bgr, model_name, threshold)
    except Exception as exc:
        return {
            "name": "Không xác định",
            "raw_name": "-",
            "similarity": 0.0,
            "threshold": float(threshold if threshold is not None else get_available_thresholds().get(model_name, 0.35)),
            "is_match": False,
            "error": f"Chưa nhận diện được frame này: {exc}",
        }


def _camera_backend_code(backend_name: str):
    """Map UI backend name to OpenCV backend constant."""
    import cv2

    normalized = (backend_name or "Auto").lower()
    if normalized == "directshow":
        return cv2.CAP_DSHOW
    if normalized == "msmf":
        return cv2.CAP_MSMF
    return None


def open_camera_capture(camera_index: int = 0, backend_name: str = "DirectShow"):
    """Open camera with the selected backend and return (capture, message)."""
    import cv2

    backend = _camera_backend_code(backend_name)
    try:
        cap = cv2.VideoCapture(int(camera_index), backend) if backend is not None else cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            cap.release()
            return None, f"Không mở được camera index {camera_index} bằng backend {backend_name}. Thử index 1/2 hoặc backend khác."

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        return cap, f"Đã mở camera index {camera_index} bằng backend {backend_name}."
    except Exception as exc:
        return None, f"Lỗi mở camera: {exc}"


def capture_camera_frame(camera_index: int = 0, backend_name: str = "DirectShow"):
    """Open camera, read one frame, close camera, and return RGB preview."""
    import cv2

    cap, message = open_camera_capture(camera_index, backend_name)
    if cap is None:
        return False, None, message
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None, "Mở được camera nhưng không đọc được frame. Camera có thể đang bị app khác chiếm."
        display_frame, _, _ = prepare_realtime_frame(frame)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        return True, rgb, message
    finally:
        cap.release()


def draw_recognition_overlay(frame_bgr: np.ndarray, result, box: Optional[Tuple[int, int, int, int]] = None):
    """Draw recognition label only after a valid face recognition attempt.

    Important: the rectangle is not a guide box. It should appear only when
    the recognizer has produced a valid face result for the current ROI. If the
    detector/model reports an error (for example no face found), return the
    clean camera frame without drawing a fake red box.
    """
    import cv2

    if not result or result.get("error"):
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if box is None:
        box_w, box_h = int(w * 0.55), int(h * 0.68)
        x, y = (w - box_w) // 2, (h - box_h) // 2
        box = (x, y, box_w, box_h)

    x, y, bw, bh = box
    if result.get("is_match"):
        color = (64, 180, 96)
        label = f"{result['name']}  {result['similarity']:.3f}"
    else:
        color = (64, 88, 255)
        label = f"Khong xac dinh  {result.get('similarity', 0):.3f}"

    cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), color, 2)
    cv2.rectangle(frame_bgr, (x, max(0, y - 38)), (min(w, x + max(250, bw)), y), color, -1)
    cv2.putText(frame_bgr, label, (x + 12, max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame_bgr


def make_webrtc_processor(model_name: str, threshold: Optional[float], recognize_every: float = 2.0):
    """Create a streamlit-webrtc VideoProcessorBase class configured for App3."""
    import threading
    import av
    import cv2
    from streamlit_webrtc import VideoProcessorBase

    class App3WebRTCProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_name = model_name
            self.threshold = threshold
            self.recognize_every = float(recognize_every)
            self.last_result = None
            self.last_error = None
            self.last_recognition_at = 0.0
            self.lock = threading.Lock()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            display_frame, recognition_roi, overlay_box = prepare_realtime_frame(img)

            now = time.time()
            if now - self.last_recognition_at >= self.recognize_every:
                self.last_recognition_at = now
                result = safe_recognize_frame_cached(recognition_roi, self.model_name, self.threshold)
                with self.lock:
                    self.last_result = result
                    self.last_error = result.get("error")

            with self.lock:
                result = dict(self.last_result) if self.last_result else None
            display = draw_recognition_overlay(display_frame, result, overlay_box)
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        def get_state(self):
            with self.lock:
                return {
                    "last_result": dict(self.last_result) if self.last_result else None,
                    "last_error": self.last_error,
                }

    return App3WebRTCProcessor
