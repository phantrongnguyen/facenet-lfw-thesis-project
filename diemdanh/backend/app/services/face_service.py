from __future__ import annotations
import io, sys, uuid
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.models.entities import FaceEmbedding

MODEL_OPTIONS = ["Facenet", "Facenet_mtcnn", "Facenet_retinaface", "Facenet512", "Facenet512_mtcnn", "Facenet512_retinaface"]
PRIMARY_MODEL_KEY = "Facenet512_retinaface"
PRIMARY_MODEL_LABEL = "Facenet512 + RetinaFace"
THRESHOLDS = {
    "Facenet_mtcnn": 0.306,
    "Facenet_retinaface": 0.401176,
    "Facenet512_mtcnn": 0.417026,
    "Facenet512_retinaface": 0.407157,
    "Facenet": 0.391451,
    "Facenet512": 0.420161,
}

def normalize(v: np.ndarray) -> np.ndarray:
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v if n==0 else v/n

def pack(v: np.ndarray) -> bytes:
    return normalize(v).astype(np.float32).tobytes()

def unpack(blob: bytes) -> np.ndarray:
    return normalize(np.frombuffer(blob, dtype=np.float32))

class FaceService:
    def __init__(self, model_key: str | None = None):
        self.settings=get_settings(); self.model_key=model_key or self.settings.default_face_model
        self.matrix=np.empty((0,0), dtype=np.float32); self.student_ids=[]; self.embedding_ids=[]
        self._ensure_project_imports()

    def _ensure_project_imports(self):
        project_root = str(Path(__file__).resolve().parents[4])
        if project_root not in sys.path: sys.path.insert(0, project_root)

    def _embedding(self, image_path: Path) -> np.ndarray:
        from src.embedding_extractor import get_embedding
        return normalize(np.asarray(get_embedding(str(image_path), self.model_key), dtype=np.float32))

    def reload(self, db: Session):
        rows=db.query(FaceEmbedding).filter(FaceEmbedding.model_key==self.model_key).all()
        vectors=[]; self.student_ids=[]; self.embedding_ids=[]
        for r in rows:
            vec=unpack(r.embedding)
            if vec.size:
                vectors.append(vec); self.student_ids.append(r.student_id); self.embedding_ids.append(r.id)
        self.matrix=np.vstack(vectors).astype(np.float32) if vectors else np.empty((0,0), dtype=np.float32)
        return {"count": len(vectors), "model_key": self.model_key, "model_label": PRIMARY_MODEL_LABEL if self.model_key == PRIMARY_MODEL_KEY else self.model_key, "threshold": THRESHOLDS.get(self.model_key, self.settings.face_match_threshold)}

    def save_image(self, image_bytes: bytes, folder: Path, prefix: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        img=Image.open(io.BytesIO(image_bytes)).convert("RGB"); img.thumbnail((1000,1000))
        path=folder/f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        img.save(path, "JPEG", quality=90); return path

    def register(self, db: Session, student_id: int, image_bytes: bytes):
        folder=self.settings.storage_path/"faces"/str(student_id)
        prepared=self.prepare_embedding(image_bytes, folder, "face")
        row=FaceEmbedding(student_id=student_id, model_key=self.model_key, embedding=pack(prepared["embedding"]), photo_path=str(prepared["path"]))
        db.add(row); db.commit(); db.refresh(row); self.reload(db)
        return {"embedding_id": row.id, "student_id": student_id, "model_key": self.model_key, "model_label": PRIMARY_MODEL_LABEL if self.model_key == PRIMARY_MODEL_KEY else self.model_key, "threshold": THRESHOLDS.get(self.model_key, self.settings.face_match_threshold), "photo_path": row.photo_path}

    def prepare_embedding(self, image_bytes: bytes, folder: Path, prefix: str):
        path=self.save_image(image_bytes, folder, prefix)
        emb=self._embedding(path)
        return {"path": path, "embedding": emb, "packed": pack(emb)}

    def activate_embedding(self, db: Session, student_id: int, image_path: str, embedding: bytes, model_key: str | None = None):
        row=FaceEmbedding(student_id=student_id, model_key=model_key or self.model_key, embedding=embedding, photo_path=image_path)
        db.add(row); db.commit(); db.refresh(row); self.reload(db)
        return row

    def verify(self, db: Session, image_bytes: bytes, session_id: int, threshold: float | None = None):
        if self.matrix.size == 0: self.reload(db)
        if self.matrix.size == 0: return {"ok": False, "error": "No face embeddings loaded"}
        folder=self.settings.storage_path/"attendance"/str(session_id)
        path=self.save_image(image_bytes, folder, "attendance")
        return self._match_image(path, threshold)

    def verify_test(self, db: Session, image_bytes: bytes, student_id: int, threshold: float | None = None):
        if self.matrix.size == 0: self.reload(db)
        if self.matrix.size == 0: return {"ok": False, "error": "No face embeddings loaded"}
        folder=self.settings.storage_path/"face_tests"/str(student_id)
        path=self.save_image(image_bytes, folder, "test")
        result=self._match_image(path, threshold)
        result["expected_student_id"] = student_id
        result["matched_self"] = bool(result.get("is_match") and result.get("student_id") == student_id)
        result["mode"] = "test_only"
        return result

    def _match_image(self, image_path: Path, threshold: float | None = None):
        query=self._embedding(image_path)
        scores=self.matrix @ query
        best_idx=int(np.argmax(scores)); score=float(scores[best_idx])
        th=threshold or THRESHOLDS.get(self.model_key, self.settings.face_match_threshold)
        return {"ok": True, "student_id": self.student_ids[best_idx], "embedding_id": self.embedding_ids[best_idx], "confidence": score, "is_match": score >= th, "threshold": th, "photo_path": str(image_path), "model_key": self.model_key, "model_label": PRIMARY_MODEL_LABEL if self.model_key == PRIMARY_MODEL_KEY else self.model_key}

face_service = FaceService()
