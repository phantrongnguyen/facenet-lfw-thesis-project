from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.entities import FaceEmbedding, FacePhotoChangeRequest, FacePhotoChangeStatus, Student, UserRole
from app.services.face_service import face_service, unpack

router=APIRouter(prefix="/face-changes", tags=["face-changes"])


def storage_url(path: str | None):
    if not path:
        return None
    storage_root = str(face_service.settings.storage_path.resolve()).replace("\\", "/")
    photo_path = str(path).replace("\\", "/")
    if photo_path.startswith(storage_root):
        return "/storage" + photo_path[len(storage_root):]
    return None


def serialize_request(row: FacePhotoChangeRequest):
    student = row.student
    return {
        "id": row.id,
        "student_id": row.student_id,
        "student_code": student.student_code if student else None,
        "full_name": student.user.full_name if student and student.user else None,
        "old_photo_url": storage_url(row.old_photo_path),
        "new_photo_url": storage_url(row.new_photo_path),
        "model_key": row.model_key,
        "confidence": row.confidence,
        "status": row.status.value if hasattr(row.status, "value") else row.status,
        "note": row.note,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.post("/requests")
async def create_change_request(file: UploadFile=File(...), db: Session=Depends(get_db), user=Depends(get_current_user)):
    if user.role.value != "student":
        raise HTTPException(403, "Chỉ sinh viên mới được yêu cầu đổi ảnh đăng ký")
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy hồ sơ sinh viên")
    pending = db.query(FacePhotoChangeRequest).filter(
        FacePhotoChangeRequest.student_id == student.id,
        FacePhotoChangeRequest.status == FacePhotoChangeStatus.pending,
    ).first()
    if pending:
        raise HTTPException(409, "Bạn đang có một yêu cầu đổi ảnh chờ duyệt")
    current = db.query(FaceEmbedding).filter(
        FaceEmbedding.student_id == student.id,
        FaceEmbedding.model_key == face_service.model_key,
    ).order_by(FaceEmbedding.created_at.desc()).first()
    try:
        prepared = face_service.prepare_embedding(await file.read(), face_service.settings.storage_path/"face_change_requests"/str(student.id), "change")
        confidence = None
        if current:
            confidence = float(unpack(current.embedding) @ prepared["embedding"])
        row = FacePhotoChangeRequest(
            student_id=student.id,
            old_photo_path=current.photo_path if current else None,
            new_photo_path=str(prepared["path"]),
            new_embedding=prepared["packed"],
            model_key=face_service.model_key,
            confidence=confidence,
            status=FacePhotoChangeStatus.pending,
        )
        db.add(row); db.commit(); db.refresh(row)
        return serialize_request(row)
    except Exception as exc:
        raise HTTPException(400, f"Không thể tạo yêu cầu đổi ảnh: {exc}") from exc


@router.get("/requests/me")
def my_change_requests(db: Session=Depends(get_db), user=Depends(get_current_user)):
    if user.role.value != "student":
        raise HTTPException(403, "Chỉ sinh viên mới xem lịch sử đổi ảnh của mình")
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy hồ sơ sinh viên")
    rows = db.query(FacePhotoChangeRequest).filter(FacePhotoChangeRequest.student_id == student.id).order_by(FacePhotoChangeRequest.created_at.desc()).all()
    return [serialize_request(r) for r in rows]


@router.get("/requests")
def list_change_requests(db: Session=Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    rows = db.query(FacePhotoChangeRequest).order_by(FacePhotoChangeRequest.created_at.desc()).all()
    return [serialize_request(r) for r in rows]


@router.post("/requests/{request_id}/approve")
def approve_change_request(request_id:int, db: Session=Depends(get_db), user=Depends(require_roles("admin", "teacher"))):
    row = db.query(FacePhotoChangeRequest).filter(FacePhotoChangeRequest.id == request_id).first()
    if not row:
        raise HTTPException(404, "Không tìm thấy yêu cầu đổi ảnh")
    if row.status != FacePhotoChangeStatus.pending:
        raise HTTPException(409, "Yêu cầu này đã được xử lý")
    face_service.activate_embedding(db, row.student_id, row.new_photo_path, row.new_embedding, row.model_key)
    row.status = FacePhotoChangeStatus.approved
    row.reviewed_by = user.id
    row.reviewed_at = datetime.utcnow()
    row.note = "Đã duyệt thủ công"
    db.add(row); db.commit(); db.refresh(row)
    return serialize_request(row)


@router.post("/requests/{request_id}/reject")
def reject_change_request(request_id:int, db: Session=Depends(get_db), user=Depends(require_roles("admin", "teacher"))):
    row = db.query(FacePhotoChangeRequest).filter(FacePhotoChangeRequest.id == request_id).first()
    if not row:
        raise HTTPException(404, "Không tìm thấy yêu cầu đổi ảnh")
    if row.status != FacePhotoChangeStatus.pending:
        raise HTTPException(409, "Yêu cầu này đã được xử lý")
    row.status = FacePhotoChangeStatus.rejected
    row.reviewed_by = user.id
    row.reviewed_at = datetime.utcnow()
    row.note = "Đã từ chối thủ công"
    db.add(row); db.commit(); db.refresh(row)
    return serialize_request(row)
