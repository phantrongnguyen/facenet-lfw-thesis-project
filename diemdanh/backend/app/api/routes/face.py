from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.entities import FaceEmbedding, Student
from app.services.face_service import face_service

router=APIRouter(prefix="/face", tags=["face"])

@router.post("/register/{student_id}")
async def register_face(student_id:int, file: UploadFile=File(...), db: Session=Depends(get_db), _=Depends(require_roles("admin","teacher","student"))):
    return face_service.register(db, student_id, await file.read())

@router.post("/test")
async def test_face(file: UploadFile=File(...), db: Session=Depends(get_db), user=Depends(get_current_user)):
    if user.role.value != "student":
        raise HTTPException(403, "Chỉ sinh viên mới được test nhận diện khuôn mặt")
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy hồ sơ sinh viên")
    result = face_service.verify_test(db, await file.read(), student.id)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error") or "Không thể kiểm tra khuôn mặt")
    matched_student = db.query(Student).filter(Student.id == result.get("student_id")).first()
    registered_face = db.query(FaceEmbedding).filter(
        FaceEmbedding.student_id == student.id,
        FaceEmbedding.model_key == result.get("model_key"),
    ).order_by(FaceEmbedding.created_at.desc()).first()
    if registered_face and registered_face.photo_path:
        storage_root = str(face_service.settings.storage_path.resolve()).replace("\\", "/")
        photo_path = str(registered_face.photo_path).replace("\\", "/")
        if photo_path.startswith(storage_root):
            result["registered_photo_url"] = "/storage" + photo_path[len(storage_root):]
    result["student"] = {
        "id": student.id,
        "student_code": student.student_code,
        "full_name": student.user.full_name if student.user else student.student_code,
    }
    if matched_student:
        result["matched_student"] = {
            "id": matched_student.id,
            "student_code": matched_student.student_code,
            "full_name": matched_student.user.full_name if matched_student.user else matched_student.student_code,
        }
    return result

@router.post("/reload")
def reload_faces(db: Session=Depends(get_db), _=Depends(require_roles("admin","teacher"))):
    return face_service.reload(db)
