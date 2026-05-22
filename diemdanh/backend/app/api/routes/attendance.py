from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.entities import AttendanceRecord, ManualAttendanceRequest, Student
from app.schemas.common import ReviewIn
from app.services.face_service import face_service
from app.services.attendance_service import (
    approve_manual_attendance_request,
    attendance_payload,
    create_attendance,
    create_manual_attendance_request,
    review_attendance,
)

router=APIRouter(prefix="/attendance", tags=["attendance"])


def _manual_request_payload(row: ManualAttendanceRequest):
    student = row.student
    return {
        "id": row.id,
        "session_id": row.session_id,
        "student_id": row.student_id,
        "student_code": student.student_code if student else None,
        "full_name": student.user.full_name if student and student.user else None,
        "photo_path": row.photo_path,
        "reason": row.reason,
        "confidence": row.confidence,
        "status": row.status.value if hasattr(row.status, "value") else row.status,
        "reviewed_by": row.reviewed_by,
        "reviewed_at": row.reviewed_at,
        "created_at": row.created_at,
    }


@router.post("/verify/{session_id}")
async def verify_attendance(session_id:int, file: UploadFile=File(...), db: Session=Depends(get_db), _=Depends(get_current_user)):
    result=face_service.verify(db, await file.read(), session_id)
    if not result.get("ok") and "student_id" not in result: return result
    row=create_attendance(db, session_id, result)
    return attendance_payload(row, result)


@router.post("/manual-requests/{session_id}")
async def create_manual_request(session_id:int, file: UploadFile=File(...), reason: str | None = Form(None), db: Session=Depends(get_db), user=Depends(get_current_user)):
    if user.role.value != "student":
        raise HTTPException(403, "Chỉ sinh viên mới gửi yêu cầu điểm danh")
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy hồ sơ sinh viên")
    path = face_service.save_image(await file.read(), face_service.settings.storage_path/"manual_attendance"/str(session_id)/str(student.id), "manual")
    row = create_manual_attendance_request(db, session_id, student.id, str(path), reason)
    return _manual_request_payload(row)


@router.get("/manual-requests/me")
def my_manual_requests(db: Session=Depends(get_db), user=Depends(get_current_user)):
    if user.role.value != "student":
        raise HTTPException(403, "Chỉ sinh viên mới xem yêu cầu của mình")
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy hồ sơ sinh viên")
    rows = db.query(ManualAttendanceRequest).filter(ManualAttendanceRequest.student_id == student.id).order_by(ManualAttendanceRequest.created_at.desc()).all()
    return [_manual_request_payload(row) for row in rows]


@router.get("/manual-requests")
def list_manual_requests(db: Session=Depends(get_db), _=Depends(require_roles("admin","teacher"))):
    rows = db.query(ManualAttendanceRequest).order_by(ManualAttendanceRequest.created_at.desc()).all()
    return [_manual_request_payload(row) for row in rows]


@router.post("/manual-requests/{request_id}/approve")
def approve_manual_request(request_id:int, data:ReviewIn, db: Session=Depends(get_db), user=Depends(require_roles("admin","teacher"))):
    row = approve_manual_attendance_request(db, request_id, user.id, True, data.reason)
    return _manual_request_payload(row)


@router.post("/manual-requests/{request_id}/reject")
def reject_manual_request(request_id:int, data:ReviewIn, db: Session=Depends(get_db), user=Depends(require_roles("admin","teacher"))):
    row = approve_manual_attendance_request(db, request_id, user.id, False, data.reason)
    return _manual_request_payload(row)


@router.get("/session/{session_id}")
def session_attendance(session_id:int, db: Session=Depends(get_db), _=Depends(require_roles("admin","teacher"))):
    return db.query(AttendanceRecord).filter(AttendanceRecord.session_id==session_id).order_by(AttendanceRecord.checked_at.desc()).all()


@router.post("/{attendance_id}/review")
def review(attendance_id:int, data:ReviewIn, db: Session=Depends(get_db), user=Depends(require_roles("admin","teacher"))):
    return review_attendance(db, attendance_id, user.id, data.new_status, data.reason)
