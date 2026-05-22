from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.entities import (
    AttendanceRecord,
    AttendanceStatus,
    ManualAttendanceRequest,
    ManualAttendanceRequestStatus,
    ManualReviewLog,
    Session as ClassSession,
    SessionStatus,
    SessionStudent,
)

STATUS_LABELS = {
    AttendanceStatus.present: "Có mặt",
    AttendanceStatus.late: "Đi trễ",
    AttendanceStatus.absent: "Vắng",
    AttendanceStatus.needs_review: "Cần giảng viên duyệt",
}


def status_for(session: ClassSession, now: datetime, is_match: bool, confidence: float, threshold: float) -> AttendanceStatus:
    if not is_match or confidence < threshold:
        return AttendanceStatus.needs_review
    late_at = session.scheduled_start.timestamp() + session.late_threshold_minutes * 60
    return AttendanceStatus.late if now.timestamp() > late_at else AttendanceStatus.present


def attendance_message(status: AttendanceStatus) -> str:
    if status == AttendanceStatus.present:
        return "Điểm danh thành công"
    if status == AttendanceStatus.late:
        return "Điểm danh thành công nhưng đã quá ngưỡng trễ"
    if status == AttendanceStatus.needs_review:
        return "Nhận diện chưa khớp. Vui lòng điểm danh lại hoặc gửi yêu cầu điểm danh thủ công."
    return "Đã ghi nhận trạng thái điểm danh"


def _get_open_session(db: Session, session_id: int) -> ClassSession:
    session = db.get(ClassSession, session_id)
    if not session:
        raise HTTPException(404, "Không tìm thấy buổi học")
    if session.status == SessionStatus.open and datetime.utcnow() >= session.scheduled_end:
        session.status = SessionStatus.closed
        db.commit()
        db.refresh(session)
    if session.status == SessionStatus.scheduled:
        raise HTTPException(400, "Buổi học chưa mở điểm danh")
    if session.status == SessionStatus.closed:
        raise HTTPException(400, "Buổi học đã đóng điểm danh")
    return session


def _ensure_enrolled(db: Session, session_id: int, student_id: int) -> None:
    exists = db.query(SessionStudent).filter(
        SessionStudent.session_id == session_id,
        SessionStudent.student_id == student_id,
    ).first()
    if not exists:
        raise HTTPException(403, "Sinh viên không nằm trong danh sách buổi học này")


def _existing_success_attendance(db: Session, session_id: int, student_id: int) -> AttendanceRecord | None:
    return db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.status.in_([AttendanceStatus.present, AttendanceStatus.late]),
    ).first()


def create_attendance(db: Session, session_id: int, result: dict) -> AttendanceRecord:
    session = _get_open_session(db, session_id)
    _ensure_enrolled(db, session_id, result["student_id"])

    if existing := _existing_success_attendance(db, session_id, result["student_id"]):
        result["already_attended"] = True
        return existing

    now = datetime.utcnow()
    st = status_for(session, now, result["is_match"], result["confidence"], result["threshold"])
    row = AttendanceRecord(
        session_id=session_id,
        student_id=result["student_id"],
        confidence=result["confidence"],
        status=st,
        photo_path=result["photo_path"],
        checked_at=now,
    )
    db.add(row)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(409, "Sinh viên đã điểm danh trong buổi học này") from exc
    db.refresh(row)
    return row


def attendance_payload(row: AttendanceRecord, result: dict) -> dict:
    status = row.status if isinstance(row.status, AttendanceStatus) else AttendanceStatus(row.status)
    return {
        "ok": status != AttendanceStatus.needs_review,
        "message": attendance_message(status),
        "attendance_id": row.id,
        "session_id": row.session_id,
        "student_id": row.student_id,
        "status": status.value,
        "status_label": STATUS_LABELS[status],
        "confidence": row.confidence,
        "threshold": result.get("threshold"),
        "is_match": result.get("is_match"),
        "model_key": result.get("model_key"),
        "model_label": result.get("model_label"),
        "photo_path": row.photo_path,
        "checked_at": row.checked_at,
        "can_request_manual_attendance": status == AttendanceStatus.needs_review,
        "already_attended": result.get("already_attended", False),
    }


def create_manual_attendance_request(db: Session, session_id: int, student_id: int, photo_path: str, reason: str | None, confidence: float | None = None):
    _get_open_session(db, session_id)
    _ensure_enrolled(db, session_id, student_id)
    if db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id, AttendanceRecord.student_id == student_id).first():
        raise HTTPException(409, "Sinh viên đã có bản ghi điểm danh trong buổi học này")
    row = ManualAttendanceRequest(
        session_id=session_id,
        student_id=student_id,
        photo_path=photo_path,
        reason=reason,
        confidence=confidence,
    )
    db.add(row)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(409, "Đã có yêu cầu điểm danh đang chờ duyệt cho buổi học này") from exc
    db.refresh(row)
    return row


def approve_manual_attendance_request(db: Session, request_id: int, reviewer_id: int, approve: bool, reason: str):
    req = db.get(ManualAttendanceRequest, request_id)
    if not req:
        raise HTTPException(404, "Không tìm thấy yêu cầu điểm danh")
    if req.status != ManualAttendanceRequestStatus.pending:
        raise HTTPException(400, "Yêu cầu này đã được xử lý")
    req.reviewed_by = reviewer_id
    req.reviewed_at = datetime.utcnow()
    req.reason = f"{req.reason or ''}\nPhản hồi giảng viên: {reason}".strip()
    if not approve:
        req.status = ManualAttendanceRequestStatus.rejected
        db.commit()
        db.refresh(req)
        return req
    session = db.get(ClassSession, req.session_id)
    st = status_for(session, datetime.utcnow(), True, req.confidence or 1.0, 0.0)
    db.add(AttendanceRecord(
        session_id=req.session_id,
        student_id=req.student_id,
        confidence=req.confidence or 1.0,
        status=st,
        photo_path=req.photo_path,
        checked_at=datetime.utcnow(),
    ))
    req.status = ManualAttendanceRequestStatus.approved
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(409, "Sinh viên đã có bản ghi điểm danh trong buổi học này") from exc
    db.refresh(req)
    return req


def review_attendance(db: Session, attendance_id: int, reviewer_id: int, new_status: str, reason: str):
    row = db.get(AttendanceRecord, attendance_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy bản ghi điểm danh")
    try:
        next_status = AttendanceStatus(new_status)
    except ValueError as exc:
        raise HTTPException(400, "Trạng thái điểm danh không hợp lệ") from exc
    old = row.status.value if hasattr(row.status, "value") else str(row.status)
    row.status = next_status
    db.add(ManualReviewLog(attendance_id=attendance_id, reviewer_id=reviewer_id, old_status=old, new_status=next_status.value, reason=reason))
    db.commit()
    db.refresh(row)
    return row
