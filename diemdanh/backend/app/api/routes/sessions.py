from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.entities import AttendanceRecord, AttendanceStatus, Class, Session as ClassSession, SessionStatus, SessionStudent, Student
from app.schemas.common import SessionCreate, SessionOut, SessionStudentOut, SessionUpdate

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _student_name(student: Student) -> str | None:
    return student.user.full_name if getattr(student, "user", None) else None


def _auto_close_if_needed(db: Session, row: ClassSession) -> None:
    if row.status == SessionStatus.open and datetime.utcnow() >= row.scheduled_end:
        row.status = SessionStatus.closed
        db.commit()
        db.refresh(row)


def _ensure_class(db: Session, class_id: int):
    if not db.get(Class, class_id):
        raise HTTPException(404, "Không tìm thấy lớp học")


def _ensure_students(db: Session, student_ids: list[int]) -> list[Student]:
    ids = sorted({int(x) for x in student_ids if x})
    if not ids:
        return []
    rows = db.query(Student).filter(Student.id.in_(ids)).all()
    found = {row.id for row in rows}
    missing = [sid for sid in ids if sid not in found]
    if missing:
        raise HTTPException(404, f"Không tìm thấy sinh viên: {', '.join(map(str, missing))}")
    return rows


def _sync_session_students(db: Session, session_id: int, student_ids: list[int]) -> None:
    _ensure_students(db, student_ids)
    ids = sorted({int(x) for x in student_ids if x})
    db.query(SessionStudent).filter(SessionStudent.session_id == session_id).delete(synchronize_session=False)
    for student_id in ids:
        db.add(SessionStudent(session_id=session_id, student_id=student_id))


def _get_session_or_404(db: Session, session_id: int) -> ClassSession:
    row = db.get(ClassSession, session_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy buổi học")
    _auto_close_if_needed(db, row)
    return row


def _session_out(db: Session, row: ClassSession) -> SessionOut:
    _auto_close_if_needed(db, row)
    records = {
        r.student_id: r
        for r in db.query(AttendanceRecord).filter(AttendanceRecord.session_id == row.id).all()
    }
    enrolled = (
        db.query(Student)
        .join(SessionStudent, SessionStudent.student_id == Student.id)
        .filter(SessionStudent.session_id == row.id)
        .order_by(Student.student_code.asc())
        .all()
    )
    present_statuses = {AttendanceStatus.present.value, AttendanceStatus.late.value}
    present_count = sum(1 for record in records.values() if _status_value(record.status) in present_statuses)
    total = len(enrolled)
    remaining = max(0, int((row.scheduled_end - datetime.utcnow()).total_seconds())) if row.status == SessionStatus.open else 0
    students = [
        SessionStudentOut(
            id=student.id,
            student_code=student.student_code,
            full_name=_student_name(student),
            status=student.status,
            class_id=student.class_id,
            class_name=student.klass.name if student.klass else None,
            attendance_status=_status_value(records[student.id].status) if student.id in records else None,
            checked_at=records[student.id].checked_at if student.id in records else None,
        )
        for student in enrolled
    ]
    return SessionOut(
        id=row.id,
        class_id=row.class_id,
        class_name=row.klass.name if getattr(row, "klass", None) else None,
        title=row.title,
        scheduled_start=row.scheduled_start,
        scheduled_end=row.scheduled_end,
        late_threshold_minutes=row.late_threshold_minutes,
        status=_status_value(row.status),
        student_count=total,
        present_count=present_count,
        attendance_progress=f"{present_count}/{total}",
        remaining_seconds=remaining,
        students=students,
    )


@router.get("", response_model=list[SessionOut])
def list_sessions(db: Session = Depends(get_db), user=Depends(get_current_user)):
    query = db.query(ClassSession).order_by(ClassSession.id.desc())
    if user.role.value == "student":
        student = db.query(Student).filter(Student.user_id == user.id).first()
        if not student:
            return []
        query = query.join(SessionStudent, SessionStudent.session_id == ClassSession.id).filter(SessionStudent.student_id == student.id)
    rows = query.all()
    return [_session_out(db, row) for row in rows]


@router.post("", response_model=SessionOut)
def create_session(data: SessionCreate, db: Session = Depends(get_db), user=Depends(require_roles("admin", "teacher"))):
    _ensure_class(db, data.class_id)
    if data.scheduled_end <= data.scheduled_start:
        raise HTTPException(400, "Thời gian kết thúc phải sau thời gian bắt đầu")
    payload = data.model_dump(exclude={"student_ids"})
    row = ClassSession(**payload, created_by=user.id)
    db.add(row)
    db.flush()
    _sync_session_students(db, row.id, data.student_ids)
    db.commit()
    db.refresh(row)
    return _session_out(db, row)


@router.put("/{session_id}", response_model=SessionOut)
def update_session(session_id: int, data: SessionUpdate, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = _get_session_or_404(db, session_id)
    if row.status == SessionStatus.closed:
        raise HTTPException(400, "Buổi học đã đóng, không thể chỉnh sửa")
    changes = data.model_dump(exclude_unset=True)
    student_ids = changes.pop("student_ids", None)
    if "class_id" in changes:
        _ensure_class(db, changes["class_id"])
    start = changes.get("scheduled_start", row.scheduled_start)
    end = changes.get("scheduled_end", row.scheduled_end)
    if end <= start:
        raise HTTPException(400, "Thời gian kết thúc phải sau thời gian bắt đầu")
    if "status" in changes:
        raise HTTPException(400, "Vui lòng dùng nút mở/đóng buổi học để đổi trạng thái")
    for field, value in changes.items():
        setattr(row, field, value)
    if student_ids is not None:
        _sync_session_students(db, row.id, student_ids)
    db.commit()
    db.refresh(row)
    return _session_out(db, row)


@router.delete("/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = _get_session_or_404(db, session_id)
    has_attendance = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).first()
    if has_attendance:
        raise HTTPException(409, "Không thể xóa buổi học đã có dữ liệu điểm danh")
    db.delete(row)
    db.commit()
    return {"ok": True, "message": "Đã xóa buổi học"}


@router.post("/{session_id}/open", response_model=SessionOut)
def open_session(session_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = _get_session_or_404(db, session_id)
    if row.status == SessionStatus.closed:
        raise HTTPException(400, "Buổi học đã đóng, không thể mở lại")
    if row.scheduled_end <= datetime.utcnow():
        row.status = SessionStatus.closed
        db.commit()
        raise HTTPException(400, "Buổi học đã hết giờ nên không thể mở")
    if row.status == SessionStatus.open:
        return _session_out(db, row)
    db.query(ClassSession).filter(
        ClassSession.class_id == row.class_id,
        ClassSession.id != row.id,
        ClassSession.status == SessionStatus.open,
    ).update({ClassSession.status: SessionStatus.closed}, synchronize_session=False)
    row.status = SessionStatus.open
    db.commit()
    db.refresh(row)
    return _session_out(db, row)


@router.post("/{session_id}/close", response_model=SessionOut)
def close_session(session_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = _get_session_or_404(db, session_id)
    if row.status == SessionStatus.scheduled:
        raise HTTPException(400, "Buổi học chưa mở nên không thể đóng")
    if row.status == SessionStatus.closed:
        return _session_out(db, row)
    row.status = SessionStatus.closed
    db.commit()
    db.refresh(row)
    return _session_out(db, row)
