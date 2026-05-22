import csv
import io
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import require_roles
from app.models.entities import AttendanceRecord, AttendanceStatus, ClassStudent, Session as ClassSession, Student
from app.services.attendance_service import STATUS_LABELS

router = APIRouter(prefix="/reports", tags=["reports"])


def _get_session_or_404(db: Session, session_id: int) -> ClassSession:
    row = db.get(ClassSession, session_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy buổi học")
    return row


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _student_name(student: Student) -> str | None:
    return student.user.full_name if getattr(student, "user", None) else None


def _session_report_rows(db: Session, session: ClassSession) -> tuple[list[dict], dict]:
    class_students = (
        db.query(Student)
        .join(ClassStudent, ClassStudent.student_id == Student.id)
        .filter(ClassStudent.class_id == session.class_id)
        .order_by(Student.student_code.asc())
        .all()
    )
    records = {
        record.student_id: record
        for record in db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session.id).all()
    }
    summary = {"total": len(class_students), "present": 0, "late": 0, "needs_review": 0, "absent": 0, "checked": 0}
    rows: list[dict] = []
    for student in class_students:
        record = records.get(student.id)
        if record:
            status = record.status if isinstance(record.status, AttendanceStatus) else AttendanceStatus(record.status)
            summary[_status_value(status)] += 1
            summary["checked"] += 1
            rows.append({
                "student_id": student.id,
                "student_code": student.student_code,
                "full_name": _student_name(student),
                "status": status.value,
                "status_label": STATUS_LABELS[status],
                "confidence": record.confidence,
                "checked_at": record.checked_at,
                "photo_path": record.photo_path,
                "attendance_id": record.id,
            })
        else:
            summary["absent"] += 1
            rows.append({
                "student_id": student.id,
                "student_code": student.student_code,
                "full_name": _student_name(student),
                "status": AttendanceStatus.absent.value,
                "status_label": STATUS_LABELS[AttendanceStatus.absent],
                "confidence": None,
                "checked_at": None,
                "photo_path": None,
                "attendance_id": None,
            })
    return rows, summary


@router.get("/session/{session_id}")
def session_report(session_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    session = _get_session_or_404(db, session_id)
    rows, summary = _session_report_rows(db, session)
    return {
        "session": {
            "id": session.id,
            "class_id": session.class_id,
            "class_name": session.klass.name if getattr(session, "klass", None) else None,
            "title": session.title,
            "status": _status_value(session.status),
            "scheduled_start": session.scheduled_start,
            "scheduled_end": session.scheduled_end,
            "late_threshold_minutes": session.late_threshold_minutes,
        },
        "summary": summary,
        "rows": rows,
    }


@router.get("/session/{session_id}.csv")
def export_session_csv(session_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    session = _get_session_or_404(db, session_id)
    rows, _ = _session_report_rows(db, session)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["student_code", "full_name", "status", "status_label", "confidence", "checked_at"])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    return Response(
        output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}_attendance.csv"},
    )
