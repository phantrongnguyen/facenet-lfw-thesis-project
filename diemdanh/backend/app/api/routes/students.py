from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.api.routes.auth import normalize_student_code
from app.core.database import get_db
from app.core.deps import require_roles
from app.models.entities import Class, Student, User, UserRole
from app.schemas.common import StudentCreate, StudentOut, StudentUpdate

router = APIRouter(prefix="/students", tags=["students"])


def _student_out(row: Student) -> StudentOut:
    return StudentOut(
        id=row.id,
        student_code=row.student_code,
        full_name=row.user.full_name if row.user else "Chưa cập nhật",
        status=row.status,
        class_id=row.class_id,
        class_name=row.klass.name if row.klass else None,
    )


def _ensure_class(db: Session, class_id: int | None):
    if class_id is None:
        return
    if not db.get(Class, class_id):
        raise HTTPException(404, "Không tìm thấy lớp học")


@router.get("", response_model=list[StudentOut])
def list_students(db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    rows = db.query(Student).order_by(Student.id.desc()).all()
    return [_student_out(row) for row in rows]


@router.post("", response_model=StudentOut)
def create_student(data: StudentCreate, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    code = normalize_student_code(data.student_code)
    if db.query(Student).filter(Student.student_code == code).first():
        raise HTTPException(409, "Mã sinh viên đã tồn tại")
    _ensure_class(db, data.class_id)
    user = None
    if data.email:
        if db.query(User).filter(User.email == data.email).first():
            raise HTTPException(409, "Email đã tồn tại")
        user = User(email=data.email, password_hash="", full_name=data.full_name.strip() or code, role=UserRole.student)
        db.add(user)
        db.flush()
    row = Student(user_id=user.id if user else None, student_code=code, class_id=data.class_id, status=data.status)
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Dữ liệu sinh viên bị trùng") from exc
    db.refresh(row)
    return _student_out(row)


@router.put("/{student_id}", response_model=StudentOut)
def update_student(student_id: int, data: StudentUpdate, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = db.get(Student, student_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy sinh viên")
    if data.student_code is not None:
        code = normalize_student_code(data.student_code)
        duplicate = db.query(Student).filter(Student.student_code == code, Student.id != student_id).first()
        if duplicate:
            raise HTTPException(409, "Mã sinh viên đã tồn tại")
        row.student_code = code
    if data.class_id is not None:
        _ensure_class(db, data.class_id)
        row.class_id = data.class_id
    if data.status is not None:
        row.status = data.status
    if data.full_name is not None:
        if row.user:
            row.user.full_name = data.full_name.strip() or row.student_code
        else:
            user = User(email=f"{row.student_code.lower()}@student.local", password_hash="", full_name=data.full_name.strip() or row.student_code, role=UserRole.student)
            db.add(user)
            db.flush()
            row.user_id = user.id
    db.commit()
    db.refresh(row)
    return _student_out(row)


@router.delete("/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = db.get(Student, student_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy sinh viên")
    db.delete(row)
    db.commit()
    return {"ok": True}
