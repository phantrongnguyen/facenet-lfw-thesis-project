from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.deps import require_roles
from app.models.entities import Class, Session as ClassSession, Student
from app.schemas.common import ClassCreate, ClassOut, ClassUpdate

router = APIRouter(prefix="/classes", tags=["classes"])


@router.get("", response_model=list[ClassOut])
def list_classes(db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    return db.query(Class).order_by(Class.id.desc()).all()


@router.post("", response_model=ClassOut)
def create_class(data: ClassCreate, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    code = data.code.strip().upper()
    if not code:
        raise HTTPException(400, "Mã lớp không hợp lệ")
    if db.query(Class).filter(Class.code == code).first():
        raise HTTPException(409, "Mã lớp đã tồn tại")
    row = Class(**data.model_dump(exclude={"code"}), code=code)
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "Dữ liệu lớp học bị trùng") from exc
    db.refresh(row)
    return row


@router.put("/{class_id}", response_model=ClassOut)
def update_class(class_id: int, data: ClassUpdate, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = db.get(Class, class_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy lớp học")
    changes = data.model_dump(exclude_unset=True)
    if "code" in changes:
        code = changes["code"].strip().upper()
        if not code:
            raise HTTPException(400, "Mã lớp không hợp lệ")
        duplicate = db.query(Class).filter(Class.code == code, Class.id != class_id).first()
        if duplicate:
            raise HTTPException(409, "Mã lớp đã tồn tại")
        row.code = code
    for field in ["name", "teacher_id", "description"]:
        if field in changes:
            setattr(row, field, changes[field])
    db.commit()
    db.refresh(row)
    return row


@router.delete("/{class_id}")
def delete_class(class_id: int, db: Session = Depends(get_db), _=Depends(require_roles("admin", "teacher"))):
    row = db.get(Class, class_id)
    if not row:
        raise HTTPException(404, "Không tìm thấy lớp học")
    has_students = db.query(Student).filter(Student.class_id == class_id).first()
    has_sessions = db.query(ClassSession).filter(ClassSession.class_id == class_id).first()
    if has_students or has_sessions:
        raise HTTPException(409, "Không thể xóa lớp đang có sinh viên hoặc buổi học")
    db.delete(row)
    db.commit()
    return {"ok": True}
