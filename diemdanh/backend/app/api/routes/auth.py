import re
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import create_token, hash_password, verify_password
from app.models.entities import Student, User, UserRole
from app.schemas.common import LoginIn, TokenOut, UserCreate, UserOut
from app.services.face_service import face_service

router=APIRouter(prefix="/auth", tags=["auth"])


def normalize_student_code(raw: str) -> str:
    settings = get_settings()
    code = raw.strip().upper()
    if not re.fullmatch(settings.student_code_pattern, code):
        raise HTTPException(400, settings.student_code_hint)
    return code
@router.post("/login", response_model=TokenOut)
def login(data: LoginIn, db: Session = Depends(get_db)):
    identifier = data.email.strip()
    if re.fullmatch(get_settings().student_code_pattern, identifier.upper()):
        identifier = f"{identifier.upper().lower()}@student.local"
    user=db.query(User).filter(User.email==identifier).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401,"Mã đăng nhập hoặc mật khẩu không đúng")

    profile_id = user.email
    student_id = None
    if user.role == UserRole.student:
        student = db.query(Student).filter(Student.user_id == user.id).first()
        if student:
            profile_id = student.student_code
            student_id = student.id
    return TokenOut(
        access_token=create_token(str(user.id)),
        role=user.role.value if hasattr(user.role, "value") else str(user.role),
        profile_id=profile_id,
        student_id=student_id,
        full_name=user.full_name,
    )

@router.post("/users", response_model=UserOut)
def create_user(data: UserCreate, db: Session = Depends(get_db)):
    user=User(email=data.email, password_hash=hash_password(data.password), full_name=data.full_name, role=UserRole(data.role))
    db.add(user); db.commit(); db.refresh(user); return user

@router.post("/students/register")
async def register_student(
    student_code: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    code=normalize_student_code(student_code)
    settings = get_settings()
    if len(password) < settings.student_password_min_length:
        raise HTTPException(400, f"Mật khẩu cần tối thiểu {settings.student_password_min_length} ký tự")
    existing_student=db.query(Student).filter(Student.student_code==code).first()
    if existing_student: raise HTTPException(409, "Mã sinh viên đã được đăng ký")
    email=f"{code.lower()}@student.local"
    existing_user=db.query(User).filter(User.email==email).first()
    if existing_user: raise HTTPException(409, "Tài khoản sinh viên đã tồn tại")
    user=User(email=email, password_hash=hash_password(password), full_name=full_name.strip() or code, role=UserRole.student)
    db.add(user); db.commit(); db.refresh(user)
    student=Student(user_id=user.id, student_code=code, status="active")
    db.add(student); db.commit(); db.refresh(student)
    try:
        face=face_service.register(db, student.id, await face_image.read())
    except Exception as exc:
        raise HTTPException(400, f"Không thể đăng ký khuôn mặt: {exc}") from exc
    return {
        "access_token": create_token(str(user.id)),
        "token_type": "bearer",
        "role": "student",
        "profile_id": code,
        "student_id": student.id,
        "full_name": user.full_name,
        "face": face,
    }
