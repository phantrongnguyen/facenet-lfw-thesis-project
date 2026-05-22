from sqlalchemy.orm import Session
from app.core.database import SessionLocal, Base, engine
from app.core.security import hash_password
from app.models.entities import User, UserRole

EMAIL = "gv01@example.com"
PASSWORD = "admin123"
FULL_NAME = "Giáo viên 01"

Base.metadata.create_all(bind=engine)
db: Session = SessionLocal()
try:
    user = db.query(User).filter(User.email == EMAIL).first()
    if user:
        user.password_hash = hash_password(PASSWORD)
        user.full_name = FULL_NAME
        user.role = UserRole.teacher
        action = "updated"
    else:
        user = User(email=EMAIL, password_hash=hash_password(PASSWORD), full_name=FULL_NAME, role=UserRole.teacher)
        db.add(user)
        action = "created"
    db.commit()
    print(f"{action} {EMAIL}")
finally:
    db.close()
