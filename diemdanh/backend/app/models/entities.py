import enum
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class UserRole(str, enum.Enum): admin="admin"; teacher="teacher"; student="student"
class SessionStatus(str, enum.Enum): scheduled="scheduled"; open="open"; closed="closed"
class AttendanceStatus(str, enum.Enum): present="present"; late="late"; absent="absent"; needs_review="needs_review"
class FacePhotoChangeStatus(str, enum.Enum): pending="pending"; approved="approved"; rejected="rejected"; auto_approved="auto_approved"
class ManualAttendanceRequestStatus(str, enum.Enum): pending="pending"; approved="approved"; rejected="rejected"

class User(Base):
    __tablename__="users"
    id: Mapped[int]=mapped_column(primary_key=True)
    email: Mapped[str]=mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str]=mapped_column(String(255))
    full_name: Mapped[str]=mapped_column(String(255))
    role: Mapped[UserRole]=mapped_column(Enum(UserRole), index=True)
    is_active: Mapped[bool]=mapped_column(Boolean, default=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)

class Class(Base):
    __tablename__="classes"
    id: Mapped[int]=mapped_column(primary_key=True)
    name: Mapped[str]=mapped_column(String(255))
    code: Mapped[str]=mapped_column(String(64), unique=True, index=True)
    teacher_id: Mapped[int | None]=mapped_column(ForeignKey("users.id"), nullable=True)
    description: Mapped[str | None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    teacher=relationship("User")
    students=relationship("Student", back_populates="klass")
    sessions=relationship("Session", back_populates="klass")

class Student(Base):
    __tablename__="students"
    id: Mapped[int]=mapped_column(primary_key=True)
    user_id: Mapped[int | None]=mapped_column(ForeignKey("users.id"), nullable=True)
    student_code: Mapped[str]=mapped_column(String(64), unique=True, index=True)
    class_id: Mapped[int | None]=mapped_column(ForeignKey("classes.id"), nullable=True)
    status: Mapped[str]=mapped_column(String(32), default="active")
    user=relationship("User")
    klass=relationship("Class", back_populates="students")

class ClassStudent(Base):
    __tablename__="class_students"
    id: Mapped[int]=mapped_column(primary_key=True)
    class_id: Mapped[int]=mapped_column(ForeignKey("classes.id"))
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"))
    __table_args__=(UniqueConstraint("class_id","student_id",name="uq_class_student"),)

class Session(Base):
    __tablename__="sessions"
    id: Mapped[int]=mapped_column(primary_key=True)
    class_id: Mapped[int]=mapped_column(ForeignKey("classes.id"), index=True)
    title: Mapped[str]=mapped_column(String(255))
    scheduled_start: Mapped[datetime]=mapped_column(DateTime)
    scheduled_end: Mapped[datetime]=mapped_column(DateTime)
    late_threshold_minutes: Mapped[int]=mapped_column(Integer, default=15)
    status: Mapped[SessionStatus]=mapped_column(Enum(SessionStatus), default=SessionStatus.scheduled, index=True)
    created_by: Mapped[int]=mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    klass=relationship("Class", back_populates="sessions")
    creator=relationship("User")
    enrolled_students=relationship("SessionStudent", cascade="all, delete-orphan", back_populates="session")
    __table_args__=(Index("ix_sessions_class_status","class_id","status"),)

class SessionStudent(Base):
    __tablename__="session_students"
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey("sessions.id"), index=True)
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"), index=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    session=relationship("Session", back_populates="enrolled_students")
    student=relationship("Student")
    __table_args__=(UniqueConstraint("session_id","student_id",name="uq_session_student"),)

class FaceEmbedding(Base):
    __tablename__="face_embeddings"
    id: Mapped[int]=mapped_column(primary_key=True)
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"), index=True)
    model_key: Mapped[str]=mapped_column(String(64), index=True)
    embedding: Mapped[bytes]=mapped_column(LargeBinary)
    photo_path: Mapped[str]=mapped_column(String(500))
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)

class FacePhotoChangeRequest(Base):
    __tablename__="face_photo_change_requests"
    id: Mapped[int]=mapped_column(primary_key=True)
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"), index=True)
    old_photo_path: Mapped[str | None]=mapped_column(String(500), nullable=True)
    new_photo_path: Mapped[str]=mapped_column(String(500))
    new_embedding: Mapped[bytes]=mapped_column(LargeBinary)
    model_key: Mapped[str]=mapped_column(String(64), index=True)
    confidence: Mapped[float | None]=mapped_column(Float, nullable=True)
    status: Mapped[FacePhotoChangeStatus]=mapped_column(Enum(FacePhotoChangeStatus), default=FacePhotoChangeStatus.pending, index=True)
    note: Mapped[str | None]=mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None]=mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None]=mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    student=relationship("Student")
    reviewer=relationship("User")

class AttendanceRecord(Base):
    __tablename__="attendance_records"
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey("sessions.id"), index=True)
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"), index=True)
    confidence: Mapped[float]=mapped_column(Float, default=0)
    status: Mapped[AttendanceStatus]=mapped_column(Enum(AttendanceStatus), index=True)
    photo_path: Mapped[str | None]=mapped_column(String(500), nullable=True)
    checked_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    __table_args__=(UniqueConstraint("session_id","student_id",name="uq_attendance_once"),)

class ManualAttendanceRequest(Base):
    __tablename__="manual_attendance_requests"
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey("sessions.id"), index=True)
    student_id: Mapped[int]=mapped_column(ForeignKey("students.id"), index=True)
    photo_path: Mapped[str]=mapped_column(String(500))
    reason: Mapped[str | None]=mapped_column(Text, nullable=True)
    confidence: Mapped[float | None]=mapped_column(Float, nullable=True)
    status: Mapped[ManualAttendanceRequestStatus]=mapped_column(Enum(ManualAttendanceRequestStatus), default=ManualAttendanceRequestStatus.pending, index=True)
    reviewed_by: Mapped[int | None]=mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None]=mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
    session=relationship("Session")
    student=relationship("Student")
    reviewer=relationship("User")
    __table_args__=(UniqueConstraint("session_id","student_id","status",name="uq_manual_attendance_pending"),)

class ManualReviewLog(Base):
    __tablename__="manual_review_logs"
    id: Mapped[int]=mapped_column(primary_key=True)
    attendance_id: Mapped[int]=mapped_column(ForeignKey("attendance_records.id"))
    reviewer_id: Mapped[int]=mapped_column(ForeignKey("users.id"))
    old_status: Mapped[str]=mapped_column(String(32))
    new_status: Mapped[str]=mapped_column(String(32))
    reason: Mapped[str]=mapped_column(Text)
    reviewed_at: Mapped[datetime]=mapped_column(DateTime, default=datetime.utcnow)
