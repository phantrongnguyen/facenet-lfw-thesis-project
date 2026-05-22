from datetime import datetime
from pydantic import BaseModel, EmailStr

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str | None = None
    profile_id: str | None = None
    student_id: int | None = None
    full_name: str | None = None

class UserOut(BaseModel):
    id: int; email: EmailStr; full_name: str; role: str; is_active: bool
    class Config: from_attributes=True

class LoginIn(BaseModel):
    email: str; password: str

class UserCreate(BaseModel):
    email: EmailStr; password: str; full_name: str; role: str

class StudentCreate(BaseModel):
    student_code: str; full_name: str; email: EmailStr | None = None; class_id: int | None = None; status: str = "active"

class StudentUpdate(BaseModel):
    student_code: str | None = None; full_name: str | None = None; class_id: int | None = None; status: str | None = None

class StudentOut(BaseModel):
    id:int; student_code:str; full_name:str|None=None; status:str; class_id:int|None=None; class_name:str|None=None
    class Config: from_attributes=True

class ClassCreate(BaseModel):
    name: str; code: str; teacher_id: int | None = None; description: str | None = None

class ClassUpdate(BaseModel):
    name: str | None = None; code: str | None = None; teacher_id: int | None = None; description: str | None = None

class ClassOut(BaseModel):
    id:int; name:str; code:str; teacher_id:int|None=None; description:str|None=None
    class Config: from_attributes=True

class SessionStudentOut(BaseModel):
    id:int; student_code:str; full_name:str|None=None; status:str|None=None; class_id:int|None=None; class_name:str|None=None; attendance_status:str|None=None; checked_at:datetime|None=None

class SessionCreate(BaseModel):
    class_id:int; title:str; scheduled_start:datetime; scheduled_end:datetime; late_threshold_minutes:int=15; student_ids:list[int]=[]

class SessionUpdate(BaseModel):
    class_id:int|None=None; title:str|None=None; scheduled_start:datetime|None=None; scheduled_end:datetime|None=None; late_threshold_minutes:int|None=None; status:str|None=None; student_ids:list[int]|None=None

class SessionOut(BaseModel):
    id:int; class_id:int; class_name:str|None=None; title:str; scheduled_start:datetime; scheduled_end:datetime; late_threshold_minutes:int; status:str; student_count:int=0; present_count:int=0; attendance_progress:str="0/0"; remaining_seconds:int=0; students:list[SessionStudentOut]=[]
    class Config: from_attributes=True

class ReviewIn(BaseModel):
    new_status: str; reason: str
