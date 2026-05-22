from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import auth, students, classes, sessions, face, face_changes, attendance, reports

settings=get_settings()
app=FastAPI(title="DiemDanh Face Attendance API", version="0.0.10-dev")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(',')], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/storage", StaticFiles(directory=settings.storage_path), name="storage")

@app.get("/health")
def health(): return {"ok": True}

for router in [auth.router, students.router, classes.router, sessions.router, face.router, face_changes.router, attendance.router, reports.router]:
    app.include_router(router, prefix="/api")
