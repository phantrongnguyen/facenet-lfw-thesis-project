from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///./diemdanh.db"
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    default_face_model: str = "Facenet512_retinaface"
    face_match_threshold: float = 0.407157
    project_root: str = "../.."
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
    max_upload_mb: int = 5
    student_code_pattern: str = r"^SV\d{3}$"
    student_code_hint: str = "Mã sinh viên phải có dạng SV000 đến SV999"
    student_password_min_length: int = 6

    @property
    def root_path(self) -> Path:
        return Path(__file__).resolve().parents[4]

    @property
    def storage_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "storage"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
