# DiemDanh AI — Face Attendance App

Full-stack app điểm danh khuôn mặt cho lớp học.

## Stack

- Frontend: React + Vite + TailwindCSS
- Backend: FastAPI
- Database: PostgreSQL + SQLAlchemy + Alembic
- AI: DeepFace + FaceNet/Facenet512 pipeline có sẵn trong dự án

## Điểm tối ưu nhận diện

Backend dùng `FaceService` tại `backend/app/services/face_service.py`:

1. Đăng ký: extract embedding 1 lần rồi lưu vào PostgreSQL.
2. Server runtime: load embeddings vào RAM dưới dạng NumPy matrix.
3. Điểm danh: chỉ gọi DeepFace 1 lần cho ảnh camera/upload.
4. So sánh: dùng vectorized cosine similarity `matrix @ query`.

Model key hỗ trợ:

- `Facenet_mtcnn`
- `Facenet_retinaface`
- `Facenet512_mtcnn`
- `Facenet512_retinaface`

Mặc định: `Facenet512_mtcnn`.

## Cấu trúc

```text
backend/    FastAPI API, SQLAlchemy models, services, routes
frontend/   React/Vite/Tailwind UI
database/   init.sql schema
```

## Chạy backend

```powershell
cd diemdanh/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

## Chạy frontend

```powershell
cd diemdanh/frontend
npm install
copy .env.example .env
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Database

Tạo database PostgreSQL tên `diemdanh`, sau đó dùng một trong hai cách:

### Cách 1: chạy SQL trực tiếp

```powershell
psql -U postgres -d diemdanh -f ..\database\init.sql
```

### Cách 2: Alembic

```powershell
cd diemdanh/backend
alembic upgrade head
```

## API chính

- `POST /api/auth/login`
- `POST /api/auth/users`
- `GET/POST /api/students`
- `GET/POST /api/classes`
- `GET/POST /api/sessions`
- `POST /api/sessions/{id}/open`
- `POST /api/sessions/{id}/close`
- `POST /api/face/register/{student_id}`
- `POST /api/face/reload`
- `POST /api/attendance/verify/{session_id}`
- `GET /api/attendance/session/{session_id}`
- `POST /api/attendance/{attendance_id}/review`
- `GET /api/reports/session/{session_id}.csv`

## MVP hiện tại

Đã có:

- Backend API foundation
- ORM schema + init SQL + Alembic scaffold
- Auth JWT/password hashing
- Face service wrap pipeline DeepFace hiện có
- Attendance service chống điểm danh trùng
- Frontend login/dashboard/camera attendance MVP

Cần hoàn thiện thêm ở phase tiếp theo:

- Full CRUD UI cho sinh viên/lớp/buổi học
- Seed admin user
- UI manual review chi tiết
- Excel export
- Rate limiting thực tế
- Liveness detection phase 2
