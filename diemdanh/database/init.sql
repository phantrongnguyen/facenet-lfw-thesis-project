CREATE TYPE userrole AS ENUM ('admin','teacher','student');
CREATE TYPE sessionstatus AS ENUM ('scheduled','open','closed');
CREATE TYPE attendancestatus AS ENUM ('present','late','absent','needs_review');

CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255) NOT NULL,
  role userrole NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS classes (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  code VARCHAR(64) UNIQUE NOT NULL,
  teacher_id INTEGER REFERENCES users(id),
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS students (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  student_code VARCHAR(64) UNIQUE NOT NULL,
  class_id INTEGER REFERENCES classes(id),
  status VARCHAR(32) DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS class_students (
  id SERIAL PRIMARY KEY,
  class_id INTEGER REFERENCES classes(id) NOT NULL,
  student_id INTEGER REFERENCES students(id) NOT NULL,
  CONSTRAINT uq_class_student UNIQUE(class_id, student_id)
);
CREATE TABLE IF NOT EXISTS sessions (
  id SERIAL PRIMARY KEY,
  class_id INTEGER REFERENCES classes(id) NOT NULL,
  title VARCHAR(255) NOT NULL,
  scheduled_start TIMESTAMP NOT NULL,
  scheduled_end TIMESTAMP NOT NULL,
  late_threshold_minutes INTEGER DEFAULT 15,
  status sessionstatus DEFAULT 'scheduled',
  created_by INTEGER REFERENCES users(id) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS face_embeddings (
  id SERIAL PRIMARY KEY,
  student_id INTEGER REFERENCES students(id) NOT NULL,
  model_key VARCHAR(64) NOT NULL,
  embedding BYTEA NOT NULL,
  photo_path VARCHAR(500) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS attendance_records (
  id SERIAL PRIMARY KEY,
  session_id INTEGER REFERENCES sessions(id) NOT NULL,
  student_id INTEGER REFERENCES students(id) NOT NULL,
  confidence FLOAT DEFAULT 0,
  status attendancestatus NOT NULL,
  photo_path VARCHAR(500),
  checked_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT uq_attendance_once UNIQUE(session_id, student_id)
);
CREATE TABLE IF NOT EXISTS manual_review_logs (
  id SERIAL PRIMARY KEY,
  attendance_id INTEGER REFERENCES attendance_records(id) NOT NULL,
  reviewer_id INTEGER REFERENCES users(id) NOT NULL,
  old_status VARCHAR(32) NOT NULL,
  new_status VARCHAR(32) NOT NULL,
  reason TEXT NOT NULL,
  reviewed_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_sessions_class_status ON sessions(class_id, status);
CREATE INDEX IF NOT EXISTS ix_face_embeddings_student_id ON face_embeddings(student_id);
