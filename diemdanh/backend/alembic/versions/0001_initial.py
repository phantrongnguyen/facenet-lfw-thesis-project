"""initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.execute(open("../database/init.sql", encoding="utf-8").read())

def downgrade():
    for table in ["manual_review_logs","attendance_records","face_embeddings","sessions","class_students","students","classes","users"]:
        op.drop_table(table)
    op.execute("DROP TYPE IF EXISTS attendancestatus")
    op.execute("DROP TYPE IF EXISTS sessionstatus")
    op.execute("DROP TYPE IF EXISTS userrole")
