"""Native desktop face recognition app with realtime, registration, image search, and database tools."""

import os
import sys
import time
import tempfile
import shutil
import pickle

import cv2
import numpy as np
from types import SimpleNamespace

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for path in (PROJECT_ROOT, APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from PySide6.QtCore import Qt, QSize, QTimer
    from PySide6.QtGui import QFont, QPixmap, QImage
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
        QLineEdit,
    )
except ModuleNotFoundError:
    print("PySide6 chưa được cài. Chạy: python -m pip install PySide6")
    raise

import app3_utils as utils
from app1_utils import verify_faces
from desktop_realtime_engine import RealtimeConfig, RealtimeWorker


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png)"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
DATASET_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DATASET_SEARCH_MODEL_OPTIONS = [
    ("Facenet + MTCNN", "Facenet_mtcnn"),
    ("Facenet + RetinaFace", "Facenet_retinaface"),
    ("Facenet512 + MTCNN", "Facenet512_mtcnn"),
    ("Facenet512 + RetinaFace", "Facenet512_retinaface"),
]
DATASET_SEARCH_MODELS = [label for label, _ in DATASET_SEARCH_MODEL_OPTIONS]
DATASET_SEARCH_MODEL_CODES = dict(DATASET_SEARCH_MODEL_OPTIONS)


class LocalImageUpload:
    """Tiny file wrapper compatible with app helpers expecting Streamlit uploads."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)

    def getbuffer(self):
        with open(self.path, "rb") as f:
            return f.read()

    def read(self, *args, **kwargs):
        with open(self.path, "rb") as f:
            return f.read(*args, **kwargs)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—"):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.value.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str):
        self.value.setText(value)


class DesktopFaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.static_camera = None
        self.static_camera_frame = None
        self.static_timer = QTimer(self)
        self.static_timer.timeout.connect(self.update_static_camera_preview)
        self.last_deleted_backup = None
        self.reg_image_path = None
        self.reg_camera = None
        self.reg_camera_frame = None
        self.reg_timer = QTimer(self)
        self.reg_timer.timeout.connect(self.update_register_camera_preview)
        self.query_image_path = None
        self.compare_image_a = None
        self.compare_image_b = None
        self.compare_camera = None
        self.compare_camera_frame = None
        self.compare_timer = QTimer(self)
        self.compare_timer.timeout.connect(self.update_compare_camera_preview)
        self.current_theme = "Sáng"
        self.thresholds = utils.get_available_thresholds()
        self.setWindowTitle("App3 Desktop Face Recognition")
        self.setMinimumSize(QSize(1280, 820))
        self._build_ui()
        self._apply_styles()
        self._load_initial_state()

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_register_tab(), "👤 Đăng ký")
        self.tabs.addTab(self._build_static_tab(), "🖼️ Ảnh tĩnh")
        self.tabs.addTab(self._build_realtime_tab(), "🎥 Realtime")
        self.tabs.addTab(self._build_compare_tab(), "🧪 So sánh")
        self.tabs.addTab(self._build_search_tab(), "🔎 Tìm kiếm")
        self.tabs.addTab(self._build_database_tab(), "🧠 Database")
        self.tabs.addTab(self._build_model_info_tab(), "📊 Thông tin mô hình")
        self.tabs.addTab(self._build_settings_tab(), "⚙️ Cài đặt")
        self.tabs.addTab(self._build_help_tab(), "❔ Hướng dẫn")

    def _build_realtime_tab(self):
        page = QWidget()
        main = QVBoxLayout(page)
        main.setContentsMargins(22, 22, 22, 22)
        main.setSpacing(12)

        title = QLabel("Realtime Face Recognition")
        title.setObjectName("heroTitle")
        main.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(16)
        main.addLayout(body, 1)

        video_area = QVBoxLayout()
        video_area.setSpacing(10)
        body.addLayout(video_area, 3)

        self.video_label = QLabel("Bấm START để mở camera")
        self.video_label.setObjectName("videoSurface")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(820, 540)
        video_area.addWidget(self.video_label, 1)

        self.status_label = QLabel("Sẵn sàng.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        video_area.addWidget(self.status_label)

        controls = QFrame()
        controls.setObjectName("realtimeControlPanel")
        controls.setMaximumWidth(410)
        controls.setMinimumWidth(340)
        control_layout = QVBoxLayout(controls)
        control_layout.setSpacing(8)
        body.addWidget(controls, 1)

        panel_title = QLabel("Control Center")
        panel_title.setObjectName("panelTitleSmall")
        control_layout.addWidget(panel_title)

        self.model_combo = QComboBox()
        self.model_combo.addItems(utils.ALL_RECOGNITION_MODELS)
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(12)
        self.model_combo.currentTextChanged.connect(self._sync_threshold)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setRange(0.05, 0.95)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 5)
        self.camera_spin.setValue(0)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["DirectShow", "MSMF", "Auto"])

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.5, 6.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(2.0)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 30)
        self.fps_spin.setValue(24)

        fields = [
            ("Model", self.model_combo),
            ("Threshold", self.threshold_spin),
            ("Camera", self.camera_spin),
            ("Backend", self.backend_combo),
            ("Recognize / sec", self.interval_spin),
            ("Max FPS", self.fps_spin),
        ]
        for label, widget in fields:
            row = QHBoxLayout()
            row.setSpacing(8)
            label_widget = QLabel(label)
            label_widget.setObjectName("compactHintLabel")
            label_widget.setFixedWidth(105)
            row.addWidget(label_widget)
            row.addWidget(widget, 1)
            control_layout.addLayout(row)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        self.default_btn = QPushButton("↺ Default")
        self.default_btn.clicked.connect(self.reset_realtime_defaults)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.default_btn)
        control_layout.addLayout(buttons)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)
        control_layout.addLayout(metrics)
        self.name_card = MetricCard("Identity", "—")
        self.sim_card = MetricCard("Similarity", "—")
        self.fps_card = MetricCard("Display FPS", "—")
        self.cache_card = MetricCard("Cache", "—")
        for i, card in enumerate((self.name_card, self.sim_card, self.fps_card, self.cache_card)):
            metrics.addWidget(card, i, 0)
        control_layout.addStretch(1)
        return page

    def _build_register_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        form = QFrame()
        form.setObjectName("controlPanel")
        form_layout = QVBoxLayout(form)
        layout.addWidget(form, 1)

        title = QLabel("Đăng ký khuôn mặt")
        title.setObjectName("panelTitle")
        form_layout.addWidget(title)
        form_layout.addWidget(QLabel("Tên không dấu"))
        self.reg_name_input = QLineEdit()
        self.reg_name_input.setPlaceholderText("Ví dụ: Gia Bao")
        form_layout.addWidget(self.reg_name_input)

        self.reg_model_combo = QComboBox()
        self.reg_model_combo.addItems(utils.ALL_RECOGNITION_MODELS)
        self.reg_model_combo.setCurrentText(utils.ALL_RECOGNITION_MODELS[0])
        self.reg_model_combo.setMinimumWidth(240)
        form_layout.addWidget(QLabel("Model đăng ký"))
        form_layout.addWidget(self.reg_model_combo)

        self.reg_gate_check = QCheckBox("Chặn ảnh nếu không giống user đã đăng ký")
        self.reg_gate_check.setChecked(True)
        form_layout.addWidget(self.reg_gate_check)

        self.reg_manual_threshold_check = QCheckBox("Tùy chỉnh ngưỡng đăng ký thủ công")
        self.reg_manual_threshold_check.setChecked(False)
        form_layout.addWidget(self.reg_manual_threshold_check)

        reg_threshold_row = QHBoxLayout()
        reg_threshold_row.addWidget(QLabel("Ngưỡng đăng ký"))
        self.reg_threshold_spin = QDoubleSpinBox()
        self.reg_threshold_spin.setDecimals(3)
        self.reg_threshold_spin.setSingleStep(0.01)
        self.reg_threshold_spin.setRange(0.05, 0.95)
        self.reg_threshold_spin.setValue(self._model_default_threshold(self.reg_model_combo.currentText()))
        self.reg_threshold_spin.setEnabled(False)
        self.reg_manual_threshold_check.toggled.connect(self.reg_threshold_spin.setEnabled)
        self.reg_model_combo.currentTextChanged.connect(self._sync_registration_threshold)
        reg_threshold_row.addWidget(self.reg_threshold_spin, 1)
        form_layout.addLayout(reg_threshold_row)

        pick_btn = QPushButton("📁 Chọn ảnh đăng ký")
        pick_btn.clicked.connect(self.pick_register_image)
        form_layout.addWidget(pick_btn)

        camera_buttons = QHBoxLayout()
        open_reg_camera_btn = QPushButton("🎥 Mở camera")
        open_reg_camera_btn.clicked.connect(self.open_register_camera)
        capture_reg_btn = QPushButton("📸 Chụp ảnh đăng ký")
        capture_reg_btn.clicked.connect(self.capture_register_image)
        clear_reg_btn = QPushButton("🧹 Xóa ảnh")
        clear_reg_btn.setObjectName("stopButton")
        clear_reg_btn.clicked.connect(self.clear_register_image)
        camera_buttons.addWidget(open_reg_camera_btn)
        camera_buttons.addWidget(capture_reg_btn)
        camera_buttons.addWidget(clear_reg_btn)
        form_layout.addLayout(camera_buttons)

        save_btn = QPushButton("💾 Đăng ký và tạo embedding")
        save_btn.setObjectName("startButton")
        save_btn.clicked.connect(self.register_selected_face)
        form_layout.addWidget(save_btn)

        self.reg_status = QTextEdit()
        self.reg_status.setReadOnly(True)
        self.reg_status.setPlaceholderText("Kết quả đăng ký sẽ hiển thị ở đây...")
        form_layout.addWidget(self.reg_status, 1)

        self.reg_preview = QLabel("Chưa chọn ảnh")
        self.reg_preview.setObjectName("imagePreview")
        self.reg_preview.setAlignment(Qt.AlignCenter)
        self.reg_preview.setMinimumSize(520, 420)
        self.reg_preview.setMaximumHeight(680)
        layout.addWidget(self.reg_preview, 2)
        return page

    def _build_static_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        left = QFrame()
        left.setObjectName("controlPanel")
        left_layout = QVBoxLayout(left)
        layout.addWidget(left, 1)

        title = QLabel("Nhận diện ảnh tĩnh")
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)

        hint = QLabel("Chọn ảnh từ máy hoặc chụp nhanh bằng camera, sau đó app sẽ xuất ảnh có khung detect.")
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        left_layout.addWidget(hint)

        self.static_model_combo = QComboBox()
        self.static_model_combo.addItems(utils.ALL_RECOGNITION_MODELS)
        self.static_model_combo.setCurrentText(utils.ALL_RECOGNITION_MODELS[0])
        self.static_model_combo.setMinimumWidth(240)
        left_layout.addWidget(QLabel("Model nhận diện"))
        left_layout.addWidget(self.static_model_combo)

        self.show_vector_check = QCheckBox("Hiển thị vector embedding query")
        self.show_vector_check.setChecked(False)
        left_layout.addWidget(self.show_vector_check)

        pick_btn = QPushButton("📁 Chọn ảnh query")
        pick_btn.clicked.connect(self.pick_query_image)
        left_layout.addWidget(pick_btn)

        camera_buttons = QHBoxLayout()
        open_camera_btn = QPushButton("🎥 Mở camera")
        open_camera_btn.clicked.connect(self.open_static_camera)
        self.snap_btn = QPushButton("📸 Chụp ảnh")
        self.snap_btn.clicked.connect(self.capture_query_image)
        clear_btn = QPushButton("🧹 Xóa ảnh")
        clear_btn.setObjectName("stopButton")
        clear_btn.clicked.connect(self.clear_query_image)
        camera_buttons.addWidget(open_camera_btn)
        camera_buttons.addWidget(self.snap_btn)
        camera_buttons.addWidget(clear_btn)
        left_layout.addLayout(camera_buttons)

        run_btn = QPushButton("🔍 Nhận diện & xuất ảnh detect")
        run_btn.setObjectName("startButton")
        run_btn.clicked.connect(self.recognize_query_image)
        left_layout.addWidget(run_btn)

        self.static_result = QTextEdit()
        self.static_result.setReadOnly(True)
        self.static_result.setPlaceholderText("Kết quả nhận diện và top matches...")
        left_layout.addWidget(self.static_result, 1)

        preview_panel = QVBoxLayout()
        layout.addLayout(preview_panel, 2)

        self.query_preview = QLabel("Ảnh gốc / camera preview")
        self.query_preview.setObjectName("imagePreview")
        self.query_preview.setAlignment(Qt.AlignCenter)
        self.query_preview.setMinimumSize(500, 280)
        self.query_preview.setMaximumHeight(520)
        preview_panel.addWidget(self.query_preview, 1)

        self.detect_preview = QLabel("Ảnh detect sẽ hiện ở đây")
        self.detect_preview.setObjectName("imagePreview")
        self.detect_preview.setAlignment(Qt.AlignCenter)
        self.detect_preview.setMinimumSize(500, 280)
        self.detect_preview.setMaximumHeight(520)
        preview_panel.addWidget(self.detect_preview, 1)
        return page

    def _build_compare_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        left = QFrame()
        left.setObjectName("controlPanel")
        left_layout = QVBoxLayout(left)
        layout.addWidget(left, 1)

        title = QLabel("So sánh 2 ảnh")
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)
        hint = QLabel(
            "Hỗ trợ ảnh có nhiều khuôn mặt: app sẽ detect tất cả mặt trong A/B, "
            "so từng cặp và đánh dấu cặp khớp tốt nhất."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        left_layout.addWidget(hint)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model so sánh"))
        self.compare_model_combo = QComboBox()
        self.compare_model_combo.addItems(utils.ALL_RECOGNITION_MODELS)
        self.compare_model_combo.setCurrentText(self.model_combo.currentText())
        self.compare_model_combo.setMinimumWidth(240)
        model_row.addWidget(self.compare_model_combo, 1)
        left_layout.addLayout(model_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold so sánh"))
        self.compare_threshold_spin = QDoubleSpinBox()
        self.compare_threshold_spin.setDecimals(3)
        self.compare_threshold_spin.setSingleStep(0.01)
        self.compare_threshold_spin.setRange(0.05, 0.95)
        self.compare_threshold_spin.setValue(self._compare_default_threshold(self.compare_model_combo.currentText()))
        self.compare_model_combo.currentTextChanged.connect(
            lambda model: self.compare_threshold_spin.setValue(self._compare_default_threshold(model))
        )
        threshold_row.addWidget(self.compare_threshold_spin, 1)
        left_layout.addLayout(threshold_row)

        pair_row = QHBoxLayout()
        img_a_btn = QPushButton("📷 Chọn ảnh A")
        img_a_btn.clicked.connect(lambda: self.pick_compare_image("a"))
        img_b_btn = QPushButton("📷 Chọn ảnh B")
        img_b_btn.clicked.connect(lambda: self.pick_compare_image("b"))
        pair_row.addWidget(img_a_btn)
        pair_row.addWidget(img_b_btn)
        left_layout.addLayout(pair_row)

        camera_row = QHBoxLayout()
        compare_camera_btn = QPushButton("🎥 Mở camera")
        compare_camera_btn.clicked.connect(self.open_compare_camera)
        capture_a_btn = QPushButton("📸 Chụp A")
        capture_a_btn.clicked.connect(lambda: self.capture_compare_image("a"))
        capture_b_btn = QPushButton("📸 Chụp B")
        capture_b_btn.clicked.connect(lambda: self.capture_compare_image("b"))
        close_camera_btn = QPushButton("✕ Tắt camera")
        close_camera_btn.setObjectName("stopButton")
        close_camera_btn.clicked.connect(self.close_compare_camera)
        camera_row.addWidget(compare_camera_btn)
        camera_row.addWidget(capture_a_btn)
        camera_row.addWidget(capture_b_btn)
        camera_row.addWidget(close_camera_btn)
        left_layout.addLayout(camera_row)

        detail_btn = QPushButton("🔎 Xem chi tiết ảnh A/B")
        detail_btn.clicked.connect(self.show_compare_details)
        left_layout.addWidget(detail_btn)

        compare_btn = QPushButton("🧪 So sánh A và B")
        compare_btn.setObjectName("startButton")
        compare_btn.clicked.connect(self.compare_two_images)
        left_layout.addWidget(compare_btn)

        self.compare_result = QTextEdit()
        self.compare_result.setReadOnly(True)
        self.compare_result.setPlaceholderText("Kết quả so sánh sẽ hiển thị ở đây...")
        left_layout.addWidget(self.compare_result, 1)

        previews = QVBoxLayout()
        previews.setSpacing(12)
        layout.addLayout(previews, 2)

        self.compare_preview_camera = QLabel("Camera so sánh preview")
        self.compare_preview_camera.setObjectName("imagePreview")
        self.compare_preview_camera.setAlignment(Qt.AlignCenter)
        self.compare_preview_camera.setMinimumSize(640, 420)
        previews.addWidget(self.compare_preview_camera, 4)

        pair_preview_row = QHBoxLayout()
        pair_preview_row.setSpacing(12)
        previews.addLayout(pair_preview_row, 1)
        self.compare_preview_a = QLabel("Ảnh A")
        self.compare_preview_b = QLabel("Ảnh B")
        for label in (self.compare_preview_a, self.compare_preview_b):
            label.setObjectName("imagePreview")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(300, 150)
            label.setMaximumHeight(220)
            pair_preview_row.addWidget(label, 1)
        return page

    def _build_search_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        left = QFrame()
        left.setObjectName("controlPanel")
        left_layout = QVBoxLayout(left)
        layout.addWidget(left, 1)

        title = QLabel("Tìm kiếm nhân vật trong Dataset")
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)
        hint = QLabel(
            "Chọn ảnh query, ví dụ ảnh nhân vật lấy từ Internet, để kiểm tra người đó có giống nhân vật nào trong dataset hay không. "
            "Tab này ưu tiên dùng cache App2 đã precompute sẵn trong models/precomputed, nên không cần rebuild lại toàn bộ dataset trong app desktop."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        left_layout.addWidget(hint)

        model_label = QLabel("Model tìm kiếm")
        model_label.setObjectName("smallLabel")
        left_layout.addWidget(model_label)
        self.dataset_model_combo = QComboBox()
        self.dataset_model_combo.addItems(DATASET_SEARCH_MODELS)
        self.dataset_model_combo.setCurrentText(self.model_combo.currentText())
        left_layout.addWidget(self.dataset_model_combo)

        search_btn = QPushButton("🔎 Chọn ảnh và tìm nhân vật trong dataset")
        search_btn.setObjectName("startButton")
        search_btn.clicked.connect(self.search_dataset_image)
        left_layout.addWidget(search_btn)

        self.search_result = QTextEdit()
        self.search_result.setReadOnly(True)
        self.search_result.setPlaceholderText("Kết quả tìm kiếm và top matches sẽ hiển thị ở đây...")
        left_layout.addWidget(self.search_result, 1)

        self.search_preview = QLabel("Ảnh query tìm kiếm")
        self.search_preview.setObjectName("imagePreview")
        self.search_preview.setAlignment(Qt.AlignCenter)
        self.search_preview.setMinimumSize(620, 520)
        layout.addWidget(self.search_preview, 2)
        return page

    def _build_database_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        left = QVBoxLayout()
        layout.addLayout(left, 1)
        title = QLabel("Database & cache")
        title.setObjectName("panelTitle")
        left.addWidget(title)

        self.people_list = QListWidget()
        self.people_list.setSelectionMode(QListWidget.MultiSelection)
        self.people_list.itemSelectionChanged.connect(self.update_database_person_preview)
        left.addWidget(self.people_list, 1)

        self.db_model_combo = QComboBox()
        self.db_model_combo.addItems(["Tất cả model"] + list(utils.ALL_RECOGNITION_MODELS))
        self.db_model_combo.setMinimumWidth(240)
        left.addWidget(QLabel("Model rebuild"))
        left.addWidget(self.db_model_combo)

        actions = QHBoxLayout()
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.refresh_database_tab)
        rebuild_btn = QPushButton("⚡ Rebuild cache")
        rebuild_btn.clicked.connect(self.rebuild_selected_people)
        delete_btn = QPushButton("🗑️ Xóa người chọn")
        delete_btn.setObjectName("stopButton")
        delete_btn.clicked.connect(self.delete_selected_people)
        undo_btn = QPushButton("↩ Hoàn tác xóa")
        undo_btn.clicked.connect(self.undo_delete_people)
        actions.addWidget(refresh_btn)
        actions.addWidget(rebuild_btn)
        actions.addWidget(delete_btn)
        actions.addWidget(undo_btn)
        left.addLayout(actions)

        right = QVBoxLayout()
        layout.addLayout(right, 1)

        self.db_face_preview = QLabel("Chọn 1 người để xem ảnh khuôn mặt")
        self.db_face_preview.setObjectName("imagePreview")
        self.db_face_preview.setAlignment(Qt.AlignCenter)
        self.db_face_preview.setMinimumSize(420, 300)
        right.addWidget(self.db_face_preview, 2)

        self.db_info = QTextEdit()
        self.db_info.setReadOnly(True)
        right.addWidget(self.db_info, 1)
        return page

    def _build_model_info_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        title = QLabel("Thông tin mô hình")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        self.model_info_tabs = QTabWidget()
        layout.addWidget(self.model_info_tabs, 1)
        for name, text in self._model_info_pages().items():
            edit = QTextEdit()
            edit.setReadOnly(True)
            edit.setText(text)
            self.model_info_tabs.addTab(edit, name)
        return page

    def _build_settings_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        title = QLabel("Cài đặt giao diện & vận hành")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        panel = QFrame()
        panel.setObjectName("controlPanel")
        form = QVBoxLayout(panel)
        form.addWidget(QLabel("Chọn theme"))

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Sáng", "Xanh công nghệ", "Tối"])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        form.addWidget(self.theme_combo)

        settings_note = QTextEdit()
        settings_note.setReadOnly(True)
        settings_note.setText(
            "Gợi ý cài đặt:\n"
            "• Theme Sáng là mặc định để demo dễ đọc trên máy chiếu.\n"
            "• Theme Xanh công nghệ đã được làm nhẹ hơn để tránh nền quá tối.\n"
            "• Nếu realtime lag: tăng Recognize / sec lên 3-5 hoặc giảm Max FPS.\n"
            "• Nếu nhận nhầm: tăng threshold nhẹ; nếu không nhận ra người quen: giảm threshold nhẹ.\n"
            "• Rebuild cache sau khi thêm/xóa nhiều ảnh hoặc đổi model chính."
        )
        form.addWidget(settings_note, 1)

        layout.addWidget(panel, 1)
        return page

    def _build_help_tab(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        self.help_menu = QListWidget()
        self.help_pages = QStackedWidget()
        topics = self._help_topics()
        for name, text in topics.items():
            self.help_menu.addItem(name)
            page_text = QTextEdit()
            page_text.setReadOnly(True)
            page_text.setText(text)
            self.help_pages.addWidget(page_text)
        self.help_menu.currentRowChanged.connect(self.help_pages.setCurrentIndex)
        self.help_menu.setCurrentRow(0)
        layout.addWidget(self.help_menu, 1)
        layout.addWidget(self.help_pages, 3)
        return page

    def _model_info_pages(self):
        summary = """Tổng quan benchmark mô hình

Nguồn số liệu chính: reports/face_model_benchmark_006.md
Bộ benchmark dùng 7.701 embedding, 4.281 người và 10.000 cặp so sánh.

Pipeline khuyến nghị cho desktop demo:
• Facenet512_retinaface_embeddings
• Accuracy: 96.13%
• FAR: 2.16% (tỷ lệ nhận nhầm người lạ)
• FRR: 5.58% (tỷ lệ từ chối nhầm người quen)
• Threshold gợi ý: 0.407
• Tốc độ so sánh: khoảng 0.0024 ms/cặp vector

Ghi chú demo:
• FaceNet512 thường ổn định hơn vì vector 512 chiều chứa nhiều đặc trưng hơn.
• RetinaFace detect chính xác hơn nhưng có thể nặng hơn MTCNN tùy máy.
• Threshold càng cao càng chặt, giảm nhận nhầm nhưng dễ Không xác định.
"""
        facenet512_retina = """Facenet512 + RetinaFace

Vai trò: pipeline ưu tiên độ chính xác.
Embedding dim: 512
Embeddings: 7.701
People: 4.281
Pairs benchmark: 10.000
Threshold benchmark: 0.407157
Accuracy: 0.9613
FAR: 0.0216
FRR: 0.0558
Compare speed: 0.002422 ms/pair

Nên dùng khi:
• Demo cần nhận diện ổn định.
• Máy đủ mạnh và muốn giảm rủi ro nhận nhầm.
• Ảnh/camera có nhiều góc mặt khác nhau.
"""
        facenet_retina = """Facenet + RetinaFace

Vai trò: cân bằng tốc độ và độ chính xác.
Embedding dim: 128
Embeddings: 7.701
People: 4.281
Pairs benchmark: 10.000
Threshold benchmark: 0.401176
Accuracy: 0.9545
FAR: 0.0214
FRR: 0.0696
Compare speed: 0.000637 ms/pair

Nên dùng khi:
• Muốn nhẹ hơn FaceNet512.
• Dataset vừa phải, camera rõ mặt.
• Cần tốc độ so sánh vector nhanh.
"""
        facenet512_mtcnn = """Facenet512 + MTCNN

Vai trò: pipeline mạnh nhưng detector nhẹ/dễ dùng hơn RetinaFace.
Embedding dim: 512
Embeddings: 7.701
People: 4.281
Pairs benchmark: 10.000
Threshold benchmark: 0.417026
Accuracy: 0.9476
FAR: 0.0196
FRR: 0.0852
Compare speed: 0.002656 ms/pair

Nên dùng khi:
• RetinaFace gặp lỗi dependency hoặc quá nặng.
• Vẫn muốn vector 512 chiều.
• Demo local cần cấu hình tương đối ổn định.
"""
        facenet_mtcnn = """Facenet + MTCNN

Vai trò: pipeline nhẹ nhất trong nhóm FaceNet/MTCNN.
Embedding dim: 128
Embeddings: 7.701
People: 4.281
Pairs benchmark: 10.000
Threshold benchmark: 0.366843
Accuracy: 0.9333
FAR: 0.0404
FRR: 0.0930
Compare speed: 0.000578 ms/pair

Nên dùng khi:
• Máy yếu hoặc cần phản hồi nhanh.
• Chấp nhận độ chính xác thấp hơn một chút.
• Dùng để demo sự đánh đổi giữa tốc độ và độ chính xác.
"""
        return {
            "Tổng quan": summary,
            "FaceNet512 + RetinaFace": facenet512_retina,
            "FaceNet + RetinaFace": facenet_retina,
            "FaceNet512 + MTCNN": facenet512_mtcnn,
            "FaceNet + MTCNN": facenet_mtcnn,
        }

    def _help_topics(self):
        return {
            "Demo nhanh": """Demo nhanh trong 3 phút

1) Vào tab Đăng ký
• Nhập tên không dấu.
• Chọn 1-3 ảnh rõ mặt hoặc ảnh chụp trực diện.
• Bấm Đăng ký và tạo embedding.

2) Vào tab Ảnh tĩnh
• Chọn ảnh hoặc Mở camera → Chụp ảnh.
• Bấm Nhận diện & xuất ảnh detect.
• Giải thích Similarity, Threshold và Top matches.

3) Vào tab Realtime
• Giữ model mặc định nếu chưa cần so sánh.
• Bấm START, nhìn vào camera, sau đó STOP.
• Nếu cần reset thông số, bấm Default.

4) Vào tab So sánh
• Chọn Ảnh A và Ảnh B để chứng minh cùng/khác người.
• Nếu ảnh có nhiều người, app tự so tất cả cặp mặt và đánh dấu cặp tốt nhất.

5) Vào tab Tìm kiếm
• Chọn ảnh bất kỳ để kiểm tra có nằm trong dataset không.
""",
            "Đăng ký": """Đăng ký khuôn mặt

Mục đích: thêm người mới vào database và tạo embedding cache để nhận diện nhanh.

Cách dùng:
• Tên nên không dấu, không ký tự đặc biệt.
• Ảnh nên rõ mặt, đủ sáng, không che quá nhiều.
• Với người đã tồn tại, bật tùy chọn chặn ảnh không giống user để tránh làm nhiễu database.
• Sau khi đăng ký nhiều ảnh, có thể sang Database để rebuild cache.

Lỗi thường gặp:
• Tên không hợp lệ: đổi sang chữ/số/khoảng trắng đơn giản.
• Ảnh không tạo embedding: dùng ảnh rõ mặt hơn.
""",
            "Ảnh tĩnh": """Nhận diện ảnh tĩnh

Luồng chuẩn:
• Chọn ảnh query từ máy, hoặc Mở camera để xem preview live.
• Preview camera chỉ hiển thị hình chuyển động, chưa detect để tránh nặng máy.
• Bấm Chụp ảnh nếu khung hình ổn.
• Bấm Nhận diện & xuất ảnh detect để app tạo vector và vẽ kết quả.

Kết quả:
• Similarity càng cao càng giống người trong database.
• Threshold là ngưỡng quyết định có nhận tên hay Không xác định.
• Top matches giúp giải thích vì sao app chọn người gần nhất.
""",
            "Realtime": """Realtime camera

Luồng chuẩn:
• Chọn model, threshold, camera và backend.
• Bấm START để mở camera.
• Bấm STOP trước khi chuyển sang camera ảnh tĩnh.
• Bấm Default để đưa thông số về cấu hình khuyến nghị.

Mẹo chỉnh:
• Lag: tăng Recognize / sec hoặc giảm Max FPS.
• Không nhận ra người quen: giảm threshold nhẹ.
• Nhận nhầm: tăng threshold nhẹ.
• Camera không mở: thử Backend MSMF hoặc Auto.
""",
            "So sánh": """So sánh 2 ảnh

Luồng chuẩn:
• Chọn Ảnh A và Ảnh B.
• Bấm So sánh A và B.
• App detect tất cả khuôn mặt trong từng ảnh rồi so toàn bộ cặp A×B.
• Cặp có similarity cao nhất sẽ được đánh dấu xanh trên ảnh preview.
• Similarity >= threshold: có ít nhất 1 cặp có khả năng cùng một người.

Trường hợp demo nên nói rõ:
• Nếu ảnh A có 2 người và ảnh B có 1 người trùng với một trong hai, app sẽ tìm cặp Aᵢ ↔ Bⱼ tốt nhất.
• Các mặt còn lại vẫn được vẽ khung phụ để chứng minh app đã detect nhiều mặt.
• Nếu detector không bắt được mặt, app fallback về so toàn ảnh để không làm đứt luồng demo.
""",
            "Tìm kiếm": """Tìm kiếm trong Dataset

Mục đích: kiểm tra một ảnh query có thuộc người đã đăng ký trong database/cache hay không.

Cách dùng:
• Bấm Chọn ảnh và tìm trong dataset.
• App dùng cache embedding hiện có để tìm top người gần nhất.
• Nếu similarity vượt threshold, app báo tên khớp.
• Nếu không vượt threshold, app báo chưa đủ chắc có trong dataset.

Lưu ý:
• Nếu cache rỗng, sang tab Database để rebuild cache.
• Model/threshold dùng chung với Control Center Realtime.
""",
            "Database": """Database & cache

Chức năng:
• Xem danh sách người và số ảnh đã đăng ký.
• Rebuild cache cho người được chọn theo một model hoặc tất cả model.
• Xóa người được chọn.
• Hoàn tác lần xóa gần nhất nếu lỡ thao tác sai.

Khi nào rebuild cache?
• Sau khi thêm/xóa nhiều ảnh.
• Khi đổi model chính để demo.
• Khi cache hiển thị ít vector hơn số ảnh đăng ký.
""",
            "Thông tin mô hình": """Thông tin mô hình

Tab này dùng để demo phần kỹ thuật:
• Số chiều vector embedding.
• Accuracy, FAR, FRR, threshold benchmark.
• So sánh tốc độ giữa các pipeline.
• Giải thích vì sao chọn FaceNet512 + RetinaFace làm khuyến nghị.

Cách nói khi demo:
• Hệ thống không so pixel ảnh trực tiếp.
• Mỗi khuôn mặt được mã hóa thành vector embedding.
• Nhận diện là bài toán so sánh similarity giữa vector query và vector database.
""",
            "FAQ": """Câu hỏi thường gặp

Q: Vì sao đã đăng ký nhưng realtime vẫn Không xác định?
A: Thử giảm threshold nhẹ, đăng ký thêm ảnh rõ hơn, hoặc rebuild cache đúng model đang dùng.

Q: Vì sao cùng một người nhưng similarity thấp?
A: Góc mặt, ánh sáng, khẩu trang/kính, ảnh mờ hoặc crop sai có thể làm vector lệch.

Q: Vì sao app nhận nhầm người?
A: Threshold có thể quá thấp hoặc database có ảnh bị nhiễu. Tăng threshold và kiểm tra lại ảnh đăng ký.

Q: FaceNet512 khác FaceNet thường ở đâu?
A: FaceNet512 tạo vector 512 chiều, giàu đặc trưng hơn; FaceNet thường là 128 chiều và nhẹ hơn.

Q: FAR và FRR là gì?
A: FAR là nhận nhầm người lạ thành người quen. FRR là từ chối nhầm người quen thành Không xác định.

Q: Có cần internet khi chạy app không?
A: Không cần cho luồng chính nếu model/cache/dependency đã có sẵn trên máy.
""",
        }

    def apply_theme(self, theme_name: str):
        self.current_theme = theme_name
        self._apply_styles()

    def _apply_styles(self):
        palettes = {
            "Tối": {
                "bg": "#09090b",
                "panel": "#18181b",
                "input": "#0f0f12",
                "text": "#f4f4f5",
                "muted": "#a1a1aa",
                "accent": "#7c3aed",
                "tab": "#27272a",
            },
            "Sáng": {
                "bg": "#f8fafc",
                "panel": "#ffffff",
                "input": "#f1f5f9",
                "text": "#0f172a",
                "muted": "#475569",
                "accent": "#2563eb",
                "tab": "#dbeafe",
            },
            "Xanh công nghệ": {
                "bg": "#eaf7ff",
                "panel": "#ffffff",
                "input": "#f0fbff",
                "text": "#0f2f4a",
                "muted": "#25637f",
                "accent": "#0284c7",
                "tab": "#bae6fd",
            },
        }
        p = palettes.get(self.current_theme, palettes["Sáng"])
        self.setStyleSheet(f"""
            QMainWindow, QTabWidget::pane {{ background: {p['bg']}; }}
            QLabel, QCheckBox {{ color: {p['text']}; font-size: 14px; }}
            #mainTabs::pane {{ border: 0; }}
            QTabBar::tab {{
                color: {p['muted']}; background: {p['tab']}; padding: 12px 18px;
                border-top-left-radius: 12px; border-top-right-radius: 12px;
                margin-right: 4px; font-weight: 800;
            }}
            QTabBar::tab:selected {{ background: {p['accent']}; color: white; }}
            #heroTitle {{ color: {p['text']}; font-size: 34px; font-weight: 800; }}
            #panelTitle {{ color: {p['text']}; font-size: 24px; font-weight: 800; }}
            #videoSurface, #imagePreview {{
                border: 1px solid rgba(148, 163, 184, 0.40);
                border-radius: 24px;
                background: {p['panel']};
                color: {p['muted']}; font-size: 22px; font-weight: 700;
            }}
            #statusLabel {{
                color: {p['text']}; padding: 10px 14px; border-radius: 14px;
                background: {p['panel']};
                border: 1px solid rgba(59, 130, 246, 0.25);
            }}
            #controlPanel, #metricCard {{
                border-radius: 20px; background: {p['panel']};
                border: 1px solid rgba(148, 163, 184, 0.28);
            }}
            #controlPanel {{ padding: 14px; }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QListWidget {{
                color: {p['text']}; background: {p['input']};
                border: 1px solid rgba(96, 165, 250, 0.40);
                border-radius: 10px; padding: 8px; min-height: 28px;
                selection-background-color: {p['accent']};
            }}
            QComboBox {{ min-width: 210px; }}
            #realtimeControlPanel QComboBox {{ min-width: 170px; max-width: 260px; }}
            QComboBox QAbstractItemView {{ min-width: 260px; background: {p['input']}; color: {p['text']}; }}
            #hintLabel {{ color: {p['muted']}; font-size: 13px; }}
            #compactHintLabel {{ color: {p['muted']}; font-size: 12px; font-weight: 700; }}
            QPushButton {{
                border: none; border-radius: 16px; padding: 14px; color: white;
                background: {p['accent']}; font-size: 15px; font-weight: 800;
            }}
            QPushButton:hover {{ background: #38bdf8; }}
            #startButton {{ background: #16a34a; }}
            #startButton:hover {{ background: #22c55e; }}
            #stopButton {{ background: #dc2626; }}
            #stopButton:hover {{ background: #ef4444; }}
            QPushButton:disabled {{ background: #64748b; color: #e2e8f0; }}
            #metricTitle {{ color: {p['muted']}; font-size: 13px; font-weight: 700; }}
            #metricValue {{ color: {p['text']}; font-size: 18px; font-weight: 800; }}
            #realtimeControlPanel {{ padding: 10px; border-radius: 18px; background: {p['panel']}; border: 1px solid rgba(148, 163, 184, 0.28); }}
            #realtimeControlPanel QLabel {{ font-size: 12px; }}
            #realtimeControlPanel QPushButton {{ padding: 9px 8px; border-radius: 12px; font-size: 12px; }}
            #realtimeControlPanel QComboBox, #realtimeControlPanel QSpinBox, #realtimeControlPanel QDoubleSpinBox {{ padding: 5px; min-height: 22px; font-size: 12px; }}
            #realtimeControlPanel #metricCard {{ border-radius: 14px; }}
            #realtimeControlPanel #metricTitle {{ font-size: 11px; }}
            #realtimeControlPanel #metricValue {{ font-size: 13px; }}
            #panelTitleSmall {{ color: {p['text']}; font-size: 18px; font-weight: 800; }}
        """)

    def _load_initial_state(self):
        self._sync_threshold(self.model_combo.currentText())
        self.refresh_database_tab()

    def _model_default_threshold(self, model_name: str, fallback: float = 0.35) -> float:
        return float(self.thresholds.get(model_name, fallback))

    def _compare_default_threshold(self, model_name: str) -> float:
        # Compare is stricter than realtime/static recognition to reduce false positives.
        return max(self._model_default_threshold(model_name), 0.65)

    def _compare_verdict(self, similarity: float, threshold: float, margin: float = 0.05):
        if similarity >= threshold:
            if similarity - threshold < margin:
                return "warning", "⚠️ Có thể cùng người nhưng điểm sát ngưỡng, nên kiểm tra thêm ảnh/model khác"
            return "match", "✅ Có khả năng cùng một người"
        if threshold - similarity <= margin:
            return "suspect", "⚠️ Nghi ngờ, điểm gần ngưỡng nhưng chưa đủ chắc để kết luận cùng người"
        return "no_match", "❌ Khác người hoặc chưa đủ giống"

    def _sync_threshold(self, model_name: str):
        self.threshold_spin.setValue(self._model_default_threshold(model_name))

    def _sync_registration_threshold(self, model_name: str):
        if hasattr(self, "reg_threshold_spin") and not self.reg_manual_threshold_check.isChecked():
            self.reg_threshold_spin.setValue(self._model_default_threshold(model_name))

    def _registration_gate_threshold(self):
        if self.reg_manual_threshold_check.isChecked():
            return float(self.reg_threshold_spin.value())
        return None

    def _refresh_cache_card(self):
        model = self.model_combo.currentText()
        try:
            status = utils.get_embedding_cache_status()
            count = status.get("by_model", {}).get(model, 0)
            self.cache_card.set_value(f"{model}: {count} vectors")
        except Exception as exc:
            self.cache_card.set_value(f"Cache status lỗi: {exc}")

    def _set_controls_enabled(self, enabled: bool):
        for widget in [
            self.model_combo,
            self.threshold_spin,
            self.camera_spin,
            self.backend_combo,
            self.interval_spin,
            self.fps_spin,
            self.default_btn,
        ]:
            widget.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    def reset_realtime_defaults(self):
        default_model = "Facenet512_retinaface" if "Facenet512_retinaface" in utils.ALL_RECOGNITION_MODELS else utils.ALL_RECOGNITION_MODELS[0]
        self.model_combo.setCurrentText(default_model)
        self.threshold_spin.setValue(float(self.thresholds.get(default_model, 0.35)))
        self.camera_spin.setValue(0)
        self.backend_combo.setCurrentText("DirectShow")
        self.interval_spin.setValue(2.0)
        self.fps_spin.setValue(24)
        self.name_card.set_value("—")
        self.sim_card.set_value("—")
        self.fps_card.set_value("—")
        self.status_label.setText("Đã đưa Realtime về cấu hình mặc định.")
        self._refresh_cache_card()

    def _set_preview(self, label: QLabel, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            label.setText("Không đọc được ảnh")
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _selected_people(self):
        return [item.data(Qt.UserRole) for item in self.people_list.selectedItems()]

    def pick_register_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh đăng ký", PROJECT_ROOT, IMAGE_FILTER)
        if path:
            self.close_register_camera()
            self.reg_image_path = path
            self.reg_camera_frame = None
            self._set_preview(self.reg_preview, path)
            self.reg_status.setText(f"Đã chọn ảnh đăng ký từ máy:\n{path}")

    def open_register_camera(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi mở camera đăng ký.")
            return
        self.close_static_camera()
        self.close_compare_camera()
        self.close_register_camera()
        cap, message = utils.open_camera_capture(int(self.camera_spin.value()), self.backend_combo.currentText())
        if cap is None:
            self.reg_status.setText(message)
            return
        self.reg_camera = cap
        self.reg_camera_frame = None
        self.reg_image_path = None
        self.reg_timer.start(max(10, int(1000 / max(int(self.fps_spin.value()), 1))))
        self.reg_status.setText(f"🎥 Camera đăng ký đang preview live. Bấm 'Chụp ảnh đăng ký' khi mặt rõ.\n{message}")

    def update_register_camera_preview(self):
        if self.reg_camera is None:
            return
        ok, frame = self.reg_camera.read()
        if not ok or frame is None:
            self.reg_status.setText("Không đọc được frame từ camera đăng ký.")
            self.close_register_camera()
            return
        self.reg_camera_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.reg_preview.setPixmap(
            pixmap.scaled(self.reg_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def close_register_camera(self):
        if self.reg_timer.isActive():
            self.reg_timer.stop()
        if self.reg_camera is not None:
            self.reg_camera.release()
            self.reg_camera = None

    def capture_register_image(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi chụp ảnh đăng ký.")
            return
        if self.reg_camera_frame is None:
            QMessageBox.information(self, "Chưa có preview", "Hãy bấm Mở camera và đợi preview chuyển động trước.")
            return
        try:
            frame = self.reg_camera_frame.copy()
            self.close_register_camera()
            path = self._save_temp_frame(frame, "desktop_register_capture")
            self.reg_image_path = path
            self.reg_camera_frame = None
            self._set_preview(self.reg_preview, path)
            self.reg_status.setText(f"📸 Đã chụp ảnh đăng ký. Có thể bấm 'Đăng ký và tạo embedding'.\n{path}")
        except Exception as exc:
            self.reg_status.setText(f"❌ Lỗi chụp ảnh đăng ký: {exc}")

    def clear_register_image(self):
        self.close_register_camera()
        self.reg_image_path = None
        self.reg_camera_frame = None
        self.reg_preview.clear()
        self.reg_preview.setText("Chưa chọn ảnh")
        self.reg_status.setText("Đã xóa ảnh đăng ký hiện tại.")

    def register_selected_face(self):
        name = self.reg_name_input.text().strip()
        if not self.reg_image_path:
            QMessageBox.warning(self, "Thiếu ảnh", "Hãy chọn hoặc chụp ảnh đăng ký trước.")
            return
        valid, msg = utils.is_valid_name(name)
        if not valid:
            QMessageBox.warning(self, "Tên không hợp lệ", msg)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = utils.register_face_image_checked(
                name,
                LocalImageUpload(self.reg_image_path),
                self.reg_model_combo.currentText(),
                self._registration_gate_threshold(),
                enforce_consistency=self.reg_gate_check.isChecked(),
            )
        except Exception as exc:
            result = {"saved": False, "gate": {"message": f"Lỗi đăng ký: {exc}"}}
        finally:
            QApplication.restoreOverrideCursor()
        gate = result.get("gate", {})
        if result.get("saved"):
            text = f"✅ Đã đăng ký: {result['path']}\n{gate.get('message', '')}"
            if gate.get("best_similarity") is not None:
                threshold_mode = "thủ công" if self.reg_manual_threshold_check.isChecked() else "mặc định theo model"
                text += f"\nSimilarity gần nhất: {gate['best_similarity']:.4f} / {gate['threshold']:.3f} ({threshold_mode})"
            self.reg_status.setText(text)
            self.refresh_database_tab()
            self._refresh_cache_card()
        else:
            self.reg_status.setText(f"❌ {gate.get('message', 'Không thể đăng ký ảnh này.')}")

    def pick_query_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh cần nhận diện", PROJECT_ROOT, IMAGE_FILTER)
        if path:
            self.close_static_camera()
            self.query_image_path = path
            self._set_preview(self.query_preview, path)
            self.detect_preview.clear()
            self.detect_preview.setText("Ảnh detect sẽ hiện ở đây")
            self.static_result.setText(f"Đã chọn: {path}")

    def open_static_camera(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi mở camera ảnh tĩnh.")
            return
        self.close_static_camera()
        cap, message = utils.open_camera_capture(int(self.camera_spin.value()), self.backend_combo.currentText())
        if cap is None:
            self.static_result.setText(message)
            return
        self.static_camera = cap
        self.static_camera_frame = None
        self.query_image_path = None
        self.static_timer.start(max(10, int(1000 / max(int(self.fps_spin.value()), 1))))
        self.detect_preview.clear()
        self.detect_preview.setText("Ảnh detect sẽ hiện ở đây")
        self.static_result.setText(f"🎥 Camera ảnh tĩnh đang preview live. Preview này chưa detect.\n{message}")

    def update_static_camera_preview(self):
        if self.static_camera is None:
            return
        ok, frame = self.static_camera.read()
        if not ok or frame is None:
            self.static_result.setText("Không đọc được frame từ camera ảnh tĩnh.")
            self.close_static_camera()
            return
        self.static_camera_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.query_preview.setPixmap(
            pixmap.scaled(self.query_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def close_static_camera(self):
        if self.static_timer.isActive():
            self.static_timer.stop()
        if self.static_camera is not None:
            self.static_camera.release()
            self.static_camera = None

    def _save_temp_frame(self, frame, prefix: str):
        capture_dir = os.path.join(APP_DIR, "temp_uploads")
        os.makedirs(capture_dir, exist_ok=True)
        path = os.path.join(capture_dir, f"{prefix}_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(path, frame)
        return path

    def capture_query_image(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi chụp ảnh tĩnh.")
            return
        if self.static_camera_frame is None:
            QMessageBox.information(self, "Chưa có preview", "Hãy bấm Mở camera và đợi preview chuyển động trước.")
            return
        try:
            frame = self.static_camera_frame.copy()
            self.close_static_camera()
            path = self._save_temp_frame(frame, "desktop_static_capture")
            self.query_image_path = path
            self._set_preview(self.query_preview, path)
            self.detect_preview.clear()
            self.detect_preview.setText("Ảnh detect sẽ hiện ở đây")
            self.static_result.setText(f"📸 Đã chụp ảnh.\n{path}")
        except Exception as exc:
            self.static_result.setText(f"❌ Lỗi chụp ảnh: {exc}")


    def clear_query_image(self):
        self.close_static_camera()
        self.query_image_path = None
        self.static_camera_frame = None
        self.query_preview.clear()
        self.query_preview.setText("Ảnh gốc / camera preview")
        self.detect_preview.clear()
        self.detect_preview.setText("Ảnh detect sẽ hiện ở đây")
        self.static_result.setText("Đã xóa ảnh query hiện tại.")

    def _render_detected_query_image(self, result: dict):
        if not self.query_image_path:
            return None
        frame = cv2.imread(self.query_image_path)
        if frame is None:
            return None

        rois = utils.detect_face_rois(frame)
        overlay_box = rois[0][1] if rois else None
        detected = utils.draw_recognition_overlay(frame.copy(), result, overlay_box)
        if not rois:
            detected = frame.copy()
            cv2.putText(
                detected,
                "No face detected",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (64, 88, 255),
                2,
                cv2.LINE_AA,
            )

        output_dir = os.path.join(APP_DIR, "temp_uploads")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"desktop_static_detect_{int(time.time())}.jpg")
        cv2.imwrite(output_path, detected)
        self._set_preview(self.detect_preview, output_path)
        return output_path

    def recognize_query_image(self):
        if not self.query_image_path:
            QMessageBox.warning(self, "Thiếu ảnh", "Hãy chọn ảnh query trước.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        detect_path = None
        total_started = time.time()
        try:
            result = utils.recognize_image_path_cached(
                self.query_image_path,
                self.static_model_combo.currentText(),
                self._model_default_threshold(self.static_model_combo.currentText()),
            )
            if result and not result.get("error"):
                detect_path = self._render_detected_query_image(result)
        except Exception as exc:
            result = {"error": str(exc)}
        finally:
            total_elapsed = time.time() - total_started
            QApplication.restoreOverrideCursor()
        if result is None:
            self.static_result.setText(f"Cache rỗng hoặc chưa có embedding hợp lệ. Hãy rebuild cache.\nTổng thời gian chờ: {total_elapsed:.2f}s")
            return
        if result.get("error"):
            self.static_result.setText(f"❌ Lỗi nhận diện: {result['error']}\nTổng thời gian chờ: {total_elapsed:.2f}s")
            return
        lines = [
            f"{'✅ Nhận diện' if result.get('is_match') else '❓ Không xác định'}: {result.get('name')}",
            f"Model: {self.static_model_combo.currentText()}",
            f"Similarity: {result.get('similarity', 0):.4f} / {result.get('threshold', 0):.3f}",
            f"Gần nhất: {result.get('raw_name', '-')}",
            f"Vector dim: {len(result.get('query_embedding', []))}",
            f"Tổng thời gian chờ: {total_elapsed:.2f}s",
            f"Thời gian model/cache: {result.get('elapsed', 0):.2f}s",
            f"Ảnh detect: {detect_path or 'không tạo được'}",
            "",
            "Top matches:",
        ]
        for item in result.get("top_matches", []):
            lines.append(f"#{item['rank']} {item['person_name']} • {item['similarity']:.4f} • {os.path.basename(item['photo_path'])}")
        if self.show_vector_check.isChecked():
            emb = result.get("query_embedding", [])
            preview = ", ".join(f"{float(v):.5f}" for v in emb[:32])
            suffix = " ..." if len(emb) > 32 else ""
            lines.extend(["", "Vector preview (32 số đầu):", preview + suffix])
        self.static_result.setText("\n".join(lines))

    def pick_compare_image(self, slot: str):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh so sánh", PROJECT_ROOT, IMAGE_FILTER)
        if not path:
            return
        self.close_compare_camera()
        if slot == "a":
            self.compare_image_a = path
            self._set_preview(self.compare_preview_a, path)
        else:
            self.compare_image_b = path
            self._set_preview(self.compare_preview_b, path)
        self.compare_result.setText(
            f"Ảnh A: {self.compare_image_a or 'chưa chọn'}\n"
            f"Ảnh B: {self.compare_image_b or 'chưa chọn'}\n\n"
            "Bấm 'So sánh A và B' để detect tất cả khuôn mặt và tìm cặp khớp tốt nhất."
        )

    def open_compare_camera(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi mở camera so sánh.")
            return
        self.close_static_camera()
        self.close_compare_camera()
        cap, message = utils.open_camera_capture(int(self.camera_spin.value()), self.backend_combo.currentText())
        if cap is None:
            self.compare_result.setText(message)
            return
        self.compare_camera = cap
        self.compare_camera_frame = None
        self.compare_timer.start(max(10, int(1000 / max(int(self.fps_spin.value()), 1))))
        self.compare_result.setText(f"🎥 Camera so sánh đang preview live. Preview này chưa detect.\n{message}")

    def update_compare_camera_preview(self):
        if self.compare_camera is None:
            return
        ok, frame = self.compare_camera.read()
        if not ok or frame is None:
            self.compare_result.setText("Không đọc được frame từ camera so sánh.")
            self.close_compare_camera()
            return
        self.compare_camera_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.compare_preview_camera.setPixmap(
            pixmap.scaled(self.compare_preview_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def close_compare_camera(self):
        if self.compare_timer.isActive():
            self.compare_timer.stop()
        if self.compare_camera is not None:
            self.compare_camera.release()
            self.compare_camera = None
        self.compare_camera_frame = None
        if hasattr(self, "compare_preview_camera"):
            self.compare_preview_camera.clear()
            self.compare_preview_camera.setText("Camera so sánh preview")

    def capture_compare_image(self, slot: str):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Camera đang bận", "Hãy dừng realtime trước khi chụp ảnh so sánh.")
            return
        if self.compare_camera_frame is None:
            QMessageBox.information(self, "Chưa có preview", "Hãy bấm Mở camera và đợi preview chuyển động trước.")
            return
        try:
            path = self._save_temp_frame(self.compare_camera_frame.copy(), f"desktop_compare_{slot}")
            if slot == "a":
                self.compare_image_a = path
                self._set_preview(self.compare_preview_a, path)
            else:
                self.compare_image_b = path
                self._set_preview(self.compare_preview_b, path)
            self.compare_result.setText(
                f"📸 Đã chụp ảnh {slot.upper()}.\n"
                f"Ảnh A: {self.compare_image_a or 'chưa chọn'}\n"
                f"Ảnh B: {self.compare_image_b or 'chưa chọn'}\n\n"
                "Có thể chụp tiếp ảnh còn lại hoặc bấm 'So sánh A và B'."
            )
        except Exception as exc:
            self.compare_result.setText(f"❌ Lỗi chụp ảnh so sánh: {exc}")

    def show_compare_details(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Chi tiết ảnh so sánh A/B")
        dialog.resize(1180, 760)
        root = QVBoxLayout(dialog)
        summary = QLabel(
            "Chi tiết so sánh: ảnh A và B được phóng lớn để demo. "
            "Thông số hiện tại hiển thị ngay phía trên ảnh."
        )
        summary.setWordWrap(True)
        summary.setObjectName("hintLabel")
        root.addWidget(summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        row = QHBoxLayout(content)
        row.setSpacing(14)

        def add_image_card(title: str, path: str | None):
            card = QFrame()
            card.setObjectName("controlPanel")
            card_layout = QVBoxLayout(card)
            info = QLabel(self._compare_image_info(title, path))
            info.setWordWrap(True)
            info.setObjectName("hintLabel")
            card_layout.addWidget(info)
            preview = QLabel("Chưa có ảnh")
            preview.setObjectName("imagePreview")
            preview.setAlignment(Qt.AlignCenter)
            preview.setMinimumSize(500, 520)
            if path and os.path.exists(path):
                pixmap = QPixmap(path)
                preview.setPixmap(pixmap.scaled(560, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            card_layout.addWidget(preview, 1)
            row.addWidget(card, 1)

        add_image_card("Ảnh A", self.compare_image_a)
        add_image_card("Ảnh B", self.compare_image_b)

        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(dialog.accept)
        root.addWidget(close_btn)
        dialog.exec()

    def _compare_image_info(self, title: str, path: str | None) -> str:
        model = self.compare_model_combo.currentText() if hasattr(self, "compare_model_combo") else "-"
        threshold = self.compare_threshold_spin.value() if hasattr(self, "compare_threshold_spin") else 0
        if not path:
            return f"{title}\nTrạng thái: chưa chọn/chưa chụp\nModel: {model}\nThreshold: {threshold:.2f}"
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        image = cv2.imread(path) if os.path.exists(path) else None
        dims = f"{image.shape[1]}×{image.shape[0]} px" if image is not None else "không đọc được"
        return (
            f"{title}\n"
            f"File: {os.path.basename(path)}\n"
            f"Kích thước ảnh: {dims}\n"
            f"Dung lượng: {size:.1f} KB\n"
            f"Model: {model}\n"
            f"Threshold: {threshold:.2f}"
        )

    def _detector_from_model(self, model: str) -> str:
        return model.split("_", 1)[1] if "_" in model else "mtcnn"

    def _base_model_from_pipeline(self, model: str) -> str:
        return model.split("_", 1)[0] if "_" in model else model

    def _detect_faces_for_compare(self, image_path: str, model: str):
        from deepface import DeepFace

        detector = self._detector_from_model(model)
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector,
            align=True,
            enforce_detection=True,
        )
        detected = []
        temp_dir = os.path.join(APP_DIR, "temp_uploads", "compare_faces")
        os.makedirs(temp_dir, exist_ok=True)
        stamp = int(time.time() * 1000)
        for idx, face in enumerate(faces):
            arr = face.get("face")
            if arr is None:
                continue
            face_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            crop_path = os.path.join(temp_dir, f"face_{stamp}_{idx}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))
            area = face.get("facial_area") or {}
            detected.append({
                "index": idx + 1,
                "crop_path": crop_path,
                "box": (
                    int(area.get("x", 0)),
                    int(area.get("y", 0)),
                    int(area.get("w", 0)),
                    int(area.get("h", 0)),
                ),
            })
        return detected

    def _render_compare_overlay(self, image_path: str, faces, best_face, label: str):
        image = cv2.imread(image_path)
        if image is None:
            return image_path
        for face in faces:
            x, y, w, h = face["box"]
            if w <= 0 or h <= 0:
                continue
            color = (40, 190, 90) if face is best_face else (80, 140, 255)
            thickness = 4 if face is best_face else 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                image,
                f"{label}{face['index']}",
                (x, max(26, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )
        output_dir = os.path.join(APP_DIR, "temp_uploads")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"compare_{label}_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(output_path, image)
        return output_path

    def _compare_all_detected_faces(self, model: str):
        from src.embedding_extractor import get_embedding
        from src.similarity import cosine_similarity

        start = time.time()
        faces_a = self._detect_faces_for_compare(self.compare_image_a, model)
        faces_b = self._detect_faces_for_compare(self.compare_image_b, model)
        if not faces_a or not faces_b:
            raise ValueError("Không detect đủ khuôn mặt trong một hoặc hai ảnh.")

        for face in faces_a + faces_b:
            face["embedding"] = get_embedding(face["crop_path"], model)

        best = None
        comparisons = []
        for face_a in faces_a:
            for face_b in faces_b:
                similarity = float(cosine_similarity(face_a["embedding"], face_b["embedding"]))
                item = {"a": face_a, "b": face_b, "similarity": similarity}
                comparisons.append(item)
                if best is None or similarity > best["similarity"]:
                    best = item

        threshold = float(self.compare_threshold_spin.value())
        overlay_a = self._render_compare_overlay(self.compare_image_a, faces_a, best["a"], "A")
        overlay_b = self._render_compare_overlay(self.compare_image_b, faces_b, best["b"], "B")
        return {
            "faces_a": faces_a,
            "faces_b": faces_b,
            "best": best,
            "comparisons": comparisons,
            "threshold": threshold,
            "is_match": best["similarity"] >= threshold,
            "verdict": self._compare_verdict(best["similarity"], threshold),
            "elapsed": time.time() - start,
            "overlay_a": overlay_a,
            "overlay_b": overlay_b,
            "embedding_dim": len(best["a"]["embedding"]),
        }

    def compare_two_images(self):
        if not self.compare_image_a or not self.compare_image_b:
            QMessageBox.warning(self, "Thiếu ảnh", "Hãy chọn đủ Ảnh A và Ảnh B trước.")
            return
        model = self.compare_model_combo.currentText() if hasattr(self, "compare_model_combo") else self.model_combo.currentText()
        threshold = float(self.compare_threshold_spin.value()) if hasattr(self, "compare_threshold_spin") else float(self.threshold_spin.value())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        fallback_note = ""
        try:
            result = self._compare_all_detected_faces(model)
            similarity = float(result["best"]["similarity"])
            verdict_key, verdict_text = result.get("verdict", self._compare_verdict(similarity, threshold))
            is_match = verdict_key == "match"
            self._set_preview(self.compare_preview_a, result["overlay_a"])
            self._set_preview(self.compare_preview_b, result["overlay_b"])
        except Exception as exc:
            fallback_note = f"\nGhi chú: detect nhiều mặt không chạy được ({exc}); app đã fallback về so toàn ảnh."
            try:
                result = verify_faces(self.compare_image_a, self.compare_image_b, model)
                similarity = float(result.get("similarity", 0))
                verdict_key, verdict_text = self._compare_verdict(similarity, threshold)
                is_match = verdict_key == "match"
                result["threshold"] = threshold
                result["is_match"] = is_match
                result["verdict"] = (verdict_key, verdict_text)
            except Exception as inner_exc:
                QApplication.restoreOverrideCursor()
                self.compare_result.setText(f"❌ Lỗi so sánh: {inner_exc}")
                return
        QApplication.restoreOverrideCursor()

        if "best" in result:
            best = result["best"]
            ranked = sorted(result["comparisons"], key=lambda item: item["similarity"], reverse=True)[:6]
            lines = [
                "Kết quả so sánh nhiều khuôn mặt",
                "",
                f"Ảnh A: detect {len(result['faces_a'])} khuôn mặt",
                f"Ảnh B: detect {len(result['faces_b'])} khuôn mặt",
                f"Cặp tốt nhất: A{best['a']['index']} ↔ B{best['b']['index']}",
                f"Model: {model}",
                f"Similarity tốt nhất: {similarity:.4f}",
                f"Threshold đang chọn: {threshold:.3f}",
                f"Vùng nghi ngờ: ±0.050 quanh threshold",
                f"Vector dim: {result.get('embedding_dim', '-')}",
                f"Thời gian: {result.get('elapsed', 0):.2f}s",
                f"Kết luận: {verdict_text}",
                "",
                "Top cặp so sánh:",
            ]
            for item in ranked:
                pair_verdict_key, _ = self._compare_verdict(item["similarity"], threshold)
                verdict = "MATCH" if pair_verdict_key == "match" else ("NGHI NGỜ" if pair_verdict_key in ("warning", "suspect") else "no")
                lines.append(f"- A{item['a']['index']} ↔ B{item['b']['index']}: {item['similarity']:.4f} ({verdict})")
            self.compare_result.setText("\n".join(lines))
            return

        self.compare_result.setText(
            "Kết quả so sánh 2 ảnh\n\n"
            f"Ảnh A: {self.compare_image_a}\n"
            f"Ảnh B: {self.compare_image_b}\n"
            f"Model: {model}\n"
            f"Similarity: {similarity:.4f}\n"
            f"Threshold đang chọn: {threshold:.3f}\n"
            "Vùng nghi ngờ: ±0.050 quanh threshold\n"
            f"Vector dim: {result.get('embedding_dim', '-')}\n"
            f"Thời gian: {result.get('inference_time', 0):.2f}s\n\n"
            f"Kết luận: {verdict_text}"
            f"{fallback_note}"
        )

    def _dataset_cache_dir(self) -> str:
        path = os.path.join(APP_DIR, "temp_uploads", "dataset_cache")
        os.makedirs(path, exist_ok=True)
        return path

    def _dataset_cache_path(self, model: str) -> str:
        safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model)
        return os.path.join(self._dataset_cache_dir(), f"{safe_model}_dataset_embeddings.npz")

    def _iter_dataset_images(self):
        if not os.path.isdir(DATASET_ROOT):
            return
        for root, _, files in os.walk(DATASET_ROOT):
            person = os.path.basename(root)
            if root == DATASET_ROOT or not person:
                continue
            for filename in files:
                if os.path.splitext(filename)[1].lower() in DATASET_IMAGE_EXTENSIONS:
                    yield person, os.path.join(root, filename)

    def _precomputed_dataset_cache_path(self, model: str) -> str:
        return os.path.join(PROJECT_ROOT, "models", "precomputed", f"{model}_embeddings.pkl")

    def _load_precomputed_dataset_cache(self, model: str):
        cache_path = self._precomputed_dataset_cache_path(model)
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or not data:
            return {"error": f"File precomputed không hợp lệ hoặc rỗng: {cache_path}"}

        records = []
        embeddings = []
        for photo_path, emb in data.items():
            if emb is None:
                continue
            emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb_arr.size == 0 or not np.any(emb_arr):
                continue
            norm = np.linalg.norm(emb_arr)
            if norm <= 0:
                continue
            emb_arr = (emb_arr / norm).astype(np.float32)
            person = os.path.basename(os.path.dirname(str(photo_path))) or "unknown"
            records.append({"person_name": person, "photo_path": str(photo_path), "embedding": emb_arr})
            embeddings.append(emb_arr)
        if not records:
            return {"error": f"File precomputed không có embedding hợp lệ: {cache_path}"}
        return {
            "records": records,
            "matrix": np.vstack(embeddings).astype(np.float32),
            "built_at": os.path.getmtime(cache_path),
            "cache_path": cache_path,
            "cache_source": "App2 precomputed .pkl",
        }

    def _load_dataset_cache(self, model: str):
        precomputed = self._load_precomputed_dataset_cache(model)
        if precomputed is not None and not precomputed.get("error"):
            return precomputed

        cache_path = self._dataset_cache_path(model)
        if not os.path.exists(cache_path):
            if precomputed is not None and precomputed.get("error"):
                return precomputed
            return None
        data = np.load(cache_path, allow_pickle=True)
        records = []
        people = data["people"].tolist()
        paths = data["paths"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
        for person, photo_path, emb in zip(people, paths, embeddings):
            records.append({"person_name": str(person), "photo_path": str(photo_path), "embedding": emb})
        return {
            "records": records,
            "matrix": embeddings,
            "built_at": float(data["built_at"]) if "built_at" in data else 0.0,
            "cache_path": cache_path,
            "cache_source": "Desktop dataset .npz",
        }

    def _build_dataset_cache(self, model: str):
        from app1_utils import _get_embedding, _normalize_embedding

        people = []
        paths = []
        embeddings = []
        errors = []
        started = time.time()
        for person, image_path in self._iter_dataset_images() or []:
            try:
                emb = _normalize_embedding(_get_embedding(image_path, model)).astype(np.float32)
                people.append(person)
                paths.append(image_path)
                embeddings.append(emb)
            except Exception as exc:
                errors.append(f"{os.path.basename(image_path)}: {exc}")
        if not embeddings:
            return {"error": "Không tạo được embedding nào từ dataset.", "errors": errors}
        matrix = np.vstack(embeddings).astype(np.float32)
        cache_path = self._dataset_cache_path(model)
        np.savez_compressed(
            cache_path,
            people=np.array(people, dtype=object),
            paths=np.array(paths, dtype=object),
            embeddings=matrix,
            built_at=np.array(time.time(), dtype=np.float64),
        )
        return {
            "records": [
                {"person_name": person, "photo_path": photo_path, "embedding": emb}
                for person, photo_path, emb in zip(people, paths, matrix)
            ],
            "matrix": matrix,
            "cache_path": cache_path,
            "built_at": time.time(),
            "elapsed": round(time.time() - started, 2),
            "errors": errors,
        }

    def _dataset_model_code(self, label: str) -> str:
        return DATASET_SEARCH_MODEL_CODES.get(label, label)

    def rebuild_dataset_search_cache(self):
        model_label = self.dataset_model_combo.currentText()
        model = self._dataset_model_code(model_label)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self._build_dataset_cache(model)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.search_result.setText(f"❌ Lỗi rebuild cache dataset: {exc}")
            return
        QApplication.restoreOverrideCursor()
        if result.get("error"):
            self.search_result.setText(f"❌ {result['error']}")
            return
        self.search_result.setText(
            "✅ Đã rebuild cache dataset cho model đang chọn\n\n"
            f"Dataset: {DATASET_ROOT}\n"
            f"Model hiển thị: {model_label}\n"
            f"Model nội bộ: {model}\n"
            f"Số embedding: {len(result.get('records', []))}\n"
            f"Cache: {result.get('cache_path')}\n"
            f"Thời gian build: {result.get('elapsed', 0):.2f}s\n"
            f"Ảnh lỗi/bỏ qua: {len(result.get('errors', []))}\n\n"
            "Cache này đã lưu thành file `.npz`, tắt app mở lại vẫn dùng được."
        )

    def rebuild_all_dataset_search_caches(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        results = []
        try:
            for model_label, model in DATASET_SEARCH_MODEL_OPTIONS:
                result = self._build_dataset_cache(model)
                if result.get("error"):
                    results.append(f"❌ {model_label} ({model}): {result['error']}")
                else:
                    results.append(
                        f"✅ {model_label} ({model}): {len(result.get('records', []))} embedding, "
                        f"{result.get('elapsed', 0):.2f}s, lỗi/bỏ qua {len(result.get('errors', []))}"
                    )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.search_result.setText(f"❌ Lỗi rebuild cache tất cả model: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self.search_result.setText(
            "Kết quả rebuild cache dataset cho tất cả model\n\n"
            f"Dataset: {DATASET_ROOT}\n"
            f"Thư mục cache: {self._dataset_cache_dir()}\n\n"
            + "\n".join(results)
            + "\n\nCache đã lưu thành file `.npz`; lần sau mở app không cần rebuild nếu dataset không đổi."
        )

    def _search_dataset_cache(self, image_path: str, model: str, threshold: float, top_k: int = 8):
        from app1_utils import _get_embedding, _normalize_embedding

        started = time.time()
        cache = self._load_dataset_cache(model)
        if cache is None:
            return {
                "error": (
                    f"Chưa có cache dataset cho model {model}. "
                    "Hãy bấm `Rebuild cache model đang chọn` hoặc `Rebuild cache tất cả model` trước khi tìm."
                )
            }
        if cache.get("error"):
            return cache
        records = cache.get("records", [])
        matrix = cache.get("matrix")
        if not records or matrix is None or len(matrix) == 0:
            return {"error": "Cache dataset rỗng."}
        query_emb = _normalize_embedding(_get_embedding(image_path, model)).astype(np.float32)
        scores = matrix @ query_emb
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_record = records[best_idx]
        is_match = best_score >= threshold
        ordered = np.argsort(scores)[::-1][:max(1, int(top_k))]
        top_matches = []
        for rank, idx in enumerate(ordered, start=1):
            record = records[int(idx)]
            top_matches.append({
                "rank": rank,
                "person_name": record["person_name"],
                "photo_path": record["photo_path"],
                "similarity": float(scores[int(idx)]),
            })
        return {
            "name": best_record["person_name"] if is_match else "Không xác định",
            "raw_name": best_record["person_name"],
            "similarity": best_score,
            "threshold": threshold,
            "is_match": is_match,
            "photo_path": best_record["photo_path"],
            "elapsed": round(time.time() - started, 3),
            "embedding_dim": int(query_emb.size),
            "top_matches": top_matches,
            "cache_path": cache.get("cache_path"),
            "cache_size": len(records),
            "built_now": False,
            "cache_source": cache.get("cache_source", "Unknown"),
        }

    def search_dataset_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh tìm trong dataset", DATASET_ROOT, IMAGE_FILTER)
        if not path:
            return
        self._set_preview(self.search_preview, path)
        model_label = self.dataset_model_combo.currentText()
        model = self._dataset_model_code(model_label)
        threshold = self._model_default_threshold(model)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self._search_dataset_cache(path, model, threshold, top_k=8)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.search_result.setText(f"❌ Lỗi tìm kiếm dataset: {exc}")
            return
        QApplication.restoreOverrideCursor()
        if result is None or result.get("error"):
            self.search_result.setText(f"❌ {result.get('error', 'Không tìm được trong dataset cache.') if result else 'Không tìm được trong dataset cache.'}")
            return
        lines = [
            "Kết quả tìm ảnh trong dataset train/test",
            "",
            f"Dataset: {DATASET_ROOT}",
            f"Ảnh query: {path}",
            f"Model hiển thị: {model_label}",
            f"Model nội bộ: {model}",
            f"Cache: {result.get('cache_path')}",
            f"Nguồn cache: {result.get('cache_source', 'Unknown')}",
            f"Số embedding cache: {result.get('cache_size', 0)}",
            f"Kết luận: {'✅ Có trong dataset' if result.get('is_match') else '❓ Chưa đủ chắc có trong dataset'}",
            f"Tên khớp: {result.get('name')}",
            f"Gần nhất: {result.get('raw_name')}",
            f"Similarity: {result.get('similarity', 0):.4f} / {result.get('threshold', 0):.3f}",
            f"Vector dim: {result.get('embedding_dim', '-')}",
            f"Thời gian tìm: {result.get('elapsed', 0):.3f}s",
            "",
            "Top matches trong dataset:",
        ]
        for item in result.get("top_matches", []):
            lines.append(f"#{item['rank']} {item['person_name']} • {item['similarity']:.4f} • {item['photo_path']}")
        self.search_result.setText("\n".join(lines))

    def refresh_database_tab(self, extra_message: str | None = None):
        people = utils.list_registered_people()
        status = utils.get_embedding_cache_status()
        self.people_list.clear()
        for person in people:
            item = QListWidgetItem(f"{person['name']}  •  {person['photo_count']} ảnh")
            item.setData(Qt.UserRole, person["name"])
            self.people_list.addItem(item)
        lines = [
            f"User: {len(people)}",
            f"Ảnh đăng ký: {status.get('registered_photos', 0)}",
            f"Tổng vector hợp lệ: {status.get('total_records', 0)}",
            "",
            "Cache theo model:",
        ]
        for model, count in status.get("by_model", {}).items():
            lines.append(f"- {model}: {count}")
        if extra_message:
            lines.extend(["", extra_message])
        self.db_info.setText("\n".join(lines))
        self.update_database_person_preview()

    def update_database_person_preview(self):
        if not hasattr(self, "db_face_preview"):
            return
        selected = self.people_list.selectedItems() if hasattr(self, "people_list") else []
        if not selected:
            self.db_face_preview.setText("Chọn 1 người để xem ảnh khuôn mặt")
            self.db_face_preview.setPixmap(QPixmap())
            return

        name = selected[0].data(Qt.UserRole)
        db_dir = utils.get_faces_db_dir()
        person_dir = os.path.join(db_dir, name)
        if not os.path.isdir(person_dir):
            self.db_face_preview.setText(f"Không tìm thấy thư mục ảnh của {name}")
            self.db_face_preview.setPixmap(QPixmap())
            return

        photos = sorted(
            os.path.join(person_dir, filename)
            for filename in os.listdir(person_dir)
            if os.path.splitext(filename)[1].lower() in DATASET_IMAGE_EXTENSIONS
        )
        if not photos:
            self.db_face_preview.setText(f"{name} chưa có ảnh đăng ký")
            self.db_face_preview.setPixmap(QPixmap())
            return

        self._set_preview(self.db_face_preview, photos[0])
        extra = "" if len(selected) == 1 else f"\nĐang chọn thêm {len(selected) - 1} người khác."
        self.db_info.append(f"\nẢnh preview database: {name}\nFile: {photos[0]}{extra}")

    def rebuild_selected_people(self):
        selected = self._selected_people()
        if not selected:
            QMessageBox.information(self, "Chưa chọn", "Hãy chọn ít nhất một người.")
            return
        selected_model = self.db_model_combo.currentText()
        models = None if selected_model == "Tất cả model" else [selected_model]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = utils.rebuild_embeddings_for_people(selected, models, force=True)
        finally:
            QApplication.restoreOverrideCursor()
        self.db_info.setText(
            f"Đã rebuild cache cho: {', '.join(selected)}\n"
            f"Model: {selected_model}\n"
            f"Created: {result.get('created', 0)}\nFailed: {result.get('failed', 0)}\nErrors: {result.get('errors', {})}"
        )
        self._refresh_cache_card()

    def delete_selected_people(self):
        selected = self._selected_people()
        if not selected:
            QMessageBox.information(self, "Chưa chọn", "Hãy chọn ít nhất một người.")
            return
        reply = QMessageBox.question(self, "Xác nhận xóa", f"Xóa {', '.join(selected)} khỏi database?")
        if reply != QMessageBox.Yes:
            return
        backup_root = os.path.join(APP_DIR, "temp_uploads", "deleted_people_backup")
        if os.path.isdir(backup_root):
            shutil.rmtree(backup_root, ignore_errors=True)
        os.makedirs(backup_root, exist_ok=True)
        db_dir = utils.get_faces_db_dir()
        backup_names = []
        for name in selected:
            src = os.path.join(db_dir, name)
            dst = os.path.join(backup_root, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                backup_names.append(name)
        result = utils.delete_registered_people(selected)
        self.last_deleted_backup = {"root": backup_root, "people": backup_names}
        deleted = result.get("deleted", [])
        errors = result.get("errors", [])
        not_deleted = [name for name in selected if name not in deleted]
        remaining_count = len(utils.list_registered_people())
        message_lines = [
            "Kết quả xóa database:",
            f"- Đã chọn: {', '.join(selected)}",
            f"- Đã xóa thành công: {', '.join(deleted) if deleted else 'không có'}",
            f"- Chưa xóa / không tìm thấy: {', '.join(not_deleted) if not_deleted else 'không có'}",
            f"- User còn lại: {remaining_count}",
            "- Có thể bấm 'Hoàn tác xóa' để khôi phục lần xóa gần nhất.",
        ]
        if errors:
            message_lines.append("- Lỗi: " + "; ".join(errors))
        delete_message = "\n".join(message_lines)
        self.refresh_database_tab(delete_message)
        self._refresh_cache_card()
        QMessageBox.information(self, "Đã xử lý xóa", delete_message)

    def undo_delete_people(self):
        if not self.last_deleted_backup or not self.last_deleted_backup.get("people"):
            QMessageBox.information(self, "Không có bản hoàn tác", "Chưa có lần xóa nào để hoàn tác.")
            return
        db_dir = utils.get_faces_db_dir()
        restored = []
        errors = []
        for name in self.last_deleted_backup["people"]:
            src = os.path.join(self.last_deleted_backup["root"], name)
            dst = os.path.join(db_dir, name)
            try:
                if os.path.exists(dst):
                    errors.append(f"{name}: đã tồn tại lại trong database")
                    continue
                shutil.copytree(src, dst)
                restored.append(name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if restored:
            utils.rebuild_embeddings_for_people(restored, None, force=True)
        self.refresh_database_tab()
        self._refresh_cache_card()
        self.db_info.append(f"\nĐã hoàn tác khôi phục: {', '.join(restored) or '-'}")
        if errors:
            self.db_info.append("Lỗi hoàn tác: " + "; ".join(errors))

    def start_camera(self):
        if self.worker and self.worker.isRunning():
            return
        config = RealtimeConfig(
            model_name=self.model_combo.currentText(),
            threshold=float(self.threshold_spin.value()),
            camera_index=int(self.camera_spin.value()),
            camera_backend=self.backend_combo.currentText(),
            recognize_every=float(self.interval_spin.value()),
            max_display_fps=int(self.fps_spin.value()),
        )
        self.worker = RealtimeWorker(config)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.result_ready.connect(self.update_result)
        self.worker.status_ready.connect(self.update_status)
        self.worker.fps_ready.connect(lambda fps: self.fps_card.set_value(f"{fps:.1f}"))
        self.worker.finished.connect(self._on_worker_finished)
        self._set_controls_enabled(False)
        self.status_label.setText("Đang mở camera...")
        self.worker.start()

    def stop_camera(self):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Đang dừng camera...")
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
        else:
            self._set_controls_enabled(True)

    def shutdown_camera(self):
        self.close_register_camera()
        self.close_static_camera()
        self.close_compare_camera()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)

    def _on_worker_finished(self):
        self.worker = None
        self._set_controls_enabled(True)

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_result(self, result: dict):
        if result.get("is_match"):
            self.name_card.set_value(result.get("name", "—"))
            self.status_label.setText(f"Đã nhận diện: {result.get('name', '—')}")
        elif result.get("error"):
            self.name_card.set_value("Chưa nhận diện")
            self.status_label.setText(result["error"])
        else:
            self.name_card.set_value(f"Không xác định\nGần nhất: {result.get('raw_name', '-')}")
            self.status_label.setText(f"Có mặt nhưng chưa khớp database. Gần nhất: {result.get('raw_name', '-')}")
        self.sim_card.set_value(f"{result.get('similarity', 0):.4f} / {result.get('threshold', 0):.3f}")

    def update_status(self, text: str):
        self.status_label.setText(text)

    def closeEvent(self, event):
        self.shutdown_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = DesktopFaceApp()
    app.aboutToQuit.connect(window.shutdown_camera)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
