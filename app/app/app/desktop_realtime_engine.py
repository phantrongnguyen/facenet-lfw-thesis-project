"""Realtime engine for the native App3 desktop application.

This module intentionally does not depend on Streamlit and does not modify app3.py.
It reuses app3_utils for cached recognition and frame preparation.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for path in (PROJECT_ROOT, APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import app3_utils as utils

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QImage
except ModuleNotFoundError as exc:  # pragma: no cover - gives app_desktop a clean message
    raise ModuleNotFoundError(
        "PySide6 is required for the native desktop app. Install it with: python -m pip install PySide6"
    ) from exc


@dataclass
class RealtimeConfig:
    model_name: str
    threshold: Optional[float]
    camera_index: int = 0
    camera_backend: str = "DirectShow"
    recognize_every: float = 2.0
    max_display_fps: int = 24


class RealtimeWorker(QThread):
    """Camera + recognition worker that never blocks the Qt UI thread."""

    frame_ready = Signal(QImage)
    result_ready = Signal(dict)
    status_ready = Signal(str)
    fps_ready = Signal(float)
    stopped_ready = Signal()

    def __init__(self, config: RealtimeConfig):
        super().__init__()
        self.config = config
        self._running = False
        self._last_result = None
        self._last_face_seen = 0.0

    def stop(self):
        # Do not call wait() here. DeepFace/TensorFlow calls can take a long time
        # and blocking this method freezes Qt during Stop/closeEvent.
        self._running = False

    def run(self):
        cap, message = utils.open_camera_capture(self.config.camera_index, self.config.camera_backend)
        self.status_ready.emit(message)
        if cap is None:
            return

        self._running = True
        # Let the camera stream appear first. Loading TensorFlow/DeepFace on the
        # first loop makes the app look like the camera failed to open.
        last_recognition_at = time.time()
        first_recognition_delay = 1.5
        last_fps_at = time.time()
        frame_count = 0
        min_frame_delay = 1.0 / max(1, int(self.config.max_display_fps))

        recognition_future = None
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            while self._running:
                loop_started = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status_ready.emit("Không đọc được frame. Camera có thể đang bị app khác chiếm.")
                    break

                display_frame, recognition_roi, overlay_box = utils.prepare_realtime_frame(frame)
                now = time.time()

                painted = utils.draw_recognition_overlay(display_frame.copy(), self._last_result, overlay_box)
                rgb = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

                if recognition_roi is None:
                    if now - self._last_face_seen >= 1.5:
                        self.status_ready.emit("Chưa detect được khuôn mặt - đưa mặt vào rõ giữa camera, đủ sáng.")
                        self._last_face_seen = now
                    self._last_result = None
                else:
                    self._last_face_seen = now

                if recognition_future is not None and recognition_future.done():
                    try:
                        self._last_result = recognition_future.result()
                        self.result_ready.emit(self._last_result)
                    except Exception as exc:
                        self._last_result = {"error": f"Lỗi nhận diện: {exc}"}
                        self.result_ready.emit(self._last_result)
                    finally:
                        recognition_future = None

                if (
                    recognition_future is None
                    and recognition_roi is not None
                    and now - last_recognition_at >= float(self.config.recognize_every)
                    and now - last_recognition_at >= first_recognition_delay
                ):
                    last_recognition_at = now
                    self.status_ready.emit("Đang nhận diện frame hiện tại...")
                    recognition_future = executor.submit(
                        utils.safe_recognize_frame_cached,
                        recognition_roi.copy(),
                        self.config.model_name,
                        self.config.threshold,
                    )
                    first_recognition_delay = 0.0

                frame_count += 1
                if now - last_fps_at >= 1.0:
                    self.fps_ready.emit(frame_count / max(1e-6, now - last_fps_at))
                    frame_count = 0
                    last_fps_at = now

                elapsed = time.time() - loop_started
                if elapsed < min_frame_delay:
                    time.sleep(min_frame_delay - elapsed)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if cap is not None:
                cap.release()
            self._running = False
            self.status_ready.emit("Đã dừng camera và release thiết bị.")
            self.stopped_ready.emit()
