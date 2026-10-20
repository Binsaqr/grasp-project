import os
import sys
import time
from collections import deque

import numpy as np

QT_PLATFORM_PATH = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
if os.path.exists(QT_PLATFORM_PATH):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QT_PLATFORM_PATH

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from event_postgrasp_monitor import DAVIS_FRAME_SIZE, DAVIS_LIVE_HISTORY_SEC, DAVIS_LIVE_ROLLING_WINDOW_SEC, EventPostGraspMonitor


def aps_channel_label(frame):
    if frame.ndim == 2:
        return 1, "grayscale APS"
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return 1, "grayscale APS"
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return 3, "color APS/RGB"
    return frame.shape[-1] if frame.ndim >= 3 else 1, "unknown APS layout"


def image_to_pixmap(rgb, max_w=692, max_h=520):
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.FastTransformation)


class DavisEventWorker(QObject):
    frame_ready = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.monitor = EventPostGraspMonitor(enabled=True)
        self.surface = np.full((DAVIS_FRAME_SIZE[1], DAVIS_FRAME_SIZE[0], 3), 18, dtype=np.uint8)
        self.total_events = 0
        self.rate_batches = deque()
        self.started_at = None
        self.last_emit = 0.0
        self.latest_stats = self.empty_live_stats()
        self.last_aps_signature = None

    def empty_live_stats(self):
        return {
            "motion_class": "STILL",
            "decision": "LIVE_VIEW_ONLY",
            "slip": 0.0,
            "rotation": 0.0,
            "vibration": 0.0,
            "stability": 100.0,
            "slip_severity": "NONE",
            "rotation_severity": "NONE",
            "vibration_severity": "NONE",
            "confidence": 0.0,
            "vertical_bias": 0.0,
            "horizontal_bias": 0.0,
        }

    @pyqtSlot()
    def run(self):
        report = self.monitor.start()
        self.started_at = time.monotonic()
        if not self.monitor.active:
            self.frame_ready.emit(self.surface.copy(), {
                "status": report.get("davis_status", "unavailable"),
                "backend": report.get("davis_backend"),
                "reason": report.get("davis_unavailable_reason"),
                "total_events": 0,
                "event_rate": 0.0,
            })
            return

        try:
            self.drain_startup_buffer()
            while self.running:
                events, aps_frames = self.monitor.poll_event_and_frame_arrays()
                self.print_aps_diagnostics(aps_frames)
                now = time.monotonic()
                event_count = int(events.shape[0]) if isinstance(events, np.ndarray) else 0
                if event_count:
                    self.total_events += event_count
                    self.rate_batches.append((now, event_count))
                    self.draw_events(events)
                    self.latest_stats = self.compute_live_stats(events)
                while self.rate_batches and now - self.rate_batches[0][0] > DAVIS_LIVE_ROLLING_WINDOW_SEC:
                    self.rate_batches.popleft()
                if now - self.last_emit >= 0.10:
                    self.fade_surface()
                    frame = self.surface.copy()
                    self.draw_roi_overlay(frame, {})
                    rate = float(sum(count for _, count in self.rate_batches) / max(DAVIS_LIVE_ROLLING_WINDOW_SEC, 1e-6))
                    stats = dict(self.latest_stats)
                    self.frame_ready.emit(frame, {
                        "status": "streaming",
                        "backend": self.monitor.backend_name,
                        "reason": None,
                        "total_events": self.total_events,
                        "event_rate": rate,
                        **stats,
                    })
                    self.last_emit = now
                time.sleep(0.005)
        finally:
            self.monitor.stop()

    def print_aps_diagnostics(self, aps_frames):
        arr = np.asarray(aps_frames) if aps_frames is not None else np.empty((0,), dtype=np.uint8)
        if arr.size == 0:
            return
        frame = arr[-1] if arr.ndim >= 3 else arr
        channel_count, label = aps_channel_label(frame)
        signature = (tuple(frame.shape), str(frame.dtype), int(channel_count), label)
        if signature == self.last_aps_signature:
            return
        self.last_aps_signature = signature
        print(
            f"DAVIS APS frame: shape={frame.shape} dtype={frame.dtype} "
            f"channel_count={channel_count} label=\"{label}\"",
            flush=True,
        )

    def drain_startup_buffer(self):
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and self.monitor.active:
            self.monitor.poll_event_array()
            time.sleep(0.01)
        self.started_at = time.monotonic()
        self.total_events = 0
        self.rate_batches.clear()

    def draw_events(self, events):
        h, w = self.surface.shape[:2]
        if events is None:
            return
        arr = np.asarray(events)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return
        xs = arr[:, 1].astype(np.int32, copy=False)
        ys = arr[:, 2].astype(np.int32, copy=False)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(valid):
            return
        xs = xs[valid]
        ys = ys[valid]
        polarity = arr[:, 3][valid].astype(bool)
        self.surface[ys[polarity], xs[polarity]] = (255, 255, 255)
        self.surface[ys[~polarity], xs[~polarity]] = (30, 170, 255)

    def compute_live_stats(self, events):
        arr = np.asarray(events)
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[0] == 0:
            return self.empty_live_stats()
        x, y, w, h = self.monitor.roi_xywh
        xs = arr[:, 1]
        ys = arr[:, 2]
        in_roi = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
        roi = arr[in_roi]
        count = int(roi.shape[0])
        if count == 0:
            return self.empty_live_stats()
        rx = roi[:, 1]
        ry = roi[:, 2]
        x_mid = x + w / 2.0
        y_mid = y + h / 2.0
        right = int(np.count_nonzero(rx > x_mid))
        left = count - right
        lower = int(np.count_nonzero(ry > y_mid))
        upper = count - lower
        vertical_bias = float((lower - upper) / max(count, 1))
        horizontal_bias = float((right - left) / max(count, 1))
        activity = min(100.0, count / 20.0)
        slip = float(np.clip(activity * max(vertical_bias, 0.0), 0.0, 100.0))
        rotation = float(np.clip(activity * abs(horizontal_bias), 0.0, 100.0))
        broadness = min(1.0, float(np.std(rx) / max(w, 1) + np.std(ry) / max(h, 1)))
        vibration = float(np.clip(activity * broadness * (1.0 - max(abs(vertical_bias), abs(horizontal_bias)) * 0.5), 0.0, 100.0))
        if count < 30:
            motion = "STILL"
            reason = "Low live event activity."
        elif slip >= 35.0:
            motion = "DOWNWARD_ACTIVITY"
            reason = f"Live ROI lower-half activity dominates; vertical bias={vertical_bias:.2f}."
        elif rotation >= 35.0:
            motion = "SIDE_ASYMMETRY"
            reason = f"Live ROI side activity is asymmetric; horizontal bias={horizontal_bias:.2f}."
        elif vibration >= 35.0:
            motion = "BROAD_ACTIVITY"
            reason = "Live ROI activity is broad without one dominant direction."
        else:
            motion = "MOTION"
            reason = "Live ROI motion detected, but not clearly slip/spin/vibration."
        return {
            "motion_class": motion,
            "decision": "LIVE_VIEW_ONLY",
            "slip": slip,
            "rotation": rotation,
            "vibration": vibration,
            "stability": float(np.clip(100.0 - max(slip, rotation, vibration), 0.0, 100.0)),
            "slip_severity": self.severity(slip),
            "rotation_severity": self.severity(rotation),
            "vibration_severity": self.severity(vibration),
            "confidence": float(np.clip(count / 5.0, 0.0, 100.0)),
            "vertical_bias": vertical_bias,
            "horizontal_bias": horizontal_bias,
            "reason": reason,
        }

    def severity(self, score):
        score = float(score or 0.0)
        if score < 18.0:
            return "NONE"
        if score < 40.0:
            return "LOW"
        if score < 70.0:
            return "MEDIUM"
        return "HIGH"

    def fade_surface(self):
        self.surface = np.maximum((self.surface.astype(np.float32) * 0.86).astype(np.uint8), 12)

    def draw_roi_overlay(self, frame, report):
        x, y, w, h = self.monitor.roi_xywh
        x2 = min(frame.shape[1] - 1, x + w)
        y2 = min(frame.shape[0] - 1, y + h)
        frame[y:y2, x] = (0, 255, 255)
        frame[y:y2, x2] = (0, 255, 255)
        frame[y, x:x2] = (0, 255, 255)
        frame[y2, x:x2] = (0, 255, 255)
        frame[y:y2, int(x + w / 2)] = (0, 150, 255)
        frame[int(y + h / 2), x:x2] = (0, 150, 255)
        active = ((report.get("window_metrics") or {}).get(report.get("active_window", "stability")) or {})
        if active.get("mean_x") is not None and active.get("mean_y") is not None:
            cx = int(round(active["mean_x"]))
            cy = int(round(active["mean_y"]))
            if 2 <= cx < frame.shape[1] - 2 and 2 <= cy < frame.shape[0] - 2:
                frame[cy - 2:cy + 3, cx] = (0, 255, 0)
                frame[cy, cx - 2:cx + 3] = (0, 255, 0)

    def stop(self):
        self.running = False


class DavisEventWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAVIS346 Event Stream")
        self.resize(960, 760)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.status = QLabel("Opening DAVIS...")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("font-size: 18px; padding: 10px;")
        layout.addWidget(self.status)

        self.image = QLabel("waiting for events")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(700, 540)
        self.image.setStyleSheet("background: #0b1020; color: white; border: 1px solid #475569;")
        layout.addWidget(self.image)

        self.metric_grid = QGridLayout()
        self.metric_labels = {}
        for idx, title in enumerate(["Motion Class", "Decision", "Event Rate", "Slip Risk", "Rotation Risk", "Vibration Risk", "Stability", "Confidence"]):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            value_label = QLabel("waiting")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setWordWrap(True)
            value_label.setStyleSheet("padding: 8px; border: 1px solid #94a3b8; background: #f8fafc;")
            self.metric_grid.addWidget(title_label, 0, idx)
            self.metric_grid.addWidget(value_label, 1, idx)
            self.metric_labels[title] = value_label
        layout.addLayout(self.metric_grid)

        self.reason_label = QLabel("Live DAVIS view: post-grasp risk classification uses impact/settle/stability windows in event_postgrasp_monitor.py.")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("padding: 8px; color: #0f172a; background: #e2e8f0;")
        layout.addWidget(self.reason_label)

        self.reset_button = QPushButton("Clear View")
        self.reset_button.clicked.connect(self.clear_view)
        layout.addWidget(self.reset_button)

        self.thread = QThread()
        self.worker = DavisEventWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.thread.start()

        self.latest_frame = None

    @pyqtSlot(object, dict)
    def on_frame(self, frame, stats):
        self.image.setPixmap(image_to_pixmap(frame))
        reason = stats.get("reason")
        if reason:
            self.status.setText(f"{stats.get('status')} | backend={stats.get('backend')} | {reason}")
        else:
            self.status.setText(
                f"{stats.get('status')} | backend={stats.get('backend')} | "
                f"events={stats.get('total_events', 0)} | rate={stats.get('event_rate', 0.0):.0f}/s"
            )
        self.metric_labels["Motion Class"].setText(str(stats.get("motion_class", "N/A")))
        self.metric_labels["Decision"].setText(str(stats.get("decision", "N/A")))
        self.metric_labels["Event Rate"].setText(f"{float(stats.get('event_rate', 0.0)):.0f} ev/s")
        self.metric_labels["Slip Risk"].setText(f"{float(stats.get('slip', 0.0)):.0f}/100 {stats.get('slip_severity', 'NONE')}")
        self.metric_labels["Rotation Risk"].setText(f"{float(stats.get('rotation', 0.0)):.0f}/100 {stats.get('rotation_severity', 'NONE')}")
        self.metric_labels["Vibration Risk"].setText(f"{float(stats.get('vibration', 0.0)):.0f}/100 {stats.get('vibration_severity', 'NONE')}")
        self.metric_labels["Stability"].setText(f"{float(stats.get('stability', 0.0)):.0f}/100")
        self.metric_labels["Confidence"].setText(f"{float(stats.get('confidence', 0.0)):.0f}/100")
        self.reason_label.setText(str(stats.get("reason", "Live DAVIS view is not the full post-grasp windowed classifier.")))

    def clear_view(self):
        if self.worker:
            self.worker.surface[:] = 18

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(1500)
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = DavisEventWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
