import argparse
import os
import time
from pathlib import Path

QT_PLATFORM_PATH = Path("/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")
if QT_PLATFORM_PATH.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(QT_PLATFORM_PATH)
    os.environ["QT_PLUGIN_PATH"] = str(QT_PLATFORM_PATH.parent)

import numpy as np

from event_postgrasp_monitor import (
    DAVIS_FRAME_SIZE,
    DAVIS_STABILITY_WINDOW_SEC,
    EventPostGraspMonitor,
    _clip_roi,
)


def event_heatmap(events, roi_xywh=None, frame_size=DAVIS_FRAME_SIZE):
    width, height = frame_size
    heat = np.zeros((height, width), dtype=np.float32)
    arr = np.asarray(events, dtype=np.float64) if events else np.empty((0, 4), dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 4 and arr.shape[0]:
        xs = np.clip(arr[:, 1].astype(np.int32), 0, width - 1)
        ys = np.clip(arr[:, 2].astype(np.int32), 0, height - 1)
        np.add.at(heat, (ys, xs), 1.0)
    log_heat = np.log1p(heat)
    max_v = float(np.max(log_heat)) if log_heat.size else 0.0
    heat_u8 = (255.0 * log_heat / max(max_v, 1e-6)).astype(np.uint8)
    t = heat_u8.astype(np.float32) / 255.0
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[..., 0] = np.clip(255 * np.minimum(1.0, 1.8 * t), 0, 255).astype(np.uint8)
    color[..., 1] = np.clip(255 * np.maximum(0.0, 1.7 * t - 0.35), 0, 255).astype(np.uint8)
    color[..., 2] = np.clip(255 * np.maximum(0.0, 2.4 * t - 1.2), 0, 255).astype(np.uint8)
    x, y, w, h = _clip_roi(roi_xywh)
    color[y:y + h, x:x + 2] = (0, 255, 255)
    color[y:y + h, x + w - 2:x + w] = (0, 255, 255)
    color[y:y + 2, x:x + w] = (0, 255, 255)
    color[y + h - 2:y + h, x:x + w] = (0, 255, 255)
    if arr.shape[0]:
        cx, cy = int(np.mean(arr[:, 1])), int(np.mean(arr[:, 2]))
        if 0 <= cx < width and 0 <= cy < height:
            color[max(0, cy - 8):min(height, cy + 9), max(0, cx - 1):min(width, cx + 2)] = (0, 255, 0)
            color[max(0, cy - 1):min(height, cy + 2), max(0, cx - 8):min(width, cx + 9)] = (0, 255, 0)
    return color


class SlipDetectionWindow:
    def __init__(self, roi_xywh=None, sensitivity="sensitive"):
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

        self.Qt = Qt
        self.QImage = QImage
        self.QPixmap = QPixmap
        self.app = QApplication.instance() or QApplication([])
        self.monitor = EventPostGraspMonitor(enabled=True, roi_xywh=roi_xywh, sensitivity=sensitivity)
        self.report = self.monitor.start()

        self.window = QMainWindow()
        self.window.setWindowTitle("DAVIS346 Realtime Slip / Rotation Monitor")
        self.window.resize(1180, 760)
        self.window.setStyleSheet(
            "QWidget { background: #eef2f7; color: #0f172a; font-family: Arial; }"
            "QLabel { color: #0f172a; }"
        )
        central = QWidget()
        layout = QVBoxLayout(central)
        self.window.setCentralWidget(central)

        self.cards = {}
        top = QGridLayout()
        for i, name in enumerate(["Motion Class", "Decision", "Event Rate", "Slip", "Rotation", "Vibration", "Stability"]):
            title = QLabel(name)
            title.setStyleSheet("font-size: 11px; font-weight: 700; color: #475569;")
            value = QLabel("starting")
            value.setMinimumHeight(62)
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet(self.card_style("neutral"))
            top.addWidget(title, 0, i)
            top.addWidget(value, 1, i)
            self.cards[name] = value
        layout.addLayout(top)

        body = QGridLayout()
        self.heat_label = QLabel("event heatmap")
        self.info_label = QLabel("waiting for DAVIS events")
        self.heat_label.setMinimumSize(690, 520)
        self.info_label.setMinimumSize(390, 520)
        self.heat_label.setAlignment(Qt.AlignCenter)
        self.heat_label.setStyleSheet("background: #0b1020; border: 1px solid #cbd5e1; border-radius: 6px;")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 14px; font-size: 16px;")
        body.addWidget(self.heat_label, 0, 0)
        body.addWidget(self.info_label, 0, 1)
        layout.addLayout(body)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    def card_style(self, state):
        colors = {
            "good": ("#ecfdf5", "#16a34a", "#14532d"),
            "warn": ("#fff7ed", "#f97316", "#7c2d12"),
            "bad": ("#fef2f2", "#dc2626", "#7f1d1d"),
            "neutral": ("#f8fafc", "#94a3b8", "#0f172a"),
            "info": ("#eff6ff", "#2563eb", "#1e3a8a"),
        }
        bg, border, fg = colors.get(state, colors["neutral"])
        return f"background: {bg}; border: 1px solid {border}; border-radius: 6px; color: {fg}; font-size: 16px; font-weight: 700; padding: 8px;"

    def set_card(self, name, text, state="neutral"):
        self.cards[name].setText(str(text))
        self.cards[name].setStyleSheet(self.card_style(state))

    def pixmap(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        image = self.QImage(rgb.data, w, h, ch * w, self.QImage.Format_RGB888).copy()
        return self.QPixmap.fromImage(image).scaled(700, 540, self.Qt.KeepAspectRatio, self.Qt.SmoothTransformation)

    def tick(self):
        now = time.monotonic()
        origin = now - DAVIS_STABILITY_WINDOW_SEC[1]
        self.report = self.monitor.update(write_report=False, analysis_origin=origin, prune_before=origin)
        r = self.report
        motion = r.get("motion_class", "N/A")
        decision = r.get("post_grasp_decision", "N/A")
        slip = float(r.get("slip_risk_score", 0.0) or 0.0)
        rot = float(r.get("rotation_risk_score", 0.0) or 0.0)
        vib = float(r.get("vibration_score", 0.0) or 0.0)
        stable = float(r.get("post_grasp_stability_score", 0.0) or 0.0)
        rate = float(r.get("event_rate", 0.0) or 0.0)

        self.set_card("Motion Class", motion, "good" if motion in {"STABLE", "IMPACT_ONLY"} else "bad" if motion in {"SLIP_DOWN", "SPIN_OR_ROTATION"} else "warn")
        self.set_card("Decision", decision, "good" if decision == "READY_TO_LIFT" else "bad" if decision in {"REGRIP", "DO_NOT_LIFT"} else "warn")
        self.set_card("Event Rate", f"{rate:.0f}/s", "info" if rate > 0 else "neutral")
        self.set_card("Slip", f"{slip:.0f}", "bad" if slip >= 60 else "warn" if slip >= 35 else "good")
        self.set_card("Rotation", f"{rot:.0f}", "bad" if rot >= 60 else "warn" if rot >= 35 else "good")
        self.set_card("Vibration", f"{vib:.0f}", "bad" if vib >= 60 else "warn" if vib >= 35 else "good")
        self.set_card("Stability", f"{stable:.0f}", "good" if stable >= 70 else "warn" if stable >= 40 else "bad")

        heat = event_heatmap(self.monitor.events, r.get("davis_roi_xywh"))
        self.heat_label.setPixmap(self.pixmap(heat))
        self.info_label.setText(
            "Realtime DAVIS Event Slip Monitor\n\n"
            f"Backend: {r.get('davis_backend')}\n"
            f"Status: {r.get('davis_status')}\n"
            f"Events in rolling 1.2s: {r.get('event_count', 0)}\n"
            f"Active window: {r.get('active_window')}\n\n"
            f"Slip reason: {r.get('slip_risk_reason', '')}\n\n"
            f"Rotation reason: {r.get('rotation_risk_reason', '')}\n\n"
            f"Vibration reason: {r.get('vibration_risk_reason', '')}\n\n"
            "Move downward for slip. Twist for rotation. Shake for vibration."
        )

    def run(self):
        self.window.show()
        code = self.app.exec_()
        self.monitor.stop()
        return code


def main():
    parser = argparse.ArgumentParser(description="DAVIS346 realtime slip/rotation/vibration GUI.")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--sensitivity", choices=["conservative", "balanced", "sensitive"], default="sensitive")
    args = parser.parse_args()
    return SlipDetectionWindow(roi_xywh=args.roi, sensitivity=args.sensitivity).run()


if __name__ == "__main__":
    raise SystemExit(main())
