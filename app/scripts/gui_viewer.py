import json
import os
import sys

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


OUTPUT_DIR = "output"


class GraspGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Perception-Based Grasp Quality Evaluation")
        self.resize(1700, 1050)

        window_layout = QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        window_layout.addWidget(self.scroll_area)

        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        main_layout.addWidget(self.section_title("Summary Dashboard"))
        realtime_status = self.load_json("output/realtime_status.json", {})
        if realtime_status:
            main_layout.addWidget(self.build_realtime_stage_dashboard(realtime_status))
        main_layout.addWidget(self.build_summary_dashboard())

        main_layout.addWidget(self.section_title("Main Results"))
        main_grid = QGridLayout()
        main_grid.setSpacing(12)
        main_grid.addWidget(self.image_panel("Scene View", "output/scene.png", "Current RGB frame from the camera."), 0, 0)
        main_grid.addWidget(self.image_panel("Bottle Detection / Segmentation View", "output/segmentation.png", "Cleaned RGB-D bottle mask, contour, and body region."), 0, 1)
        main_grid.addWidget(self.image_panel("Planned Grasp Overlay", "output/best_grasp_overlay.png", "Bottle centerline, planned grasp line, target contacts, width, and alignment."), 1, 0)
        main_grid.addWidget(self.report_text_panel(), 1, 1)
        main_layout.addLayout(main_grid)

        if realtime_status:
            main_layout.addWidget(self.section_title("State Timeline"))
            main_layout.addWidget(self.state_history_panel())

        multiview = self.load_json("output/multiview_report.json", {})
        if multiview:
            main_layout.addWidget(self.section_title("Multi-View Evaluation"))
            multiview_grid = QGridLayout()
            multiview_grid.setSpacing(12)
            multiview_grid.addWidget(
                self.image_panel(
                    "Multi-View Summary",
                    "output/multiview_summary.png",
                    "Saved RGB-D viewpoints ranked by hard gates, score, depth confidence, epsilon, and risk.",
                    require_current=False,
                ),
                0,
                0,
            )
            multiview_grid.addWidget(self.multiview_report_panel(multiview), 0, 1)
            main_layout.addLayout(multiview_grid)

        main_layout.addWidget(self.section_title("Wrench Safety Summary"))
        wrench_summary_grid = QGridLayout()
        wrench_summary_grid.setSpacing(12)
        wrench_summary_grid.addWidget(
            self.image_panel(
                "Wrench-Space Safety",
                "output/wrench_safety_summary.png",
                "Practical interpretation of force closure, epsilon margin, push resistance, slip resistance, and rotation resistance.",
            ),
            0,
            0,
        )
        wrench_summary_grid.addWidget(self.wrench_safety_summary_panel(), 0, 1)
        main_layout.addLayout(wrench_summary_grid)

        main_layout.addWidget(self.section_title("Advanced Wrench Debug"))
        wrench_grid = QGridLayout()
        wrench_grid.setSpacing(12)
        wrench_grid.addWidget(
            self.image_panel(
                "Wrench Fx/Fy",
                "output/wrench_space_fx_fy.png",
                "Fx/Fy: force balance projection. The green circle is only a 2D projection; the green marker/line shows the projected 3D epsilon touch point.",
            ),
            0,
            0,
        )
        wrench_grid.addWidget(
            self.image_panel(
                "Wrench Fx/Tau",
                "output/wrench_space_fx_tau.png",
                "Fx/Tau: horizontal force vs torque resistance. The marker shows the closest hull face / epsilon touch point projected into this view.",
            ),
            0,
            1,
        )
        wrench_grid.addWidget(
            self.image_panel(
                "Wrench Fy/Tau",
                "output/wrench_space_fy_tau.png",
                "Fy/Tau: vertical force vs torque resistance. Actual epsilon is computed in 3D planar wrench space [Fx, Fy, tau].",
            ),
            1,
            0,
        )
        wrench_grid.addWidget(self.wrench_analysis_panel(), 1, 1)
        main_layout.addLayout(wrench_grid)

        debug_widget = QWidget()
        debug_layout = QVBoxLayout(debug_widget)
        debug_layout.setContentsMargins(0, 0, 0, 0)
        debug_layout.addWidget(self.section_title("Debug Views"))

        debug_grid = QGridLayout()
        debug_grid.setSpacing(12)
        pregrasp_report = self.load_json("output/pregrasp_safety_report.json", {})
        if pregrasp_report.get("pregrasp_decision") and self.is_current_pregrasp_report():
            debug_grid.addWidget(self.image_panel("Mask Cleanup Debug", "output/mask_cleanup_debug.png", "Raw candidate mask, cleaned depth-filtered mask, and body-only mask."), 0, 0)
            debug_grid.addWidget(self.image_panel("Smoothed Depth Map", "output/depth_smooth_colormap.png", "Smoothed depth image with planned contact targets."), 0, 1)
            debug_grid.addWidget(self.image_panel("Raw Depth Map", "output/depth_raw_colormap.png", "Unfiltered raw depth image kept for diagnostics."), 1, 0)
            debug_grid.addWidget(self.image_panel("Depth Noise Debug", "output/depth_noise_debug.png", "Raw depth, smoothed depth, and absolute difference."), 1, 1)
        else:
            debug_grid.addWidget(self.image_panel("RGB-D Contact Debug", "output/rgbd_contact_validation_debug.png", "Legacy physical contact validation debug view.", require_current=False), 0, 0)
            debug_grid.addWidget(self.image_panel("Depth Map", "output/depth_colormap.png", "Depth image with contact targets.", require_current=False), 0, 1)
            debug_grid.addWidget(self.image_panel("Mask Cleanup Debug", "output/mask_cleanup_debug.png", "Raw candidate mask, cleaned depth-filtered mask, and body-only mask.", require_current=False), 1, 0, 1, 2)
        debug_layout.addLayout(debug_grid)

        main_layout.addWidget(debug_widget)
        main_layout.addStretch(1)

        scroll_content.adjustSize()
        self.scroll_area.setWidget(scroll_content)
        self.install_wheel_filter(scroll_content)
        self.scroll_area.viewport().installEventFilter(self)

    def install_wheel_filter(self, widget):
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta:
                bar = self.scroll_area.verticalScrollBar()
                bar.setValue(bar.value() - delta)
                return True
        return super().eventFilter(watched, event)

    def section_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; margin-top: 6px;")
        return label

    def build_summary_dashboard(self):
        pregrasp = self.load_json("output/pregrasp_safety_report.json", {})
        if pregrasp.get("pregrasp_decision") and self.is_current_pregrasp_report():
            return self.build_pregrasp_summary_dashboard(pregrasp)

        contacts = self.load_json("output/contact_points.json", {})
        epsilon = self.load_json("output/epsilon_result.json", {})

        left = contacts.get("left", {})
        right = contacts.get("right", {})
        object_detected = os.path.exists("output/object_mask.png")

        rgbd_valid = bool(left.get("contact_valid")) and bool(right.get("contact_valid"))
        rgbd_state = "Valid" if rgbd_valid else ("Uncertain" if left or right else "Invalid")

        cards = [
            ("Object detected", "Yes" if object_detected else "No", "good" if object_detected else "bad"),
            ("Left gripper", self.gripper_card_text(left), self.side_state(left)),
            ("Right gripper", self.gripper_card_text(right), self.side_state(right)),
            ("RGB-D contact", rgbd_state, "good" if rgbd_valid else "warn"),
            ("Epsilon closure", str(bool(epsilon.get("force_closure", False))), "good" if epsilon.get("force_closure") else "bad"),
            ("Final quality", self.quality_from_score(epsilon), self.quality_state(epsilon)),
            ("Final score", self.score_text(epsilon), self.quality_state(epsilon)),
        ]

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (title, value, state) in enumerate(cards):
            grid.addWidget(self.status_card(title, value, state), 0, i)

        container = QWidget()
        container.setLayout(grid)
        return container

    def build_pregrasp_summary_dashboard(self, report):
        decision = report.get("pregrasp_decision", report.get("decision", "DO NOT CLOSE"))
        score = float(report.get("safety_score", report.get("final_score", 0.0)) or 0.0)
        raw_score = float(report.get("raw_weighted_score", score) or 0.0)
        scores = report.get("scores", {})
        risks = report.get("risk_scores", {})
        plan = report.get("planned_contacts", {})
        epsilon = report.get("epsilon_result", {})
        wrench_summary = report.get("wrench_safety_summary", {})
        active = report.get("active_positioning", {})
        full_visible = bool(report.get("full_bottle_visible", False))
        contact_ready = bool(report.get("contact_model_ready", bool(plan)))

        def state_from_score(value):
            value = float(value or 0.0)
            if value >= 75:
                return "good"
            if value >= 55:
                return "warn"
            return "bad"

        decision_state = "good" if decision in {"READY TO CLOSE", "SAFE TO GRIP/LIFT"} else ("warn" if decision in {"ADJUST POSITION", "ADJUST_VIEW"} else "bad")
        positioning_score = plan.get("positioning_guidance_score", plan.get("workspace_reachability_score", scores.get("positioning_guidance_score")))
        action_state = active.get("state", "POSITIONING")
        cards = [
            ("Full Bottle Visible", "Yes" if full_visible else "No", "good" if full_visible else "bad"),
            ("View Status", report.get("visibility", {}).get("view_status", "UNKNOWN"), "good" if full_visible else "warn"),
            ("Recommended View Action", report.get("recommended_view_action", active.get("recommended_action", "UNKNOWN")), "good" if report.get("recommended_view_action") == "VIEW_OK" else "warn"),
            ("Planned Contact Model", "Yes" if contact_ready else "No", "good" if contact_ready else "bad"),
            ("Recommended Action", active.get("recommended_action", "UNKNOWN"), "good" if action_state == "READY_TO_CLOSE" else ("bad" if action_state == "UNSAFE" else "warn")),
            ("Positioning State", action_state, "good" if action_state == "READY_TO_CLOSE" else ("bad" if action_state == "UNSAFE" else "warn")),
            ("Pre-Grasp Decision", decision, decision_state),
            ("Raw Score", f"{raw_score:.1f}/100", state_from_score(raw_score)),
            ("Decision Score", f"{score:.1f}/100", state_from_score(score)),
            ("Cap Reason", report.get("cap_reason", "none"), "warn" if report.get("cap_reason", "none") != "none" else "good"),
            ("Bottle detected", "Yes" if report.get("bottle_detected") else "No", "good" if report.get("bottle_detected") else "bad"),
            ("Alignment", f"{report.get('alignment_error_px', 0):.1f} px", state_from_score(scores.get("alignment"))),
            ("Grip plan", f"{plan.get('grip_width_px', 0):.0f} px", state_from_score(scores.get("grip_width_reachability"))),
            ("Positioning Guidance", f"{float(positioning_score or 0):.0f}/100", state_from_score(positioning_score)),
            ("Miss risk", f"{risks.get('miss_risk', 0):.0f}/100", "good" if risks.get("miss_risk", 100) <= 35 else "warn"),
            ("Slip risk", f"{risks.get('slip_risk', 0):.0f}/100", "good" if risks.get("slip_risk", 100) <= 35 else "warn"),
            ("Rotation risk", f"{risks.get('rotation_risk', 0):.0f}/100", "good" if risks.get("rotation_risk", 100) <= 35 else "warn"),
            ("Wrench safety", wrench_summary.get("overall_wrench_safety", "UNKNOWN"), "good" if wrench_summary.get("overall_wrench_safety") == "PASS" else "bad"),
            ("Epsilon closure", str(bool(epsilon.get("force_closure", False))), "good" if epsilon.get("force_closure") else "bad"),
        ]

        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (title, value, state) in enumerate(cards):
            grid.addWidget(self.status_card(title, value, state), i // 5, i % 5)

        container = QWidget()
        container.setLayout(grid)
        return container

    def build_realtime_stage_dashboard(self, status):
        score = float(status.get("safety_score", 0.0) or 0.0)
        decision = status.get("stage_decision", "UNKNOWN")
        state = status.get("current_state", "UNKNOWN")

        def state_from_decision(value):
            if value in {"VIEW_OK", "CONTACT_MODEL_READY", "READY_TO_APPROACH", "READY_TO_CLOSE", "READY_TO_LIFT", "MONITOR_ONLY"}:
                return "good"
            if value in {"ADJUST_POSITION", "ADJUST_VIEW", "CONTACT_MODEL_DISABLED", "BACK_UP", "SEARCH", "REGRIP"}:
                return "warn"
            return "bad"

        def state_from_score(value):
            if value >= 80:
                return "good"
            if value >= 60:
                return "warn"
            return "bad"

        cards = [
            ("Current Stage", status.get("current_stage", state), state_from_decision(decision)),
            ("Stage Decision", decision, state_from_decision(decision)),
            ("Recommended Action", status.get("recommended_action", "UNKNOWN"), state_from_decision(decision)),
            ("Safety Score", f"{score:.1f}/100", state_from_score(score)),
            ("State Machine", state, "bad" if state == "FAIL_SAFE" else ("good" if state in {"READY_TO_CLOSE", "READY_TO_LIFT"} else "warn")),
        ]
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (title, value, card_state) in enumerate(cards):
            grid.addWidget(self.status_card(title, value, card_state), 0, i)
        container = QWidget()
        container.setLayout(grid)
        return container

    def status_card(self, title, value, state):
        colors = {
            "good": ("#e7f5ea", "#1b5e20"),
            "warn": ("#fff4e5", "#b45309"),
            "bad": ("#fdeaea", "#b91c1c"),
        }
        bg, fg = colors.get(state, colors["warn"])
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 2px solid {fg}; border-radius: 8px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {fg};")
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setWordWrap(True)
        value_label.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {fg};")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame

    def current_generated_images(self):
        report = self.load_json("output/pregrasp_safety_report.json", {})
        if not report.get("run_id"):
            return None
        return set(report.get("generated_images", []))

    def image_belongs_to_current_run(self, image_path):
        generated = self.current_generated_images()
        if generated is None:
            return True
        return os.path.basename(image_path) in generated

    def image_panel(self, title, image_path, caption, require_current=True):
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 17px; font-weight: bold;")

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(620, 330)

        is_current = self.image_belongs_to_current_run(image_path) if require_current else True
        pixmap = QPixmap(image_path) if is_current else QPixmap()
        if not is_current:
            image_label.setText(f"Not generated in current run\n{image_path}")
            image_label.setStyleSheet("font-size: 15px; color: #b45309; border: 1px solid #999;")
        elif pixmap.isNull():
            image_label.setText(f"Not generated in current run\n{image_path}")
            image_label.setStyleSheet("font-size: 15px; color: #b91c1c; border: 1px solid #999;")
        else:
            image_label.setPixmap(pixmap.scaled(700, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        caption_label = QLabel(caption)
        caption_label.setAlignment(Qt.AlignCenter)
        caption_label.setWordWrap(True)
        caption_label.setStyleSheet("font-size: 13px; color: #444;")

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame { background: #fafafa; border: 1px solid #cfcfcf; border-radius: 6px; } QLabel { border: none; }")
        layout = QVBoxLayout(frame)
        layout.addWidget(title_label)
        layout.addWidget(image_label)
        layout.addWidget(caption_label)
        return frame

    def report_text_panel(self):
        pregrasp = self.load_json("output/pregrasp_safety_report.json", {})
        if pregrasp.get("pregrasp_decision") and self.is_current_pregrasp_report():
            return self.pregrasp_report_text_panel(pregrasp)

        contacts = self.load_json("output/contact_points.json", {})
        epsilon = self.load_json("output/epsilon_result.json", {})
        left = contacts.get("left", {})
        right = contacts.get("right", {})

        warnings = []
        for side_name, side in [("Left", left), ("Right", right)]:
            if side.get("warning"):
                warnings.append(f"{side_name}: {side.get('warning')}")
            if not side.get("contact_valid"):
                warnings.append(f"{side_name}: contact invalid/uncertain")
        if epsilon.get("error"):
            warnings.append(f"Epsilon: {epsilon.get('error')}")
        if not epsilon.get("force_closure", False):
            warnings.append("Origin outside wrench hull or force closure false")

        def fmt(value, digits=2):
            if value is None:
                return "unknown"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        def side_block(name, side):
            return f"""{name}
  Mode: {side.get('gripper_detection_mode', 'unknown')}
  Block confidence: {fmt(side.get('block_confidence'))}
  Thin-finger confidence: {fmt(side.get('thin_finger_confidence'))}
  Selected mode reason: {side.get('selected_mode_reason', 'n/a')}
  Contact valid: {side.get('contact_valid')}
  Contact source: {side.get('selected_contact_source', 'unknown')}
  Object contact px: {side.get('object_point_xy')}
  Gripper sample px: {side.get('gripper_face_point_xy')}
  Pixel distance: {fmt(side.get('distance_px'))} px
  Object depth: {fmt(side.get('object_depth_mm'))} mm
  Gripper depth: {fmt(side.get('gripper_face_depth_mm'))} mm
  Depth difference: {fmt(side.get('depth_difference_mm'))} mm
  Depth valid ratio: obj {fmt(side.get('object_depth_valid_ratio'))}, grip {fmt(side.get('gripper_depth_valid_ratio'))}
  Gripper vs object median: {fmt(side.get('gripper_depth_difference_from_object_median_mm'))} mm
  Normal: {side.get('normal_source', 'unknown')} conf {fmt(side.get('normal_confidence'))}"""

        object_depth = left or right
        text = f"""Epsilon Quality Report

Detection
{side_block("Left gripper", left)}

{side_block("Right gripper", right)}

RGB-D Contact
  Object median depth: {fmt(object_depth.get('object_depth_median_mm'))} mm
  Object mean/std depth: {fmt(object_depth.get('object_depth_mean_mm'))} / {fmt(object_depth.get('object_depth_std_mm'))} mm
  Object depth valid ratio: {fmt(object_depth.get('object_depth_valid_ratio_global'))}
  Object depth band: {object_depth.get('object_depth_band_mm')}
  Depth affected selection: L={left.get('depth_affected_selection')} R={right.get('depth_affected_selection')}

Epsilon Metric
  Epsilon: {fmt(epsilon.get('epsilon'), 4)}
  Force closure: {epsilon.get('force_closure')}
  Wrench points: {epsilon.get('num_wrench_points')}
  Friction coefficient: {fmt(epsilon.get('mu'))}
  Weak direction: {epsilon.get('weakest_direction')}
  Closest hull point: {epsilon.get('closest_hull_point')}
  Closest face distance: {fmt(epsilon.get('closest_face_distance'), 4)}

Scores
  Final score is shown in the summary card and saved report image.
  Epsilon-derived quality: {self.quality_from_score(epsilon)}

Warnings
  {chr(10).join('- ' + warning for warning in warnings) if warnings else 'No critical warnings.'}

Final Result
  RGB-D contact: {"Valid" if left.get('contact_valid') and right.get('contact_valid') else "Invalid/Uncertain"}
  Force closure: {epsilon.get('force_closure')}
  Final quality: {self.quality_from_score(epsilon)}"""

        has_warnings = bool(warnings)
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        border = "#b45309" if has_warnings else "#1b5e20"
        bg = "#fff8ed" if has_warnings else "#f4fbf5"
        frame.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 2px solid {border}; border-radius: 6px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet("font-size: 16px; color: #222; font-family: monospace;")
        layout.addWidget(label)
        return frame

    def pregrasp_report_text_panel(self, report):
        def fmt(value, digits=2):
            if value is None:
                return "unknown"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        scores = report.get("scores", {})
        risks = report.get("risk_scores", {})
        geometry = report.get("bottle_geometry", {})
        plan = report.get("planned_contacts", {})
        workspace = plan.get("gripper_workspace", {})
        workspace_zones = workspace.get("zones", {})
        active = report.get("active_positioning", {})
        failures = report.get("failure_reasons", [])
        reposition = report.get("reposition_reasons", [])
        warnings = report.get("warnings", [])
        epsilon = report.get("epsilon_result", {})
        wrench_summary = report.get("wrench_safety_summary", {})
        metric_audit = report.get("metric_audit", [])
        hard_gates = report.get("hard_gates", {})
        calibration = report.get("calibration_status", {})
        validation = report.get("known_bottle_validation", {})
        validation_measured = validation.get("measured", {})
        validation_expected = validation.get("expected", {})
        validation_checks = validation.get("checks", {})
        visibility = report.get("visibility", {})
        selected_contact = report.get("selected_contact_pair")
        candidates = report.get("candidate_contacts", [])
        rejected_candidates = report.get("rejected_candidates", [])
        metric_lines = []
        for metric in metric_audit:
            status = "PASS" if metric.get("passed") else "FAIL"
            metric_lines.append(
                f"- {metric.get('name')}: {fmt(metric.get('value'))}/100, "
                f"{status}, threshold {metric.get('thresholds')}, "
                f"weight {fmt(metric.get('weight'))}, contribution {fmt(metric.get('weighted_contribution'))}"
            )

        text = f"""Pre-Grasp Safety Report

Decision
  Run id: {report.get('run_id')}
  {report.get('pregrasp_decision', report.get('decision', 'DO NOT CLOSE'))}
  Stage: {report.get('decision_stage', 'pre_close')}
  Raw weighted score: {fmt(report.get('raw_weighted_score'))} / 100
  Capped decision score: {fmt(report.get('capped_decision_score', report.get('safety_score', report.get('final_score'))))} / 100
  Cap reason: {report.get('cap_reason', 'none')}
  Legacy lift wording: {report.get('legacy_decision')}
  Generated images this run: {report.get('generated_images', [])}

Hard Gates
  Passed overall: {hard_gates.get('passed')}
  {chr(10).join('- ' + name + ': ' + ('PASS' if passed else 'FAIL') for name, passed in hard_gates.get('status', {}).items()) if hard_gates.get('status') else 'No gate details available.'}

Bottle Geometry
  Bottle center px: {geometry.get('center_px')}
  Centerline alignment error: {fmt(report.get('alignment_error_px'))} px
  BBox xywh: {geometry.get('bbox_xywh')}
  Axis angle: {fmt(geometry.get('axis_angle_degrees'))} deg
  Bottle depth: {fmt(report.get('depth_stats', {}).get('bottle_depth_median_mm'))} mm
  Depth confidence: {fmt(report.get('depth_stats', {}).get('bottle_depth_confidence'))}
  Depth smoothing enabled: {report.get('depth_stats', {}).get('depth_smoothing_enabled')}
  Raw depth median: {fmt(report.get('depth_stats', {}).get('raw_depth_median_mm'))} mm
  Smoothed depth median: {fmt(report.get('depth_stats', {}).get('smoothed_depth_median_mm'))} mm
  Raw/smoothed valid ratio: {fmt(report.get('depth_stats', {}).get('raw_depth_valid_ratio'))} / {fmt(report.get('depth_stats', {}).get('smoothed_depth_valid_ratio'))}

Full Bottle Visibility
  Full bottle visible: {report.get('full_bottle_visible')}
  Visibility score: {fmt(report.get('visibility_score'))} / 100
  View status: {visibility.get('view_status')}
  Recommended view action: {report.get('recommended_view_action')}
  Border margins L/R/T/B: {report.get('border_margins')}
  Required margin: {fmt(visibility.get('required_margin_px'))} px
  Missing region reasons: {visibility.get('missing_region_reason', [])}
  Contact model ready: {report.get('contact_model_ready')}

Known Bottle Validation
  Expected diameter: {validation_expected.get('bottle_diameter_mm_range')} mm
  Measured grip width: {fmt(validation_measured.get('planned_grip_width_px'))} px / {fmt(validation_measured.get('planned_grip_width_mm'))} mm
  Expected aspect: {validation_expected.get('bbox_aspect_ratio_range')}
  Measured bbox width/height/aspect: {validation_measured.get('bbox_width_px')} px / {validation_measured.get('bbox_height_px')} px / {fmt(validation_measured.get('bbox_aspect_ratio'))}
  Expected depth range: {validation_expected.get('depth_mm_range')} mm
  Measured depth: {fmt(validation_measured.get('bottle_depth_median_mm'))} mm
  Checks: {validation_checks}
  Candidate score: {validation_measured.get('candidate_score_breakdown', {})}

Calibration Status
  workspace_config_source: {calibration.get('workspace_config_source')}
  grip_width_limits_source: {calibration.get('grip_width_limits_source')}
  calibration warnings: {calibration.get('warnings', [])}

Active Positioning
  State: {active.get('state')}
  Recommended action: {active.get('recommended_action')}
  Reason: {active.get('reason')}
  Current bottle center: {active.get('current_bottle_center_px')}
  Ideal center: {active.get('ideal_center_px')}
  dx error: {fmt(active.get('dx_error_px'))} px
  dy error: {fmt(active.get('dy_error_px'))} px
  grasp-line y error: {fmt(active.get('grasp_y_error_px'))} px
  depth error: {fmt(active.get('depth_error_mm'))} mm
  Action vector: dx {fmt(active.get('action_dx_px'))} px, dy {fmt(active.get('action_dy_px'))} px, depth {fmt(active.get('action_depth_mm'))} mm

Planned Grasp
  Grasp center px: {plan.get('grasp_center_px')}
  Grasp angle: {fmt(plan.get('grasp_angle_degrees'))} deg
  Body fraction: {fmt(plan.get('body_fraction'))}
  Left target px: {plan.get('left_contact_px')}
  Right target px: {plan.get('right_contact_px')}
  Grip width: {fmt(plan.get('grip_width_px'))} px / {fmt(plan.get('grip_width_mm'))} mm
  Width reachable: {plan.get('width_reachable')}
  Contact model ready: {report.get('contact_model_ready')}
  Contact model ready reasons: {report.get('contact_model_ready_reasons', [])}
  Positioning guidance score: {fmt(plan.get('positioning_guidance_score', plan.get('workspace_reachability_score')))}
  Workspace guidance only: {report.get('workspace_guidance_only')}
  Left/right in preferred guide zone: {plan.get('left_reachable')} / {plan.get('right_reachable')}
  Left positioning guide zone: {workspace_zones.get('left')}
  Right positioning guide zone: {workspace_zones.get('right')}
  Contacts on body: {plan.get('contacts_on_body')}

Candidate Contact Modeling
  Candidate count: {len(candidates)}
  Selected contact pair: {selected_contact}
  Rejected candidates: {len(rejected_candidates)}
  Rejection examples: {rejected_candidates[:3]}

Scores
  Bottle detection: {fmt(scores.get('bottle_detection_confidence'))}
  Alignment: {fmt(scores.get('alignment'))}
  Positioning guidance: {fmt(scores.get('positioning_guidance_score', scores.get('workspace_score')))}
  Target contact quality: {fmt(scores.get('target_contact_quality'))}
  Grip width score: {fmt(scores.get('grip_width_score'))}
  Grip width feasibility: {fmt(scores.get('grip_width_reachability'))}
  Rotation score: {fmt(scores.get('rotation_score', scores.get('rotation_stability')))}
  Slip score: {fmt(scores.get('slip_score', scores.get('slip_risk_score')))}
  Miss score: {fmt(scores.get('miss_score'))}
  Epsilon: {fmt(scores.get('epsilon'))}
  Depth confidence: {fmt(scores.get('depth_confidence'))}

Risks
  Miss risk: {fmt(risks.get('miss_risk'))}/100 where lower is better
  Slip risk: {fmt(risks.get('slip_risk'))}/100 where lower is better
  Rotation risk: {fmt(risks.get('rotation_risk'))}/100 where lower is better

Score Breakdown / Metric Audit
  {chr(10).join(metric_lines) if metric_lines else 'No metric audit available.'}

Epsilon
  Force closure: {epsilon.get('force_closure')}
  Epsilon: {fmt(epsilon.get('epsilon'), 4)}
  Wrench points: {epsilon.get('num_wrench_points')}

Wrench Safety Summary
  Overall wrench safety: {wrench_summary.get('overall_wrench_safety')}
  Push resistance: {fmt(wrench_summary.get('push_resistance_score'))}/100
  Slip resistance: {fmt(wrench_summary.get('slip_resistance_score'))}/100
  Rotation resistance: {fmt(wrench_summary.get('rotation_resistance_score'))}/100
  Weakest disturbance: {wrench_summary.get('weakest_disturbance_direction')}

Hard Failures
  {chr(10).join('- ' + item for item in failures) if failures else 'None'}

Reposition Reasons
  {chr(10).join('- ' + item for item in reposition) if reposition else 'None'}

Warnings
  {chr(10).join('- ' + item for item in warnings) if warnings else 'None'}"""

        has_fails = bool(failures)
        has_warnings = bool(warnings)
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        border = "#b91c1c" if has_fails else ("#b45309" if has_warnings else "#1b5e20")
        bg = "#fdeaea" if has_fails else ("#fff8ed" if has_warnings else "#f4fbf5")
        frame.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 2px solid {border}; border-radius: 6px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet("font-size: 16px; color: #222; font-family: monospace;")
        layout.addWidget(label)
        return frame

    def state_history_panel(self):
        history = self.load_json("output/stage_history.json", [])
        if not isinstance(history, list):
            history = []
        lines = []
        for index, item in enumerate(history[-12:], start=max(1, len(history) - 11)):
            lines.append(
                f"{index}. {item.get('from_state')} -> {item.get('to_state')} | "
                f"{item.get('stage')} | {item.get('stage_decision')} | "
                f"{item.get('recommended_action')} | score {item.get('safety_score')}"
            )
            reasons = item.get("reasons") or []
            if reasons:
                lines.append(f"   reason: {reasons[0]}")
        text = "\n".join(lines) if lines else "No realtime state history yet."
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background: #f8fafc; border: 1px solid #94a3b8; border-radius: 6px; }"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet("font-size: 15px; color: #222; font-family: monospace;")
        layout.addWidget(label)
        return frame

    def multiview_report_panel(self, report):
        def fmt(value, digits=2):
            if value is None:
                return "unknown"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        results = report.get("view_results", [])
        lines = [
            "Multi-View Report",
            "",
            f"Views evaluated: {report.get('views_evaluated', 0)}",
            f"Selected best view: {report.get('selected_best_view')}",
            f"Selection reason: {report.get('selection_reason')}",
            f"Ranking order: {report.get('ranking_order', [])}",
            "",
            "View Results",
        ]
        for item in results:
            marker = "*" if item.get("view_id") == report.get("selected_best_view") else "-"
            lines.append(
                f"{marker} {item.get('view_id')}: {item.get('decision')} | "
                f"score {fmt(item.get('safety_score'))} | eps {fmt(item.get('epsilon_value'), 4)} | "
                f"closure {item.get('force_closure')} | depth {fmt(item.get('depth_confidence'))} | "
                f"align {fmt(item.get('alignment_error_px'))} px"
            )
            failures = item.get("failure_reasons") or []
            if failures:
                lines.append(f"  fail: {failures[0]}")
        lines.extend([
            "",
            "Future Robot Offset Placeholders",
            "center, left, right, closer, higher, lower",
            report.get("future_robot_motion", "No robot motion commands are sent."),
        ])
        text = "\n".join(lines)
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background: #f8fafc; border: 1px solid #2563eb; border-radius: 6px; }"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet("font-size: 15px; color: #111827; font-family: monospace;")
        layout.addWidget(label)
        return frame

    def wrench_safety_summary_panel(self):
        report = self.load_json("output/pregrasp_safety_report.json", {})
        summary = report.get("wrench_safety_summary") if report.get("pregrasp_decision") and self.is_current_pregrasp_report() else None
        if not summary:
            summary = self.load_json("output/wrench_safety_summary.json", {})

        def fmt(value, digits=2):
            if value is None:
                return "unknown"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        status = summary.get("overall_wrench_safety", "FAIL")
        force_closure = bool(summary.get("force_closure", False))
        weak = summary.get("weakest_disturbance_direction")
        if isinstance(weak, list):
            weak_text = "[" + ", ".join(f"{float(v):.3f}" for v in weak[:3]) + "]"
        else:
            weak_text = str(weak) if weak is not None else "unknown"
        bg = "#e7f5ea" if status == "PASS" else "#fdeaea"
        fg = "#1b5e20" if status == "PASS" else "#b91c1c"
        text = f"""Wrench-Space Safety: {status}

Force closure: {force_closure}
Epsilon margin: {fmt(summary.get('epsilon_margin'), 4)}
Weakest disturbance: {weak_text}
Weakest interpretation: {summary.get('weakest_direction_interpretation', 'unknown')}

Practical Scores
  Force balance / push: {fmt(summary.get('push_resistance_score'))}/100
  Slip resistance: {fmt(summary.get('slip_resistance_score'))}/100
  Rotation resistance: {fmt(summary.get('rotation_resistance_score'))}/100
  Epsilon margin score: {fmt(summary.get('epsilon_margin_score'))}/100

Explanation
  {summary.get('explanation', 'The system computes epsilon in 3D planar wrench space [Fx, Fy, tau]. The summary converts this into practical resistance scores for pushing, slipping, and rotation.')}"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 2px solid {fg}; border-radius: 6px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setStyleSheet(f"font-size: 16px; color: {fg}; font-family: monospace; font-weight: bold;")
        layout.addWidget(label)
        return frame

    def wrench_analysis_panel(self):
        pregrasp = self.load_json("output/pregrasp_safety_report.json", {})
        epsilon = pregrasp.get("epsilon_result") if pregrasp.get("pregrasp_decision") and self.is_current_pregrasp_report() else None
        if not epsilon:
            epsilon = self.load_json("output/epsilon_result.json", {})
        force_closure = bool(epsilon.get("force_closure", False))
        epsilon_value = epsilon.get("epsilon", 0) or 0
        num_wrenches = epsilon.get("num_wrench_points", epsilon.get("wrench_points_count", 0)) or 0
        mu = epsilon.get("mu", epsilon.get("friction_coefficient", 0.5))
        weak_direction = epsilon.get("weakest_direction")
        closest_hull_point = epsilon.get("closest_hull_point")
        closest_face = epsilon.get("closest_face_distance")
        origin_text = "inside" if force_closure else "outside"
        note = (
            "Origin inside wrench hull. Grasp has planar force closure."
            if force_closure
            else "Origin outside 3D wrench hull. Grasp cannot resist all planar disturbances."
        )
        bg = "#e7f5ea" if force_closure else "#fdeaea"
        fg = "#1b5e20" if force_closure else "#b91c1c"

        if isinstance(weak_direction, list):
            weak_text = "[" + ", ".join(f"{float(v):.3f}" for v in weak_direction[:3]) + "]"
        else:
            weak_text = str(weak_direction) if weak_direction is not None else "unknown"
        if isinstance(closest_hull_point, list):
            touch_text = "[" + ", ".join(f"{float(v):.3f}" for v in closest_hull_point[:3]) + "]"
        else:
            touch_text = str(closest_hull_point) if closest_hull_point is not None else "unknown"
        closest_text = f"{float(closest_face):.4f}" if closest_face is not None else "unknown"

        text = f"""Wrench-Space Analysis

{note}

Epsilon value: {epsilon_value:.4f}
Force closure: {force_closure}
Origin: {origin_text} hull
Weak direction: {weak_text}
Closest hull point: {touch_text}
Closest hull face distance: {closest_text}
Friction coefficient: {float(mu):.2f}
Wrench points: {num_wrenches}

Can the grasp resist force in different directions? {"Yes" if force_closure else "No"}
Can it resist torque? {"Yes" if force_closure else "Not in all planar directions"}
Is the origin inside the wrench hull? {"Yes" if force_closure else "No"}
How large is the epsilon radius? {epsilon_value:.4f}
What is the weakest disturbance direction? {weak_text}

The green circle is only a 2D projection. The green marker is the projected closest point from the origin to the closest 3D hull face."""

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 2px solid {fg}; border-radius: 6px; }}"
            "QLabel { border: none; }"
        )
        layout = QVBoxLayout(frame)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setStyleSheet(f"font-size: 17px; color: {fg}; font-weight: bold;")
        layout.addWidget(label)
        return frame

    def load_json(self, path, default):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default

    def is_current_pregrasp_report(self):
        report_path = "output/pregrasp_safety_report.json"
        epsilon_path = "output/epsilon_result.json"
        if not os.path.exists(report_path):
            return False
        if not os.path.exists(epsilon_path):
            return True
        return os.path.getmtime(report_path) >= os.path.getmtime(epsilon_path)

    def gripper_card_text(self, side):
        if not side:
            return "failed\nconf 0.00"
        mode = side.get("gripper_detection_mode", "unknown")
        confidence = side.get("gripper_confidence") or 0
        valid = "valid" if side.get("contact_valid") else "invalid"
        return f"{mode}\nconf {confidence:.2f}\n{valid}"

    def side_state(self, side):
        if not side:
            return "bad"
        if not side.get("contact_valid"):
            return "bad"
        confidence = side.get("gripper_confidence") or 0
        return "good" if confidence >= 0.65 else "warn"

    def score_text(self, epsilon):
        # The detailed final score lives in quality_matrix.png; this card uses epsilon when JSON lacks the aggregate score.
        epsilon_value = epsilon.get("epsilon", 0) or 0
        return f"eps {epsilon_value:.3f}"

    def quality_from_score(self, epsilon):
        if epsilon.get("force_closure") and (epsilon.get("epsilon", 0) or 0) > 0.08:
            return "Good"
        if epsilon.get("force_closure"):
            return "Medium"
        return "Poor"

    def quality_state(self, epsilon):
        quality = self.quality_from_score(epsilon)
        return "good" if quality == "Good" else ("warn" if quality == "Medium" else "bad")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraspGUI()
    window.showMaximized()
    sys.exit(app.exec_())
