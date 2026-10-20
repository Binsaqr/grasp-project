import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bottle_pose import detect_bottle_pose  # noqa: E402
from bottle_visibility import check_full_bottle_visibility  # noqa: E402
from depth_utils import (  # noqa: E402
    DEPTH_BILATERAL_ENABLED,
    DEPTH_MEDIAN_KERNEL,
    DEPTH_PATCH_SIZE,
    DEPTH_VALID_MAX_MM,
    DEPTH_VALID_MIN_MM,
    USE_DEPTH_SMOOTHING,
    depth_stats as compute_depth_stats,
    smooth_depth_pipeline,
    valid_depth_mask,
)
from epsilon_quality import (  # noqa: E402
    compute_epsilon_quality,
    save_all_wrench_space_projections,
    save_epsilon_result_json,
)
from gripper_workspace import WORKSPACE_CONFIG_SOURCE  # noqa: E402
from pregrasp_action import recommend_pregrasp_action  # noqa: E402
from pregrasp_planner import (  # noqa: E402
    GRIP_WIDTH_LIMITS_SOURCE,
    GRIP_WIDTH_MM_RANGE,
    GRIP_WIDTH_PX_RANGE,
    plan_pregrasp,
)
from pregrasp_safety import evaluate_pregrasp_safety  # noqa: E402
from wrench_safety import compute_wrench_safety_summary, save_wrench_safety_summary  # noqa: E402


RGB_PATH = PROJECT_ROOT / "output" / "captured_frame.png"
DEPTH_PATH = PROJECT_ROOT / "output" / "depth_frame.png"
INTRINSICS_PATH = PROJECT_ROOT / "output" / "camera_intrinsics.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
CURRENT_RUN_ID = None
GUI_IMAGE_FILES = [
    "scene.png",
    "segmentation.png",
    "best_grasp_overlay.png",
    "bottle_grasp_debug.png",
    "dashboard_summary.png",
    "wrench_debug_dashboard.png",
    "wrench_safety_summary.png",
    "wrench_space_fx_fy.png",
    "wrench_space_fx_tau.png",
    "wrench_space_fy_tau.png",
    "mask_cleanup_debug.png",
    "depth_colormap.png",
    "depth_raw_colormap.png",
    "depth_smooth_colormap.png",
    "depth_noise_debug.png",
    "rgbd_contact_validation_debug.png",
    "multiview_summary.png",
]
LEGACY_DEBUG_IMAGE_FILES = [
    "object_mask.png",
    "contact_debug.png",
    "contact_points_debug.png",
    "gripper_debug.png",
    "left_gripper_debug.png",
    "right_gripper_debug.png",
    "quality_matrix.png",
    "contact_validation_debug.png",
    "grasp_contacts_debug.png",
    "object_detection_debug.png",
]
FRICTION_COEFFICIENT = 0.5
READY_DECISION = "READY TO CLOSE"
ADJUST_DECISION = "ADJUST POSITION"
BLOCK_DECISION = "DO NOT CLOSE"
LEGACY_BLOCK_DECISION = "DO NOT LIFT"
EXPECTED_BOTTLE = {
    "diameter_mm_range": [45.0, 85.0],
    "aspect_ratio_range": [1.5, 7.5],
}

PIPELINE_STAGES = [
    ("Input", "Load raw upright RGB, aligned depth, and RealSense intrinsics from output/."),
    ("Segmentation", "Find the known bottle with RGB color cues plus RGB-D foreground candidates."),
    ("Mask cleanup", "Use morphology and depth-band filtering; cleaned bottle/body masks are source of truth."),
    ("Pose", "Estimate bbox, center, depth, main axis, body region, and mask/depth confidence."),
    ("Full bottle visibility", "Gate contact modeling until the full bottle, margins, and reliable depth are visible in raw camera coordinates."),
    ("Contact modeling", "Sample body contact lines, reject cap/bottom/edge candidates, and rank valid left/right planned contact pairs."),
    ("Pre-grasp planning", "Select the best body grasp line perpendicular to the bottle axis and expose it as the planned grasp."),
    ("Positioning guidance", "Compare planned contacts with preferred camera-frame guide zones; this is not a hard gripper reach gate."),
    ("Epsilon", "Run unchanged wrench-space epsilon on the selected planned contact pair."),
    ("Safety", "Combine detection, alignment, depth, reachability, balance, risks, and epsilon into the pre-lift decision."),
    ("Outputs", "Save JSON report, overlays, mask/depth debug views, summary image, and wrench plots for the GUI."),
]


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def start_current_run():
    global CURRENT_RUN_ID
    CURRENT_RUN_ID = new_run_id()
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename in GUI_IMAGE_FILES + LEGACY_DEBUG_IMAGE_FILES:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()
    manifest = {
        "run_id": CURRENT_RUN_ID,
        "mode": "bottle_pregrasp",
        "generated_images": [],
        "cleared_images": GUI_IMAGE_FILES + LEGACY_DEBUG_IMAGE_FILES,
    }
    with open(OUTPUT_DIR / "current_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    return CURRENT_RUN_ID


def finalize_current_run(report, generated_images):
    generated = []
    for filename in generated_images:
        if (OUTPUT_DIR / filename).exists():
            generated.append(filename)
    report["run_id"] = CURRENT_RUN_ID
    report["generated_images"] = generated
    report["current_run_manifest"] = {
        "run_id": CURRENT_RUN_ID,
        "generated_images": generated,
    }
    write_report(report)
    with open(OUTPUT_DIR / "current_run_manifest.json", "w") as f:
        json.dump(report["current_run_manifest"], f, indent=4)
    return report


def print_pipeline_audit():
    print("Bottle pre-grasp pipeline stages:")
    for index, (name, description) in enumerate(PIPELINE_STAGES, start=1):
        print(f"{index}. {name}: {description}")


def load_inputs():
    rgb_bgr = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {RGB_PATH}")
    depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {DEPTH_PATH}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    intrinsics = None
    if INTRINSICS_PATH.exists():
        with open(INTRINSICS_PATH, "r") as f:
            intrinsics = json.load(f)
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), depth, intrinsics


def build_depth_context(depth_raw, mask=None):
    depth_smooth = smooth_depth_pipeline(depth_raw)
    raw_stats = compute_depth_stats(depth_raw, mask) if mask is not None else global_depth_stats(depth_raw)
    smooth_stats = compute_depth_stats(depth_smooth, mask) if mask is not None else global_depth_stats(depth_smooth)
    return {
        "depth_raw": depth_raw,
        "depth_smooth": depth_smooth,
        "report": {
            "depth_smoothing_enabled": bool(USE_DEPTH_SMOOTHING),
            "depth_smoothing_pipeline": {
                "median_kernel": int(DEPTH_MEDIAN_KERNEL),
                "bilateral_enabled": bool(DEPTH_BILATERAL_ENABLED),
                "patch_size_px": int(DEPTH_PATCH_SIZE),
                "valid_min_mm": int(DEPTH_VALID_MIN_MM),
                "valid_max_mm": int(DEPTH_VALID_MAX_MM),
                "notes": "Raw depth is preserved; smoothed depth is used for visualization, mask cleanup support, and robust patch measurements.",
            },
            "raw_depth": raw_stats,
            "smoothed_depth": smooth_stats,
        },
    }


def global_depth_stats(depth):
    mask = np.ones(depth.shape[:2], dtype=np.uint8)
    return compute_depth_stats(depth, mask)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if k not in {"mask", "raw_mask", "cleaned_mask", "body_mask", "contour", "body_contour"}}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def line_endpoints(center, direction, length):
    c = np.asarray(center, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-9)
    return c - d * length, c + d * length


def build_calibration_status():
    warnings = []
    if WORKSPACE_CONFIG_SOURCE == "default":
        warnings.append("Positioning guidance zones are using default camera-frame values; calibrate before using them as robot motion targets.")
    if GRIP_WIDTH_LIMITS_SOURCE == "default":
        warnings.append("Grip width limits are using default values; replace with measured jaw travel for the real end-effector.")
    return {
        "workspace_config_source": WORKSPACE_CONFIG_SOURCE,
        "grip_width_limits_source": GRIP_WIDTH_LIMITS_SOURCE,
        "grip_width_mm_range": list(GRIP_WIDTH_MM_RANGE),
        "grip_width_px_range_when_depth_unavailable": list(GRIP_WIDTH_PX_RANGE),
        "warnings": warnings,
    }


def build_known_bottle_validation(pose, plan, active_positioning):
    bbox = pose.get("bbox_xywh", [None, None, None, None])
    _, _, bbox_w, bbox_h = bbox
    width_px = float(plan.get("grip_width_px", 0.0) or 0.0)
    width_mm = plan.get("grip_width_mm")
    depth_mm = pose.get("depth_median_mm")
    measured_aspect = None
    expected_height_px_range = None
    if bbox_w and bbox_h:
        measured_aspect = float(max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1))
        short_side = float(max(min(bbox_w, bbox_h), 1))
        expected_height_px_range = [
            short_side * EXPECTED_BOTTLE["aspect_ratio_range"][0],
            short_side * EXPECTED_BOTTLE["aspect_ratio_range"][1],
        ]
    depth_range = (
        active_positioning.get("ideal_zone", {})
        .get("acceptable_depth_range_mm", [None, None])
    )
    width_in_expected_range = None
    if width_mm is not None:
        width_in_expected_range = bool(
            EXPECTED_BOTTLE["diameter_mm_range"][0]
            <= float(width_mm)
            <= EXPECTED_BOTTLE["diameter_mm_range"][1]
        )
    aspect_in_expected_range = None
    if measured_aspect is not None:
        aspect_in_expected_range = bool(
            EXPECTED_BOTTLE["aspect_ratio_range"][0]
            <= measured_aspect
            <= EXPECTED_BOTTLE["aspect_ratio_range"][1]
        )
    depth_in_expected_range = None
    if depth_mm is not None and depth_range[0] is not None:
        depth_in_expected_range = bool(float(depth_range[0]) <= float(depth_mm) <= float(depth_range[1]))
    return {
        "expected": {
            "bottle_diameter_mm_range": EXPECTED_BOTTLE["diameter_mm_range"],
            "bbox_aspect_ratio_range": EXPECTED_BOTTLE["aspect_ratio_range"],
            "height_px_range_for_measured_short_side": expected_height_px_range,
            "depth_mm_range": depth_range,
            "notes": "Pixel height depends on distance; compare aspect ratio and depth-normalized grip width for the known bottle.",
        },
        "measured": {
            "bbox_width_px": bbox_w,
            "bbox_height_px": bbox_h,
            "bbox_aspect_ratio": measured_aspect,
            "planned_grip_width_px": width_px,
            "planned_grip_width_mm": width_mm,
            "bottle_depth_median_mm": depth_mm,
            "candidate_score_breakdown": pose.get("candidate_score_breakdown", {}),
        },
        "checks": {
            "width_matches_known_bottle": width_in_expected_range,
            "aspect_matches_known_bottle": aspect_in_expected_range,
            "depth_in_pregrasp_range": depth_in_expected_range,
        },
    }


def save_failure(rgb, message):
    OUTPUT_DIR.mkdir(exist_ok=True)
    image_shape = rgb.shape if rgb is not None else (480, 848, 3)
    active_positioning = recommend_pregrasp_action(
        None,
        None,
        {"decision": BLOCK_DECISION, "failure_reasons": [message]},
        image_shape,
    )
    calibration_status = build_calibration_status()
    report = {
        "mode": "bottle_pregrasp",
        "run_id": CURRENT_RUN_ID,
        "generated_images": [],
        "bottle_detected": False,
        "decision": BLOCK_DECISION,
        "pregrasp_decision": BLOCK_DECISION,
        "legacy_decision": LEGACY_BLOCK_DECISION,
        "final_score": 0.0,
        "safety_score": 0.0,
        "raw_weighted_score": 0.0,
        "decision_score": 0.0,
        "capped_decision_score": 0.0,
        "cap_reason": "hard_gate_failure",
        "decision_stage": "pre_close",
        "active_positioning": active_positioning,
        "failure_reasons": [message],
        "reposition_reasons": [],
        "warnings": calibration_status["warnings"],
        "calibration_status": calibration_status,
        "known_bottle_validation": {},
        "metric_audit": [],
        "hard_gates": {"passed": False, "failed": [message]},
        "risk_scores": {},
        "scores": {},
        "score_weights": {},
        "epsilon_result": skipped_epsilon_result(message),
        "epsilon_audit": build_epsilon_audit(skipped_epsilon_result(message)),
        "full_bottle_visible": False,
        "visibility_score": 0.0,
        "border_margins": {},
        "visibility": {
            "full_bottle_visible": False,
            "visibility_score": 0.0,
            "missing_region_reason": [message],
            "recommended_view_action": "SEARCH",
            "view_status": "SEARCH_BOTTLE",
        },
        "recommended_view_action": "SEARCH",
        "contact_model_ready": False,
        "candidate_contacts": [],
        "selected_contact_pair": None,
        "rejected_candidates": [],
    }
    with open(OUTPUT_DIR / "pregrasp_safety_report.json", "w") as f:
        json.dump(report, f, indent=4)
    with open(OUTPUT_DIR / "prelift_safety_report.json", "w") as f:
        json.dump(report, f, indent=4)
    write_scoring_audit_report(report)
    save_placeholder_wrench_outputs(report["epsilon_result"], message)

    fig, ax = plt.subplots(figsize=(10, 6))
    if rgb is not None:
        ax.imshow(rgb)
    ax.axis("off")
    ax.text(0.04, 0.82, BLOCK_DECISION, transform=ax.transAxes, fontsize=22, weight="bold", color="red")
    ax.text(0.04, 0.70, message, transform=ax.transAxes, fontsize=13, color="white" if rgb is not None else "black",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"})
    fig.savefig(OUTPUT_DIR / "bottle_grasp_debug.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "best_grasp_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 2.8))
    plt.axis("off")
    plt.text(0.04, 0.68, f"Pre-Grasp Decision: {BLOCK_DECISION}", fontsize=18, weight="bold", color="#b91c1c")
    plt.text(0.04, 0.38, message, fontsize=13, color="#222")
    fig.savefig(OUTPUT_DIR / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    plt.axis("off")
    plt.text(0.05, 0.70, "No Wrench Analysis", fontsize=20, weight="bold", color="#b91c1c")
    plt.text(
        0.05,
        0.52,
        "A valid planned pre-grasp contact pair was not available, so epsilon/wrench-space analysis was not run.",
        fontsize=13,
        color="#222",
        wrap=True,
    )
    plt.text(0.05, 0.36, message, fontsize=12, color="#444", wrap=True)
    fig.savefig(OUTPUT_DIR / "wrench_debug_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    finalize_current_run(report, [
        "bottle_grasp_debug.png",
        "best_grasp_overlay.png",
        "dashboard_summary.png",
        "wrench_debug_dashboard.png",
        "wrench_safety_summary.png",
        "wrench_space_fx_fy.png",
        "wrench_space_fx_tau.png",
        "wrench_space_fy_tau.png",
    ])
    print(f"Pre-grasp safety failed: {message}")


def skipped_epsilon_result(reason):
    return {
        "epsilon": 0.0,
        "force_closure": False,
        "weakest_direction": None,
        "closest_face_distance": None,
        "closest_hull_point": None,
        "num_wrench_points": 0,
        "error": reason,
        "skipped": True,
    }


def save_placeholder_wrench_outputs(epsilon_result, reason):
    with open(OUTPUT_DIR / "epsilon_result.json", "w") as f:
        json.dump(json_safe(epsilon_result), f, indent=4)
    with open(OUTPUT_DIR / "bottle_epsilon_result.json", "w") as f:
        json.dump(json_safe(epsilon_result), f, indent=4)
    summary = compute_wrench_safety_summary(epsilon_result)
    summary["explanation"] = f"{summary.get('explanation', '')} Epsilon was skipped: {reason}"
    save_wrench_safety_summary(summary, OUTPUT_DIR)
    for filename, title in [
        ("wrench_space_fx_fy.png", "Wrench Fx/Fy"),
        ("wrench_space_fx_tau.png", "Wrench Fx/Tau"),
        ("wrench_space_fy_tau.png", "Wrench Fy/Tau"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.64, title, ha="center", va="center", fontsize=16, weight="bold")
        ax.text(0.5, 0.42, "Epsilon skipped", ha="center", va="center", fontsize=13, color="#b91c1c")
        ax.text(0.5, 0.28, reason, ha="center", va="center", fontsize=10, color="#444", wrap=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def build_visibility_gate_report(pose, visibility):
    calibration_status = build_calibration_status()
    reason = "Full bottle not visible. Contact modeling disabled until view improves."
    failure_reasons = [reason] + list(visibility.get("missing_region_reason", []))
    active_positioning = {
        "state": "POSITIONING",
        "recommended_action": visibility.get("recommended_view_action", "ADJUST_VIEW"),
        "reason": "; ".join(visibility.get("missing_region_reason", [])) or reason,
        "current_bottle_center_px": pose.get("center_px"),
        "ideal_center_px": [pose["cleaned_mask"].shape[1] / 2.0, pose["cleaned_mask"].shape[0] / 2.0],
        "dx_error_px": visibility.get("center_error_px", {}).get("x"),
        "dy_error_px": visibility.get("center_error_px", {}).get("y"),
        "grasp_y_error_px": None,
        "depth_error_mm": None,
        "action_dx_px": visibility.get("center_error_px", {}).get("x"),
        "action_dy_px": visibility.get("center_error_px", {}).get("y"),
        "action_depth_mm": None,
        "ideal_zone": {
            "acceptable_depth_range_mm": [450.0, 1100.0],
        },
    }
    epsilon_result = skipped_epsilon_result(reason)
    score = min(float(visibility.get("visibility_score", 0.0) or 0.0), 59.0)
    report = {
        "mode": "bottle_pregrasp",
        "run_id": CURRENT_RUN_ID,
        "generated_images": [],
        "decision": "ADJUST_VIEW",
        "pregrasp_decision": "ADJUST_VIEW",
        "legacy_decision": ADJUST_DECISION,
        "decision_stage": "full_bottle_view_check",
        "final_score": score,
        "safety_score": score,
        "raw_weighted_score": float(visibility.get("visibility_score", 0.0) or 0.0),
        "decision_score": score,
        "capped_decision_score": score,
        "cap_reason": "full_bottle_visibility_gate",
        "decision_policy": {
            "hard_gate": "full_bottle_visible must pass before contact modeling, epsilon evaluation, or close decision",
            "next_allowed_stage": "CONTACT_MODELING" if visibility.get("full_bottle_visible") else "FULL_BOTTLE_VIEW_CHECK",
        },
        "active_positioning": active_positioning,
        "failure_reasons": failure_reasons,
        "reposition_reasons": visibility.get("missing_region_reason", []),
        "warnings": calibration_status["warnings"],
        "calibration_status": calibration_status,
        "known_bottle_validation": {
            "expected": {
                "bottle_diameter_mm_range": EXPECTED_BOTTLE["diameter_mm_range"],
                "bbox_aspect_ratio_range": EXPECTED_BOTTLE["aspect_ratio_range"],
            },
            "measured": {
                "bbox_xywh": pose.get("bbox_xywh"),
                "bottle_depth_median_mm": pose.get("depth_median_mm"),
                "candidate_score_breakdown": pose.get("candidate_score_breakdown", {}),
            },
            "checks": {"full_bottle_visible": False},
        },
        "bottle_detected": True,
        "bottle_geometry": json_safe(pose),
        "planned_contacts": {},
        "depth_stats": {
            "bottle_depth_median_mm": pose.get("depth_median_mm"),
            "bottle_depth_mean_mm": pose.get("depth_mean_mm"),
            "bottle_depth_std_mm": pose.get("depth_std_mm"),
            "bottle_depth_confidence": pose.get("depth_confidence"),
        },
        "alignment_error_px": visibility.get("center_error_px", {}).get("x", 0.0),
        "scores": {
            "visibility_score": float(visibility.get("visibility_score", 0.0) or 0.0),
            "depth_confidence": float(pose.get("depth_confidence", 0.0) or 0.0) * 100.0,
        },
        "risk_scores": {},
        "metric_audit": [],
        "hard_gates": {
            "passed": False,
            "status": {
                "bottle_detected": True,
                "full_bottle_visible": False,
                "contact_model_ready": False,
            },
            "failed": failure_reasons,
        },
        "score_weights": {},
        "epsilon_result": epsilon_result,
        "epsilon_audit": build_epsilon_audit(epsilon_result),
        "full_bottle_visible": False,
        "visibility_score": float(visibility.get("visibility_score", 0.0) or 0.0),
        "border_margins": visibility.get("border_margins", {}),
        "visibility": visibility,
        "recommended_view_action": visibility.get("recommended_view_action"),
        "contact_model_ready": False,
        "candidate_contacts": [],
        "selected_contact_pair": None,
        "rejected_candidates": [],
    }
    return report


def save_scene_and_segmentation(rgb, pose):
    cv2.imwrite(str(OUTPUT_DIR / "scene.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb)
    overlay = np.zeros((*pose["cleaned_mask"].shape, 4), dtype=float)
    overlay[pose["cleaned_mask"] > 0] = [0.1, 0.7, 1.0, 0.35]
    body = pose.get("body_mask")
    if body is not None:
        overlay[body > 0] = [0.0, 0.9, 0.25, 0.25]
    ax.imshow(overlay)
    x, y, w, h = pose["bbox_xywh"]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="yellow", linewidth=2))
    box = np.asarray(pose.get("oriented_bbox", []), dtype=float)
    if box.shape == (4, 2):
        closed = np.vstack([box, box[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="orange", linewidth=2)
    ax.set_title("Bottle Detection / Cleaned RGB-D Mask")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "segmentation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask_debug_images(pose):
    raw = pose.get("raw_mask")
    cleaned = pose.get("cleaned_mask")
    body = pose.get("body_mask")
    masks = [
        ("Raw candidate mask", raw, OUTPUT_DIR / "bottle_raw_mask.png"),
        ("Cleaned depth-filtered mask", cleaned, OUTPUT_DIR / "bottle_cleaned_mask.png"),
        ("Body-only mask", body, OUTPUT_DIR / "bottle_body_mask.png"),
    ]
    for _, mask, path in masks:
        if mask is not None:
            cv2.imwrite(str(path), (mask.astype(np.uint8) > 0).astype(np.uint8) * 255)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, mask, _) in zip(axes, masks):
        ax.set_title(title)
        if mask is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", color="red")
        else:
            ax.imshow(mask, cmap="gray")
        ax.axis("off")
    fig.suptitle("Bottle Mask Cleanup Debug")
    fig.savefig(OUTPUT_DIR / "mask_cleanup_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pregrasp_overlay(rgb, pose, plan, report):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb)
    mask = pose["cleaned_mask"]
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = [0.1, 0.7, 1.0, 0.32]
    ax.imshow(overlay)

    x, y, w, h = pose["bbox_xywh"]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="yellow", linewidth=2.5))
    visibility = report.get("visibility", {})
    margins = visibility.get("border_margins", {})
    required_margin = float(visibility.get("required_margin_px", 0.0) or 0.0)
    if required_margin > 0:
        img_h, img_w = rgb.shape[:2]
        ax.add_patch(
            plt.Rectangle(
                (required_margin, required_margin),
                max(img_w - 2 * required_margin, 1),
                max(img_h - 2 * required_margin, 1),
                fill=False,
                edgecolor="orange",
                linestyle="--",
                linewidth=1.8,
                label="full-bottle margin",
            )
        )
    active = report.get("active_positioning", {})
    ideal = active.get("ideal_zone", {})
    center_region = ideal.get("ideal_center_region")
    grasp_region = ideal.get("ideal_grasp_height_region")
    if center_region:
        ax.add_patch(
            plt.Rectangle(
                (center_region["x_min"], center_region["y_min"]),
                center_region["x_max"] - center_region["x_min"],
                center_region["y_max"] - center_region["y_min"],
                fill=True,
                color="gold",
                alpha=0.16,
                label="ideal bottle center",
                linewidth=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (center_region["x_min"], center_region["y_min"]),
                center_region["x_max"] - center_region["x_min"],
                center_region["y_max"] - center_region["y_min"],
                fill=False,
                edgecolor="gold",
                linestyle="-.",
                linewidth=2,
            )
        )
    if grasp_region:
        ax.axhspan(
            grasp_region["y_min"],
            grasp_region["y_max"],
            color="orange",
            alpha=0.10,
            label="ideal grasp height",
        )
    workspace = plan.get("gripper_workspace", {}).get("zones", {})
    for side, color in [("left", "cyan"), ("right", "magenta")]:
        zone = workspace.get(side)
        if zone:
            ax.add_patch(
                plt.Rectangle(
                    (zone["x_min"], zone["y_min"]),
                    zone["x_max"] - zone["x_min"],
                    zone["y_max"] - zone["y_min"],
                    fill=True,
                    color=color,
                    alpha=0.10,
                    label=f"{side} positioning guide",
                    linewidth=0,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (zone["x_min"], zone["y_min"]),
                    zone["x_max"] - zone["x_min"],
                    zone["y_max"] - zone["y_min"],
                    fill=False,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=2,
                )
            )
    center = np.asarray(pose["center_px"], dtype=float)
    axis = np.asarray(pose["axis_vector"], dtype=float)
    axis_start, axis_end = line_endpoints(center, axis, max(w, h) * 0.75)
    ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], color="lime", linewidth=2.5, label="bottle axis")
    ax.axvline(rgb.shape[1] / 2.0, color="white", linestyle="--", linewidth=1.5, label="image/gripper center")
    contact_band = plan.get("valid_contact_band", {})
    band_min = contact_band.get("body_fraction_min")
    band_max = contact_band.get("body_fraction_max")
    if band_min is not None and band_max is not None:
        mask_pts_y, mask_pts_x = np.where(mask > 0)
        if mask_pts_x.size:
            pts = np.column_stack([mask_pts_x, mask_pts_y]).astype(float)
            projections = pts @ axis
            min_proj, max_proj = float(np.min(projections)), float(np.max(projections))
            axis_center_projection = float(center @ axis)
            grasp_direction_for_band = np.asarray(plan.get("grasp_direction", [1.0, 0.0]), dtype=float)
            for fraction in [float(band_min), float(band_max)]:
                target_projection = min_proj + fraction * (max_proj - min_proj)
                band_center = center + axis * (target_projection - axis_center_projection)
                p1, p2 = line_endpoints(band_center, grasp_direction_for_band, max(w, h) * 0.42)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="orange", linestyle=":", linewidth=2.0, label="valid body band" if fraction == float(band_min) else None)
    for candidate in plan.get("candidate_contacts", []):
        line_left = candidate.get("line_left_px")
        line_right = candidate.get("line_right_px")
        if line_left and line_right:
            color = "#94a3b8" if candidate.get("rejected") else "#38bdf8"
            ax.plot(
                [line_left[0], line_right[0]],
                [line_left[1], line_right[1]],
                color=color,
                linewidth=1.3,
                alpha=0.45,
                label="candidate contact lines" if candidate == plan.get("candidate_contacts", [None])[0] else None,
            )
    ideal_center = active.get("ideal_center_px")
    current_center = active.get("current_bottle_center_px")
    if ideal_center and current_center:
        ax.scatter([ideal_center[0]], [ideal_center[1]], c=["gold"], s=130, marker="x", linewidths=3, zorder=6)
        ax.annotate(
            "",
            xy=ideal_center,
            xytext=current_center,
            arrowprops={"arrowstyle": "->", "color": "gold", "lw": 3},
        )

    left = np.asarray(plan["left_contact_px"], dtype=float)
    right = np.asarray(plan["right_contact_px"], dtype=float)
    ax.plot([left[0], right[0]], [left[1], right[1]], color="red", linewidth=2.8, label="planned grasp line")
    ax.scatter([left[0], right[0]], [left[1], right[1]], c=["red", "blue"], s=145, edgecolors="black", zorder=5)
    ax.annotate("L target", xy=left, xytext=(-62, -24), textcoords="offset points", color="white",
                bbox={"facecolor": "red", "alpha": 0.75, "edgecolor": "none"})
    ax.annotate("R target", xy=right, xytext=(10, -24), textcoords="offset points", color="white",
                bbox={"facecolor": "blue", "alpha": 0.75, "edgecolor": "none"})

    text = (
        f"{active.get('recommended_action', report['decision'])} | {active.get('state', 'POSITIONING')}\n"
        f"{report['decision']}: {report['final_score']:.1f}/100\n"
        f"width: {plan['grip_width_px']:.1f} px"
        + (f" / {plan['grip_width_mm']:.1f} mm" if plan.get("grip_width_mm") is not None else "")
        + f"\nalignment: {report['alignment_error_px']:.1f} px"
        + f"\nfull bottle: {report.get('full_bottle_visible')} | view: {report.get('recommended_view_action')}"
        + f"\ncontact candidates: {len(report.get('candidate_contacts', []))}"
        + f"\nmiss/slip/rot risk: {report['risk_scores']['miss_risk']:.0f}/"
        f"{report['risk_scores']['slip_risk']:.0f}/{report['risk_scores']['rotation_risk']:.0f}"
    )
    if margins:
        text += (
            f"\nmargins L/R/T/B: {margins.get('left', 0):.0f}/"
            f"{margins.get('right', 0):.0f}/"
            f"{margins.get('top', 0):.0f}/"
            f"{margins.get('bottom', 0):.0f} px"
        )
    if report["failure_reasons"]:
        text += "\nunsafe: " + report["failure_reasons"][0]
    elif report["warnings"]:
        text += "\nwarning: " + report["warnings"][0]
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=12, color="white",
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none"})
    ax.set_title("Pre-Grasp Planned Bottle Grasp")
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.savefig(OUTPUT_DIR / "bottle_grasp_debug.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "best_grasp_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_visibility_gate_overlay(rgb, pose, visibility, report):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb)
    mask = pose["cleaned_mask"]
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = [0.1, 0.7, 1.0, 0.32]
    ax.imshow(overlay)

    x, y, w, h = pose["bbox_xywh"]
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="yellow", linewidth=2.5))
    margins = visibility.get("border_margins", {})
    required = float(visibility.get("required_margin_px", 0.0) or 0.0)
    img_h, img_w = rgb.shape[:2]
    ax.add_patch(
        plt.Rectangle(
            (required, required),
            max(img_w - 2 * required, 1),
            max(img_h - 2 * required, 1),
            fill=False,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="required full-bottle margin",
        )
    )
    center = np.asarray(pose["center_px"], dtype=float)
    axis = np.asarray(pose["axis_vector"], dtype=float)
    axis_start, axis_end = line_endpoints(center, axis, max(w, h) * 0.75)
    ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], color="lime", linewidth=2.5, label="bottle axis")
    ax.axvline(img_w / 2.0, color="white", linestyle="--", linewidth=1.5, label="image/gripper center")
    ax.scatter([center[0]], [center[1]], c=["gold"], s=120, marker="x", linewidths=3, zorder=5)
    text = (
        "Bottle not fully visible.\n"
        "Contact modeling disabled until view improves.\n"
        f"Recommended view action: {visibility.get('recommended_view_action', 'ADJUST_VIEW')}\n"
        f"Visibility score: {visibility.get('visibility_score', 0.0):.1f}/100\n"
        f"Margins L/R/T/B: {margins.get('left', 0):.0f}/"
        f"{margins.get('right', 0):.0f}/"
        f"{margins.get('top', 0):.0f}/"
        f"{margins.get('bottom', 0):.0f} px"
    )
    reasons = visibility.get("missing_region_reason", [])
    if reasons:
        text += "\nreason: " + reasons[0]
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=12, color="white",
            bbox={"facecolor": "black", "alpha": 0.76, "edgecolor": "none"})
    ax.set_title("Full Bottle Visibility Gate")
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.savefig(OUTPUT_DIR / "bottle_grasp_debug.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "best_grasp_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_depth_debug(depth_raw, depth_smooth, pose, plan):
    save_depth_colormap(depth_raw, pose, plan, "depth_raw_colormap.png", "Raw Depth Map with Planned Contacts")
    save_depth_colormap(depth_smooth, pose, plan, "depth_smooth_colormap.png", "Smoothed Depth Map with Planned Contacts")
    save_depth_colormap(depth_smooth, pose, plan, "depth_colormap.png", "Smoothed Depth Map with Planned Contacts")

    raw_valid = valid_depth_mask(depth_raw)
    smooth_valid = valid_depth_mask(depth_smooth)
    raw_float = np.where(raw_valid, depth_raw.astype(float), np.nan)
    smooth_float = np.where(smooth_valid, depth_smooth.astype(float), np.nan)
    diff = np.abs(raw_float - smooth_float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, image, title in [
        (axes[0], raw_float, "Raw depth"),
        (axes[1], smooth_float, "Smoothed depth"),
        (axes[2], diff, "Absolute raw-smooth difference"),
    ]:
        im = ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Depth Noise / Smoothing Debug")
    fig.savefig(OUTPUT_DIR / "depth_noise_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    valid = valid_depth_mask(depth_smooth)
    masked = np.where(valid, depth_smooth.astype(float), np.nan)
    left = np.asarray(plan["left_contact_px"], dtype=float)
    right = np.asarray(plan["right_contact_px"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(masked, cmap="viridis")
    ax.imshow(np.ma.masked_where(pose["cleaned_mask"] == 0, pose["cleaned_mask"]), cmap="cool", alpha=0.35)
    ax.scatter([left[0], right[0]], [left[1], right[1]], c=["red", "blue"], s=140, edgecolors="white")
    ax.set_title("RGB-D Planned Contact Debug")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "rgbd_contact_validation_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_depth_colormap(depth, pose, plan, filename, title):
    valid = valid_depth_mask(depth)
    masked = np.where(valid, depth.astype(float), np.nan)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(masked, cmap="viridis")
    if plan is not None:
        left = np.asarray(plan["left_contact_px"], dtype=float)
        right = np.asarray(plan["right_contact_px"], dtype=float)
        ax.scatter([left[0], right[0]], [left[1], right[1]], c=["red", "blue"], s=140, edgecolors="white")
        ax.plot([left[0], right[0]], [left[1], right[1]], color="white", linewidth=2)
    if pose is not None and pose.get("bbox_xywh"):
        x, y, w, h = pose["bbox_xywh"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color="yellow", linewidth=2))
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, label="Depth (mm)")
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_depth_input_debug(depth_raw, depth_smooth, pose=None):
    save_depth_colormap(depth_raw, pose, None, "depth_raw_colormap.png", "Raw Depth Map")
    save_depth_colormap(depth_smooth, pose, None, "depth_smooth_colormap.png", "Smoothed Depth Map")
    save_depth_colormap(depth_smooth, pose, None, "depth_colormap.png", "Smoothed Depth Map")
    raw_valid = valid_depth_mask(depth_raw)
    smooth_valid = valid_depth_mask(depth_smooth)
    raw_float = np.where(raw_valid, depth_raw.astype(float), np.nan)
    smooth_float = np.where(smooth_valid, depth_smooth.astype(float), np.nan)
    diff = np.abs(raw_float - smooth_float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, image, title in [
        (axes[0], raw_float, "Raw depth"),
        (axes[1], smooth_float, "Smoothed depth"),
        (axes[2], diff, "Absolute raw-smooth difference"),
    ]:
        im = ax.imshow(image, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Depth Noise / Smoothing Debug")
    fig.savefig(OUTPUT_DIR / "depth_noise_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dashboard_summary(report):
    scores = report.get("scores", {})
    risks = report.get("risk_scores", {})
    planned = report.get("planned_contacts", {})
    active = report.get("active_positioning", {})
    full_visible = bool(report.get("full_bottle_visible", False))
    contact_ready = bool(report.get("contact_model_ready", bool(planned)))
    wrench_summary = report.get("wrench_safety_summary", {})
    decision = report.get("decision", BLOCK_DECISION)
    final_score = float(report.get("final_score", 0.0) or 0.0)
    raw_score = float(report.get("raw_weighted_score", final_score) or 0.0)
    cards = [
        ("Full Bottle Visible", "Yes" if full_visible else "No", "good" if full_visible else "bad"),
        ("View Status", report.get("visibility", {}).get("view_status", "UNKNOWN"), "good" if full_visible else "warn"),
        ("Recommended View Action", report.get("recommended_view_action", active.get("recommended_action", "UNKNOWN")), "good" if report.get("recommended_view_action") == "VIEW_OK" else "warn"),
        ("Planned Contact Model", "Yes" if contact_ready else "No", "good" if contact_ready else "bad"),
        ("Recommended Action", active.get("recommended_action", "UNKNOWN"), "good" if active.get("state") == "READY_TO_CLOSE" else ("bad" if active.get("state") == "UNSAFE" else "warn")),
        ("Pre-Grasp Decision", decision, "good" if decision == READY_DECISION else ("warn" if decision in {ADJUST_DECISION, "ADJUST_VIEW"} else "bad")),
        ("Raw Score", f"{raw_score:.1f}/100", "good" if raw_score >= 80 else ("warn" if raw_score >= 60 else "bad")),
        ("Decision Score", f"{final_score:.1f}/100", "good" if final_score >= 80 else ("warn" if final_score >= 60 else "bad")),
        ("Cap Reason", report.get("cap_reason", "none"), "warn" if report.get("cap_reason", "none") != "none" else "good"),
        ("Bottle", "Yes" if report.get("bottle_detected") else "No", "good" if scores.get("bottle_detection_confidence", 0) >= 70 or report.get("bottle_detected") else "bad"),
        ("Alignment", f"{float(report.get('alignment_error_px', 0.0) or 0.0):.1f} px", "good" if scores.get("alignment", 0) >= 75 else "warn"),
        ("Grip Plan", f"{float(planned.get('grip_width_px', 0.0) or 0.0):.0f} px", "good" if scores.get("grip_width_reachability", 0) >= 60 else "bad"),
        ("Positioning Guidance", f"{float(planned.get('positioning_guidance_score', planned.get('workspace_reachability_score', 0.0)) or 0.0):.0f}/100", "good" if planned.get('positioning_guidance_score', planned.get('workspace_reachability_score', 0)) >= 75 else "warn"),
        ("Miss Risk", f"{float(risks.get('miss_risk', 100.0) or 0.0):.0f}/100", "good" if risks.get("miss_risk", 100) <= 35 else "warn"),
        ("Slip Risk", f"{float(risks.get('slip_risk', 100.0) or 0.0):.0f}/100", "good" if risks.get("slip_risk", 100) <= 35 else "warn"),
        ("Rotation Risk", f"{float(risks.get('rotation_risk', 100.0) or 0.0):.0f}/100", "good" if risks.get("rotation_risk", 100) <= 35 else "warn"),
        ("Wrench Safety", wrench_summary.get("overall_wrench_safety", "UNKNOWN"), "good" if wrench_summary.get("overall_wrench_safety") == "PASS" else "bad"),
        ("Epsilon", str(bool(report.get("epsilon_result", {}).get("force_closure", False))), "good" if report.get("epsilon_result", {}).get("force_closure") else "bad"),
    ]
    colors = {"good": ("#e7f5ea", "#1b5e20"), "warn": ("#fff4e5", "#b45309"), "bad": ("#fdeaea", "#b91c1c")}
    fig = plt.figure(figsize=(24, 5.0))
    rows = 2
    cols = int(np.ceil(len(cards) / rows))
    axes = fig.subplots(rows, cols).ravel()
    for ax, (title, value, state) in zip(axes, cards):
        bg, fg = colors[state]
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color(fg)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.68, title, ha="center", va="center", color=fg, fontsize=11, weight="bold")
        ax.text(0.5, 0.34, value, ha="center", va="center", color=fg, fontsize=14, weight="bold", wrap=True)
    for ax in axes[len(cards):]:
        ax.axis("off")
    fig.suptitle("Known Water Bottle Pre-Grasp Safety", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_wrench_dashboard(report):
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.14)
    panels = [
        ("Wrench Fx/Fy", OUTPUT_DIR / "wrench_space_fx_fy.png"),
        ("Wrench Fx/Tau", OUTPUT_DIR / "wrench_space_fx_tau.png"),
        ("Wrench Fy/Tau", OUTPUT_DIR / "wrench_space_fy_tau.png"),
    ]
    for idx, (title, path) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        ax.set_title(title, fontsize=14, weight="bold")
        if path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.text(0.5, 0.5, f"Missing image:\n{path}", ha="center", va="center", color="red")
        ax.axis("off")
    ax = fig.add_subplot(grid[1, 1])
    eps = report["epsilon_result"]
    force_closure = bool(eps.get("force_closure", False))
    ax.set_facecolor("#e7f5ea" if force_closure else "#fdeaea")
    ax.set_xticks([])
    ax.set_yticks([])
    text = f"""Pre-Grasp Wrench Analysis

Decision: {report['decision']}
Safety score: {report['final_score']:.1f}/100

Epsilon: {float(eps.get('epsilon', 0.0) or 0.0):.4f}
Force closure: {force_closure}
Weak disturbance direction: {eps.get('weakest_direction')}
Closest hull face distance: {eps.get('closest_face_distance')}
Wrench points: {eps.get('num_wrench_points')}

The green circle is a 2D projection of the 3D planar wrench space [Fx, Fy, tau].
The green marker shows the projected closest hull face point."""
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=13, weight="bold")
    fig.suptitle("Pre-Grasp Wrench Debug Dashboard", fontsize=18, weight="bold")
    fig.savefig(OUTPUT_DIR / "wrench_debug_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_wrench_summary_outputs(epsilon_result):
    summary = compute_wrench_safety_summary(epsilon_result)
    save_wrench_safety_summary(summary, OUTPUT_DIR)
    return summary


def build_epsilon_audit(epsilon_result):
    return {
        "wrench_vector": "[Fx, Fy, tau]",
        "torque_formula": "tau = rx * Fy - ry * Fx",
        "contact_position_normalization": "r = (contact_point - object_centroid) / object_radius",
        "convex_hull": "scipy.spatial.ConvexHull over sampled 3D planar wrench vectors",
        "origin_inside_check": "all hull face equations evaluated at origin satisfy offset <= tolerance",
        "epsilon_formula": "if origin is inside hull, epsilon = minimum distance from origin to hull faces; otherwise epsilon = 0",
        "closest_hull_point": "orthogonal projection of origin onto closest hull face plane",
        "force_closure_blocks_safe": True,
        "force_closure": bool(epsilon_result.get("force_closure", False)),
        "epsilon": float(epsilon_result.get("epsilon", 0.0) or 0.0),
        "closest_face_distance": epsilon_result.get("closest_face_distance"),
        "weakest_direction": epsilon_result.get("weakest_direction"),
        "num_wrench_points": epsilon_result.get("num_wrench_points"),
    }


def build_report(pose, plan, epsilon_result, safety, visibility=None, depth_context=None):
    visibility = visibility or {}
    depth_context = depth_context or {}
    depth_report = depth_context.get("report", {})
    active_positioning = recommend_pregrasp_action(pose, plan, safety, pose["cleaned_mask"].shape)
    calibration_status = build_calibration_status()
    warnings = list(safety["warnings"])
    warnings.extend(calibration_status["warnings"])
    return {
        "mode": "bottle_pregrasp",
        "run_id": CURRENT_RUN_ID,
        "generated_images": [],
        "decision": safety["decision"],
        "pregrasp_decision": safety.get("pregrasp_decision", safety["decision"]),
        "legacy_decision": safety.get("legacy_decision"),
        "decision_stage": safety.get("decision_stage", "pre_close"),
        "final_score": safety["final_score"],
        "safety_score": safety["final_score"],
        "raw_weighted_score": safety.get("raw_weighted_score", safety["final_score"]),
        "decision_score": safety.get("decision_score", safety["final_score"]),
        "capped_decision_score": safety.get("capped_decision_score", safety["final_score"]),
        "cap_reason": safety.get("cap_reason", "none"),
        "decision_policy": safety.get("decision_policy", {}),
        "active_positioning": active_positioning,
        "failure_reasons": safety["failure_reasons"],
        "reposition_reasons": safety.get("reposition_reasons", []),
        "warnings": warnings,
        "calibration_status": calibration_status,
        "known_bottle_validation": build_known_bottle_validation(pose, plan, active_positioning),
        "bottle_detected": True,
        "bottle_geometry": json_safe(pose),
        "planned_contacts": {
            "grasp_center_px": plan["grasp_center_px"],
            "grasp_direction": plan["grasp_direction"],
            "grasp_angle_degrees": plan["grasp_angle_degrees"],
            "body_fraction": plan["body_fraction"],
            "left_contact_px": plan["left_contact_px"],
            "right_contact_px": plan["right_contact_px"],
            "left_contact_3d_m": plan["left_contact_3d_m"],
            "right_contact_3d_m": plan["right_contact_3d_m"],
            "grip_width_px": plan["grip_width_px"],
            "grip_width_mm": plan["grip_width_mm"],
            "width_reachable": plan["width_reachable"],
            "left_reachable": plan["left_reachable"],
            "right_reachable": plan["right_reachable"],
            "raw_grip_width_score": plan.get("raw_grip_width_score"),
            "grip_width_in_range": plan.get("grip_width_in_range"),
            "workspace_reachability_score": plan.get("workspace_reachability_score"),
            "positioning_guidance_score": plan.get("positioning_guidance_score", plan.get("workspace_reachability_score")),
            "workspace_guidance_only": plan.get("workspace_guidance_only", True),
            "workspace_both_reachable": plan.get("workspace_both_reachable"),
            "gripper_workspace": plan.get("gripper_workspace"),
            "contacts_on_body": plan["contacts_on_body"],
            "target_contact_quality": plan["target_contact_quality"],
            "left_depth_stats": plan["left_depth_stats"],
            "right_depth_stats": plan["right_depth_stats"],
            "contact_model_ready": plan.get("contact_model_ready"),
            "contact_model_ready_reasons": plan.get("contact_model_ready_reasons", []),
            "selected_contact_pair": plan.get("selected_contact_pair"),
            "candidate_count": len(plan.get("candidate_contacts", [])),
        },
        "depth_stats": {
            "bottle_depth_median_mm": pose["depth_median_mm"],
            "bottle_depth_mean_mm": pose["depth_mean_mm"],
            "bottle_depth_std_mm": pose["depth_std_mm"],
            "bottle_depth_confidence": pose["depth_confidence"],
            "depth_smoothing_enabled": depth_report.get("depth_smoothing_enabled", USE_DEPTH_SMOOTHING),
            "raw_depth_median_mm": (depth_report.get("raw_depth") or {}).get("median_mm"),
            "smoothed_depth_median_mm": (depth_report.get("smoothed_depth") or {}).get("median_mm"),
            "raw_depth_valid_ratio": (depth_report.get("raw_depth") or {}).get("valid_ratio"),
            "smoothed_depth_valid_ratio": (depth_report.get("smoothed_depth") or {}).get("valid_ratio"),
            "left_patch_depth_stats": plan["left_depth_stats"],
            "right_patch_depth_stats": plan["right_depth_stats"],
        },
        "depth_smoothing": depth_report,
        "alignment_error_px": safety["alignment_error_px"],
        # propagate the axis tilt so the post-close tilt->REGRIP gate in
        # stage_evaluators can actually read it (previously dropped here, so the
        # gate silently never fired)
        "axis_vertical_error_degrees": safety.get("axis_vertical_error_degrees", 0.0),
        "axis_angle_degrees": safety.get("axis_angle_degrees", float(pose.get("axis_angle_degrees", 0.0))),
        "axis_tilt_score": safety.get("axis_tilt_score"),
        "scores": safety["scores"],
        "risk_scores": safety["risk_scores"],
        "metric_audit": safety.get("metric_audit", []),
        "hard_gates": safety.get("hard_gates", {}),
        "true_hard_gate_reasons": safety.get("true_hard_gate_reasons", safety.get("failure_reasons", [])),
        "contact_model_ready_reasons": safety.get("contact_model_ready_reasons", plan.get("contact_model_ready_reasons", [])),
        "positioning_guidance_score": safety.get("positioning_guidance_score", plan.get("positioning_guidance_score", plan.get("workspace_reachability_score"))),
        "workspace_guidance_only": True,
        "alignment_guidance": safety.get("alignment_guidance", {}),
        "selected_preclose_decision": safety.get("selected_preclose_decision", safety["decision"]),
        "score_weights": safety["score_weights"],
        "epsilon_result": epsilon_result,
        "epsilon_audit": build_epsilon_audit(epsilon_result),
        "wrench_safety_summary": compute_wrench_safety_summary(epsilon_result),
        "full_bottle_visible": bool(visibility.get("full_bottle_visible", True)),
        "visibility_score": float(visibility.get("visibility_score", 100.0) or 0.0),
        "border_margins": visibility.get("border_margins", {}),
        "visibility": visibility,
        "recommended_view_action": visibility.get("recommended_view_action", "VIEW_OK"),
        "contact_model_ready": bool(plan.get("contact_model_ready", True)),
        "candidate_contacts": plan.get("candidate_contacts", []),
        "selected_contact_pair": plan.get("selected_contact_pair"),
        "rejected_candidates": plan.get("rejected_candidates", []),
    }


def write_scoring_audit_report(report):
    lines = []
    lines.append("Pre-Grasp Scoring Audit Report")
    lines.append("=" * 34)
    lines.append(f"Decision: {report.get('decision')}")
    lines.append(f"Legacy lift wording: {report.get('legacy_decision')}")
    lines.append(f"Raw weighted score: {float(report.get('raw_weighted_score', report.get('final_score', 0.0)) or 0.0):.2f}/100")
    lines.append(f"Capped decision score: {float(report.get('capped_decision_score', report.get('final_score', 0.0)) or 0.0):.2f}/100")
    lines.append(f"Cap reason: {report.get('cap_reason', 'none')}")
    lines.append(f"Final score: {float(report.get('final_score', 0.0) or 0.0):.2f}/100")
    lines.append("")
    lines.append("Decision Logic")
    lines.append("- Stage: pre-close, so the ready state is READY TO CLOSE.")
    lines.append("- If any hard gate fails: DO NOT CLOSE")
    lines.append("- Else if repositioning is needed: ADJUST POSITION")
    lines.append("- Else if score >= 80: READY TO CLOSE")
    lines.append("- Else if score >= 60: ADJUST POSITION")
    lines.append("- Else: DO NOT CLOSE")
    lines.append("")
    lines.append("Hard Gates")
    gates = report.get("hard_gates", {})
    for name, passed in gates.get("status", {}).items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    if gates.get("failed"):
        lines.append("Failed reasons:")
        for reason in gates["failed"]:
            lines.append(f"- {reason}")
    lines.append("")
    lines.append("Full Bottle Visibility")
    visibility = report.get("visibility", {})
    lines.append(f"- full_bottle_visible: {report.get('full_bottle_visible')}")
    lines.append(f"- visibility_score: {float(report.get('visibility_score', 0.0) or 0.0):.2f}/100")
    lines.append(f"- border_margins: {report.get('border_margins', {})}")
    lines.append(f"- required_margin_px: {visibility.get('required_margin_px')}")
    lines.append(f"- recommended_view_action: {report.get('recommended_view_action')}")
    lines.append(f"- missing_region_reason: {visibility.get('missing_region_reason', [])}")
    lines.append(f"- contact_model_ready: {report.get('contact_model_ready')}")
    lines.append("")
    lines.append("Candidate Contact Model")
    lines.append(f"- candidates: {len(report.get('candidate_contacts', []))}")
    lines.append(f"- selected_contact_pair: {report.get('selected_contact_pair')}")
    rejected = report.get("rejected_candidates", [])
    lines.append(f"- rejected_candidates: {len(rejected)}")
    for item in rejected[:8]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("Reposition Reasons")
    reposition = report.get("reposition_reasons", [])
    if reposition:
        for reason in reposition:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Calibration Status")
    calibration = report.get("calibration_status", {})
    lines.append(f"- workspace_config_source: {calibration.get('workspace_config_source')}")
    lines.append(f"- grip_width_limits_source: {calibration.get('grip_width_limits_source')}")
    for warning in calibration.get("warnings", []):
        lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("Known Bottle Validation")
    validation = report.get("known_bottle_validation", {})
    lines.append(f"- expected: {validation.get('expected', {})}")
    lines.append(f"- measured: {validation.get('measured', {})}")
    lines.append(f"- checks: {validation.get('checks', {})}")
    lines.append("")
    lines.append("Metric Details")
    for metric in report.get("metric_audit", []):
        lines.append(f"\n{metric['name']}")
        lines.append(f"  value: {metric['value']:.2f}/100")
        lines.append(f"  formula: {metric['formula']}")
        lines.append(f"  inputs: {metric['inputs']}")
        lines.append(f"  units: {metric['units']}")
        lines.append(f"  range: {metric['range'][0]:.0f}-{metric['range'][1]:.0f}")
        lines.append(f"  clamped_0_100: {metric['clamped_0_100']}")
        lines.append(f"  higher_is_better: {metric['higher_is_better']}")
        lines.append(f"  thresholds: {metric['thresholds']}")
        lines.append(f"  can_trigger_hard_fail: {metric['can_trigger_hard_fail']}")
        lines.append(f"  pass/fail: {'PASS' if metric['passed'] else 'FAIL'}")
        lines.append(f"  weight: {metric['weight']:.2f}")
        lines.append(f"  weighted contribution: {metric['weighted_contribution']:.2f}")
    lines.append("")
    lines.append("Risk Metrics")
    lines.append("- miss_risk, slip_risk, and rotation_risk are risk values where lower is better.")
    lines.append("- miss_score, slip_score, and rotation_score are safety scores where higher is better.")
    for key, value in report.get("risk_scores", {}).items():
        lines.append(f"- {key}: {float(value):.2f}/100")
    lines.append("")
    lines.append("Epsilon Audit")
    for key, value in report.get("epsilon_audit", {}).items():
        lines.append(f"- {key}: {value}")
    with open(OUTPUT_DIR / "scoring_audit_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_report(report):
    with open(OUTPUT_DIR / "pregrasp_safety_report.json", "w") as f:
        json.dump(json_safe(report), f, indent=4)
    with open(OUTPUT_DIR / "prelift_safety_report.json", "w") as f:
        json.dump(json_safe(report), f, indent=4)
    with open(OUTPUT_DIR / "bottle_grasp_result.json", "w") as f:
        json.dump(json_safe(report), f, indent=4)
    write_scoring_audit_report(report)


def run_bottle_pregrasp():
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_id = start_current_run()
    print(f"Current bottle run id: {run_id}")
    print_pipeline_audit()
    rgb, depth_raw, intrinsics = load_inputs()
    depth_context = build_depth_context(depth_raw)
    depth_smooth = depth_context["depth_smooth"]
    pose, pose_error = detect_bottle_pose(rgb, depth_smooth)
    if pose is None:
        save_failure(rgb, pose_error)
        raise SystemExit(1)
    depth_context = build_depth_context(depth_raw, pose.get("cleaned_mask"))

    visibility = check_full_bottle_visibility(pose, rgb.shape)
    if not visibility.get("full_bottle_visible", False):
        report = build_visibility_gate_report(pose, visibility)
        report["depth_smoothing"] = depth_context["report"]
        report["depth_stats"].update({
            "depth_smoothing_enabled": depth_context["report"].get("depth_smoothing_enabled"),
            "raw_depth_median_mm": depth_context["report"].get("raw_depth", {}).get("median_mm"),
            "smoothed_depth_median_mm": depth_context["report"].get("smoothed_depth", {}).get("median_mm"),
            "raw_depth_valid_ratio": depth_context["report"].get("raw_depth", {}).get("valid_ratio"),
            "smoothed_depth_valid_ratio": depth_context["report"].get("smoothed_depth", {}).get("valid_ratio"),
        })
        write_report(report)
        save_scene_and_segmentation(rgb, pose)
        save_mask_debug_images(pose)
        save_visibility_gate_overlay(rgb, pose, visibility, report)
        save_depth_input_debug(depth_raw, depth_smooth, pose)
        save_placeholder_wrench_outputs(report["epsilon_result"], "Full bottle is not visible.")
        save_dashboard_summary(report)
        save_wrench_dashboard(report)
        report = finalize_current_run(report, [
            "scene.png",
            "segmentation.png",
            "bottle_raw_mask.png",
            "bottle_cleaned_mask.png",
            "bottle_body_mask.png",
            "mask_cleanup_debug.png",
            "depth_colormap.png",
            "depth_raw_colormap.png",
            "depth_smooth_colormap.png",
            "depth_noise_debug.png",
            "bottle_grasp_debug.png",
            "best_grasp_overlay.png",
            "dashboard_summary.png",
            "wrench_debug_dashboard.png",
            "wrench_safety_summary.png",
            "wrench_space_fx_fy.png",
            "wrench_space_fx_tau.png",
            "wrench_space_fy_tau.png",
        ])
        print("Bottle pre-grasp mode stopped at full-bottle visibility gate.")
        print(f"Decision: {report['decision']}")
        print(f"Visibility score: {report['visibility_score']:.1f}/100")
        print(f"Recommended view action: {report['recommended_view_action']}")
        print("Contact modeling disabled until the full bottle is visible.")
        if report["failure_reasons"]:
            print("Failure reasons:")
            for reason in report["failure_reasons"]:
                print(f"- {reason}")
        print("Saved:")
        print("output/pregrasp_safety_report.json")
        print("output/scoring_audit_report.txt")
        print("output/bottle_grasp_debug.png")
        print("output/dashboard_summary.png")
        print("output/wrench_debug_dashboard.png")
        return

    plan, plan_error = plan_pregrasp(pose, depth_smooth, intrinsics)
    if plan is None:
        save_failure(rgb, plan_error)
        raise SystemExit(1)
    plan["full_bottle_visible"] = bool(visibility.get("full_bottle_visible", False))
    plan["visibility_score"] = float(visibility.get("visibility_score", 0.0) or 0.0)
    plan["visibility"] = visibility

    object_radius = max(plan["grip_width_px"] / 2.0, 1.0)
    epsilon_result = compute_epsilon_quality(
        plan["epsilon_contacts"],
        object_centroid=pose["center_px"],
        object_radius=object_radius,
        mu=FRICTION_COEFFICIENT,
        samples_per_contact=8,
    )
    save_epsilon_result_json(
        epsilon_result,
        plan["epsilon_contacts"],
        object_centroid=pose["center_px"],
        object_radius=object_radius,
        output_path=OUTPUT_DIR / "epsilon_result.json",
    )
    save_epsilon_result_json(
        epsilon_result,
        plan["epsilon_contacts"],
        object_centroid=pose["center_px"],
        object_radius=object_radius,
        output_path=OUTPUT_DIR / "bottle_epsilon_result.json",
    )
    save_all_wrench_space_projections(epsilon_result, output_dir=str(OUTPUT_DIR), mu=FRICTION_COEFFICIENT)
    save_wrench_summary_outputs(epsilon_result)

    safety = evaluate_pregrasp_safety(pose, plan, epsilon_result, rgb.shape)
    report = build_report(pose, plan, epsilon_result, safety, visibility, depth_context)
    write_report(report)

    save_scene_and_segmentation(rgb, pose)
    save_mask_debug_images(pose)
    save_pregrasp_overlay(rgb, pose, plan, report)
    save_depth_debug(depth_raw, depth_smooth, pose, plan)
    save_dashboard_summary(report)
    save_wrench_dashboard(report)
    report = finalize_current_run(report, [
        "scene.png",
        "segmentation.png",
        "bottle_raw_mask.png",
        "bottle_cleaned_mask.png",
        "bottle_body_mask.png",
        "mask_cleanup_debug.png",
        "bottle_grasp_debug.png",
        "best_grasp_overlay.png",
        "depth_colormap.png",
        "depth_raw_colormap.png",
        "depth_smooth_colormap.png",
        "depth_noise_debug.png",
        "rgbd_contact_validation_debug.png",
        "dashboard_summary.png",
        "wrench_debug_dashboard.png",
        "wrench_safety_summary.png",
        "wrench_space_fx_fy.png",
        "wrench_space_fx_tau.png",
        "wrench_space_fy_tau.png",
    ])

    print("Bottle pre-grasp mode complete.")
    print(f"Pre-grasp decision: {report['decision']}")
    print(f"Safety score: {report['final_score']:.1f}/100")
    print(f"Bottle detected: {report['bottle_detected']}")
    print(f"Bottle confidence: {pose['detection_confidence']:.2f}")
    print(f"Full bottle visible: {report['full_bottle_visible']}")
    print(f"Visibility score: {report['visibility_score']:.1f}/100")
    print(f"Recommended view action: {report['recommended_view_action']}")
    print(f"Contact candidates: {len(report.get('candidate_contacts', []))}")
    print(f"Bottle center px: {pose['center_px']}")
    print(f"Bottle axis angle deg: {pose['axis_angle_degrees']:.1f}")
    print(f"Grip width px: {plan['grip_width_px']:.1f}")
    if plan.get("grip_width_mm") is not None:
        print(f"Grip width mm: {plan['grip_width_mm']:.1f}")
    print(f"Workspace reach score: {plan.get('workspace_reachability_score', 0.0):.1f}")
    print(f"Left/right reachable: {plan.get('left_reachable')} / {plan.get('right_reachable')}")
    print(f"Recommended action: {report['active_positioning']['recommended_action']}")
    print(f"Positioning state: {report['active_positioning']['state']}")
    print(
        "Action vector px/mm: "
        f"dx={report['active_positioning'].get('action_dx_px')}, "
        f"dy={report['active_positioning'].get('action_dy_px')}, "
        f"depth={report['active_positioning'].get('action_depth_mm')}"
    )
    print(f"Alignment error px: {report['alignment_error_px']:.1f}")
    print(f"Miss risk: {report['risk_scores']['miss_risk']:.1f}")
    print(f"Slip risk: {report['risk_scores']['slip_risk']:.1f}")
    print(f"Rotation risk: {report['risk_scores']['rotation_risk']:.1f}")
    print(f"Epsilon closure: {epsilon_result.get('force_closure', False)}")
    if report["failure_reasons"]:
        print("Failure reasons:")
        for reason in report["failure_reasons"]:
            print(f"- {reason}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print("Saved:")
    print("output/pregrasp_safety_report.json")
    print("output/scoring_audit_report.txt")
    print("output/bottle_grasp_debug.png")
    print("output/dashboard_summary.png")
    print("output/wrench_debug_dashboard.png")


if __name__ == "__main__":
    run_bottle_pregrasp()
