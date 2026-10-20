"""Object scan workflow built from existing RGB-D/point-cloud outputs.

The scanner creates reviewable object-profile drafts. It does not train YOLO,
open cameras, change RealSense tracking, alter point-cloud algorithms, or
bypass production safety checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from grasp_project.common.config import OUTPUT_DIR, PROJECT_ROOT
from grasp_project.common.file_io import read_json, write_json_atomic
from .object_profile import create_object_profile, match_profiles, observation_descriptor, slugify, utc_iso
from .object_registry import PROFILE_OUTPUT_DIR, load_profile, list_profiles, save_profile

OBJECT_SCAN_DURATION_SEC = 3.0
SCAN_STATES = {"IDLE", "WAITING_FOR_OBJECT", "SCANNING", "PROCESSING", "REVIEW_PROFILE", "SAVED", "FAILED"}
LATEST_SCAN_REPORT_PATH = PROFILE_OUTPUT_DIR / "latest_scan_report.json"
LATEST_SCAN_DRAFT_PATH = PROFILE_OUTPUT_DIR / "latest_profile_draft.json"


def _try_cv2():
    try:
        import cv2

        return cv2
    except Exception:
        return None


def _read_image(path: Path):
    cv2 = _try_cv2()
    if cv2 is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, rgb) -> bool:
    cv2 = _try_cv2()
    if cv2 is None or rgb is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return bool(cv2.imwrite(str(path), arr))
    return bool(cv2.imwrite(str(path), cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)))


def _mask_from_bbox(shape: tuple[int, int], bbox) -> np.ndarray | None:
    if not bbox or len(bbox) < 4:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def dominant_object_mask(rgb=None, depth=None, bbox=None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find a prominent object mask using existing bbox, depth, then RGB cues."""
    cv2 = _try_cv2()
    if rgb is not None:
        shape = np.asarray(rgb).shape[:2]
    elif depth is not None:
        shape = np.asarray(depth).shape[:2]
    else:
        return None, {"mask_source": "none", "reason": "missing_rgb_and_depth"}
    bbox_mask = _mask_from_bbox(shape, bbox)
    if bbox_mask is not None:
        return bbox_mask, {"mask_source": "current_realsense_bbox", "reason": "used_current_live_bbox"}
    if cv2 is None:
        return None, {"mask_source": "unavailable", "reason": "opencv_unavailable"}
    if depth is not None:
        arr = np.asarray(depth)
        gray = arr[..., 0] if arr.ndim == 3 else arr
        valid = np.isfinite(gray) & (gray > 0)
        if np.any(valid):
            values = gray[valid].astype(float)
            lo, hi = np.percentile(values, [5, 45])
            mask = ((gray >= lo) & (gray <= hi) & valid).astype(np.uint8)
            cleaned = _clean_mask(mask)
            if np.count_nonzero(cleaned) > 100:
                return cleaned, {"mask_source": "depth_foreground", "reason": "dominant_near_depth_component"}
    if rgb is not None:
        gray = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Keep the smaller side when it is likely to be the foreground object.
        mask = (otsu > 0).astype(np.uint8)
        if np.count_nonzero(mask) > mask.size * 0.55:
            mask = 1 - mask
        cleaned = _clean_mask(mask)
        if np.count_nonzero(cleaned) > 100:
            return cleaned, {"mask_source": "rgb_foreground", "reason": "dominant_rgb_component"}
    return None, {"mask_source": "none", "reason": "no_prominent_object_component"}


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    cv2 = _try_cv2()
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if cv2 is None:
        return mask
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(cleaned)
    cv2.drawContours(out, [largest], -1, 1, thickness=-1)
    return out


def _save_artifacts(scan_dir: Path, rgb=None, mask=None, depth=None, pointcloud_status=None) -> dict[str, str]:
    scan_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if rgb is not None and _write_rgb(scan_dir / "rgb_snapshot.jpg", rgb):
        artifacts["rgb_snapshot"] = str((scan_dir / "rgb_snapshot.jpg").relative_to(PROJECT_ROOT))
        cv2 = _try_cv2()
        if cv2 is not None:
            gray = cv2.cvtColor(np.asarray(rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 160)
            if _write_rgb(scan_dir / "grayscale_template.jpg", gray):
                artifacts["grayscale_template"] = str((scan_dir / "grayscale_template.jpg").relative_to(PROJECT_ROOT))
            if _write_rgb(scan_dir / "edge_template.jpg", edges):
                artifacts["edge_template"] = str((scan_dir / "edge_template.jpg").relative_to(PROJECT_ROOT))
    if mask is not None:
        mask_img = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if _write_rgb(scan_dir / "mask.png", mask_img):
            artifacts["mask"] = str((scan_dir / "mask.png").relative_to(PROJECT_ROOT))
        contour_payload = {"contour_points_normalized": []}
        from .object_profile import normalized_contour_from_mask

        contour_payload["contour_points_normalized"] = normalized_contour_from_mask(mask_img)
        write_json_atomic(scan_dir / "contour.json", contour_payload)
        artifacts["contour"] = str((scan_dir / "contour.json").relative_to(PROJECT_ROOT))
    if depth is not None and _write_rgb(scan_dir / "depth_snapshot.png", depth):
        artifacts["depth_snapshot"] = str((scan_dir / "depth_snapshot.png").relative_to(PROJECT_ROOT))
    pc = pointcloud_status or {}
    for key, out_name in (("latest_realsense_pointcloud_ply", "pointcloud.ply"), ("latest_bottle_pointcloud_ply", "object_pointcloud.ply")):
        src = pc.get(key)
        if src:
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = PROJECT_ROOT / src_path
            if src_path.exists():
                dst = scan_dir / out_name
                shutil.copyfile(src_path, dst)
                artifacts["pointcloud" if out_name == "pointcloud.ply" else "object_cloud"] = str(dst.relative_to(PROJECT_ROOT))
    preview = pc.get("pointcloud_preview_path") or pc.get("bottle_pointcloud_preview_path")
    if preview:
        src_path = Path(preview)
        if not src_path.is_absolute():
            src_path = PROJECT_ROOT / src_path
        if src_path.exists():
            dst = scan_dir / "pointcloud_preview.jpg"
            shutil.copyfile(src_path, dst)
            artifacts["pointcloud_preview"] = str(dst.relative_to(PROJECT_ROOT))
    return artifacts


def scan_from_arrays(
    *,
    display_name: str = "Scanned Object",
    rgb=None,
    depth=None,
    mask=None,
    mask_meta_override: dict[str, Any] | None = None,
    intrinsics: dict[str, Any] | None = None,
    pointcloud_status: dict[str, Any] | None = None,
    bbox=None,
    duration_sec: float = OBJECT_SCAN_DURATION_SEC,
    output_root: Path = PROFILE_OUTPUT_DIR,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    started = time.time()
    status = {
        "object_scan_status": "SCANNING",
        "object_scan_progress": 0.1,
        "object_scan_error": None,
        "object_scan_frame_count": max(1, int(round(float(duration_sec) * 5.0))),
        "object_scan_stable_frames": 0,
        "object_scan_confidence": 0.0,
    }
    if mask is None:
        mask, mask_meta = dominant_object_mask(rgb=rgb, depth=depth, bbox=bbox)
    else:
        mask_meta = mask_meta_override or {"mask_source": "provided_mask", "reason": "caller_supplied_mask"}
    if mask is None or np.count_nonzero(mask) < 100:
        status.update({
            "object_scan_status": "FAILED",
            "object_scan_progress": 1.0,
            "object_scan_error": mask_meta.get("reason") or "object_not_found",
        })
        return {"ok": False, "status": status, "message": status["object_scan_error"], "profile_draft": None, "scan_report": status}
    status["object_scan_status"] = "PROCESSING"
    object_id = slugify(display_name)
    scan_id = f"{object_id}-{int(started)}"
    scan_dir = output_root / object_id / scan_id
    artifacts = _save_artifacts(scan_dir, rgb=rgb, mask=mask, depth=depth, pointcloud_status=pointcloud_status) if save_artifacts else {}
    profile = create_object_profile(
        object_id=object_id,
        display_name=display_name,
        rgb=rgb,
        mask=mask,
        depth=depth,
        intrinsics=intrinsics,
        pointcloud_status=pointcloud_status,
        artifact_paths={
            "rgb_snapshot": artifacts.get("rgb_snapshot"),
            "grayscale_template": artifacts.get("grayscale_template"),
            "edge_template": artifacts.get("edge_template"),
        },
        source="object_scan",
    )
    if artifacts.get("pointcloud"):
        profile["representative_pointcloud_path"] = artifacts["pointcloud"]
    if artifacts.get("object_cloud"):
        profile["bottle_or_object_cloud_path"] = artifacts["object_cloud"]
    profile["scan_artifact_dir"] = str(scan_dir.relative_to(PROJECT_ROOT)) if scan_dir.exists() else None
    matches = match_profiles(observation_descriptor(rgb=rgb, mask=mask, depth=depth, intrinsics=intrinsics, pointcloud_status=pointcloud_status), [p for p in _safe_profiles()])
    confidence = float(profile.get("profile_confidence") or 0.0)
    status.update({
        "object_scan_status": "REVIEW_PROFILE",
        "object_scan_progress": 1.0,
        "object_scan_stable_frames": status["object_scan_frame_count"],
        "object_scan_confidence": confidence,
    })
    report = {
        **status,
        "ok": True,
        "scan_id": scan_id,
        "scan_duration_sec": duration_sec,
        "mask_source": mask_meta.get("mask_source"),
        "mask_reason": mask_meta.get("reason"),
        "profile_draft": profile,
        "profile_match": matches,
        "artifacts": artifacts,
        "production_safety_unchanged": True,
        "scan_state_sequence": ["WAITING_FOR_OBJECT", "SCANNING", "PROCESSING", "REVIEW_PROFILE"],
        "message": "Object profile draft ready for review.",
    }
    if save_artifacts:
        write_json_atomic(scan_dir / "scan_report.json", report)
        write_json_atomic(LATEST_SCAN_REPORT_PATH, report)
        write_json_atomic(LATEST_SCAN_DRAFT_PATH, profile)
    return {"ok": True, "status": status, "profile_draft": profile, "scan_report": report, "message": report["message"]}


def _safe_profiles() -> list[dict[str, Any]]:
    try:
        return [load_profile(p["object_id"]) for p in list_profiles()]
    except Exception:
        return []


def scan_from_live_outputs(display_name: str = "Scanned Object", duration_sec: float = OBJECT_SCAN_DURATION_SEC) -> dict[str, Any]:
    rgb_path = OUTPUT_DIR / "latest_rgb_live.jpg"
    depth_path = OUTPUT_DIR / "latest_depth_live.jpg"
    realtime = read_json(OUTPUT_DIR / "latest_realtime_status.json", {})
    pointcloud_status = read_json(OUTPUT_DIR / "pointcloud_status.json", {})
    rgb = _read_image(rgb_path)
    depth_img = _read_image(depth_path)
    if not realtime.get("realsense_rgb_active") or rgb is None:
        return _failed_live_scan("realsense_rgb_inactive_or_missing")
    if not realtime.get("realsense_depth_active") and depth_img is None:
        return _failed_live_scan("realsense_depth_inactive_or_missing")
    bbox = (
        realtime.get("bottle_extent_bbox_xyxy")
        or ((realtime.get("realsense_perception") or {}).get("bottle_bbox_xyxy"))
        or (((realtime.get("target_lock") or {}).get("current_tracked_target") or {}).get("accepted_current_bottle_bbox"))
    )
    intrinsics = realtime.get("camera_intrinsics") or realtime.get("intrinsics") or {}
    mask, mask_meta = dominant_object_mask(rgb=rgb, depth=depth_img, bbox=bbox)
    return scan_from_arrays(
        display_name=display_name,
        rgb=rgb,
        depth=None,
        mask=mask,
        mask_meta_override=mask_meta,
        intrinsics=intrinsics,
        pointcloud_status=pointcloud_status,
        bbox=bbox,
        duration_sec=duration_sec,
        save_artifacts=True,
    )


def _failed_live_scan(reason: str) -> dict[str, Any]:
    report = {
        "ok": False,
        "object_scan_status": "FAILED",
        "object_scan_progress": 1.0,
        "object_scan_error": reason,
        "object_scan_frame_count": 0,
        "object_scan_stable_frames": 0,
        "object_scan_confidence": 0.0,
        "message": reason,
        "timestamp": utc_iso(),
    }
    write_json_atomic(LATEST_SCAN_REPORT_PATH, report)
    return {"ok": False, "status": report, "profile_draft": None, "scan_report": report, "message": reason}


def save_latest_draft(display_name: str | None = None, notes: str = "") -> dict[str, Any]:
    draft = read_json(LATEST_SCAN_DRAFT_PATH, {})
    if not draft:
        raise ValueError("No object profile draft is available. Run Scan Object first.")
    if display_name:
        draft["display_name"] = display_name
        draft["object_id"] = slugify(display_name)
    if notes:
        draft["notes"] = notes
    saved = save_profile(draft)
    report = read_json(LATEST_SCAN_REPORT_PATH, {})
    report.update({"object_scan_status": "SAVED", "saved_profile": saved, "message": "Object profile saved."})
    write_json_atomic(LATEST_SCAN_REPORT_PATH, report)
    return saved


def enhance_latest_profile_with_multiview(object_id: str | None = None) -> dict[str, Any]:
    profile = load_profile(object_id) if object_id else read_json(LATEST_SCAN_DRAFT_PATH, {})
    if not profile:
        raise ValueError("Profile not found for multi-view enhancement.")
    mv = read_json(OUTPUT_DIR / "multiview" / "latest_status.json", {})
    if not mv or not mv.get("filtered_accumulated_point_count"):
        profile["multiview_enhancement_status"] = "unavailable"
        profile["multiview_enhancement_reason"] = "no_accumulated_multiview_model"
        return profile
    profile["multiview_enhancement_status"] = "updated"
    profile["multiview_surface_coverage_percent"] = mv.get("estimated_surface_coverage_percent") or mv.get("model_coverage_score")
    profile["multiview_accumulated_point_count"] = mv.get("filtered_accumulated_point_count")
    profile["pointcloud_quality"] = max(float(profile.get("pointcloud_quality") or 0.0), 0.70)
    profile["geometry_confidence"] = max(float(profile.get("geometry_confidence") or 0.0), 0.70)
    if object_id:
        return save_profile(profile)
    write_json_atomic(LATEST_SCAN_DRAFT_PATH, profile)
    return profile


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scan/enroll an object profile from existing live outputs.")
    parser.add_argument("--scan", action="store_true", help="Scan current live object and save profile")
    parser.add_argument("--name", default="Scanned Object", help="Display name for scanned profile")
    parser.add_argument("--duration", type=float, default=OBJECT_SCAN_DURATION_SEC, help="Scan duration metadata in seconds")
    parser.add_argument("--draft-only", action="store_true", help="Create draft without saving to registry")
    args = parser.parse_args(argv)
    if not args.scan:
        parser.print_help()
        return 0
    result = scan_from_live_outputs(args.name, duration_sec=args.duration)
    if not result.get("ok"):
        print(json.dumps(result, indent=2))
        return 1
    if not args.draft_only:
        result["saved_profile"] = save_latest_draft(args.name)
        result["status"]["object_scan_status"] = "SAVED"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
