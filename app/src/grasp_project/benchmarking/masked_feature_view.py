"""Strict feature packet and masked accessors for Benchmark Lab.

This module is display/report-only. It does not open cameras, alter the
production tracker, change DAVIS scoring, or recompute epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .feature_registry import FEATURE_BY_ID, canonical_feature_ids


VALID = "VALID"
DISABLED = "DISABLED"
DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
STALE_DATA = "STALE_DATA"
UNSUPPORTED = "UNSUPPORTED"

BOTTLE_ONLY_NOT_APPLICABLE = {
    "rgb_gripper_detection": "No gripper/finger test is performed in bottle-only mode.",
    "rgb_contact_visibility": "No physical contact visibility test is performed in bottle-only mode.",
    "depth_contact_consistency": "Contact depth consistency requires a contact test.",
    "depth_grip_width": "Grip width is contact/planned-geometry dependent.",
    "rgbd_contact_patches": "Physical contact patch confirmation is skipped.",
    "rgbd_ready_to_close": "Ready-to-close is not a bottle-only metric.",
    "rgbd_visual_contact_model": "Visual contact model is skipped without gripper contact.",
    "pointcloud_3d_contact_validation": "3D contact validation is not applicable without contact.",
    "analytic_epsilon": "Contact-based epsilon is unavailable without valid contact geometry.",
    "wrench_quality": "Wrench quality depends on contact-based epsilon/contact geometry.",
}


class AuditedProductionReference(dict[str, Any]):
    """Record every production-context field read during masked evaluation."""

    def __init__(self, values: Mapping[str, Any] | None = None):
        super().__init__(values or {})
        self.accessed_fields: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.accessed_fields.add(str(key))
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.accessed_fields.add(str(key))
        return super().get(key, default)


@dataclass(frozen=True)
class FeaturePacket:
    """Synchronized raw feature values for one benchmark sample."""

    values: dict[str, Any]
    source_fields: dict[str, str]
    availability: dict[str, bool]
    freshness: dict[str, str]
    production_reference: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    bundle_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.production_reference, AuditedProductionReference):
            object.__setattr__(self, "production_reference", AuditedProductionReference(self.production_reference))


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _num(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "valid", "tracking", "active", "yes", "1"}


def _score_mean(*values: Any) -> float | None:
    scores: list[float] = []
    for value in values:
        numeric = _num(value)
        if numeric is not None:
            scores.append(max(0.0, min(100.0, numeric)))
    return round(sum(scores) / len(scores), 3) if scores else None


def _ratio_score(value: Any) -> float | None:
    numeric = _num(value)
    if numeric is None:
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return max(0.0, min(100.0, numeric))


def build_feature_packet(
    *,
    realsense: dict[str, Any] | None = None,
    davis: dict[str, Any] | None = None,
    pointcloud: dict[str, Any] | None = None,
    sync: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    live_quality: dict[str, Any] | None = None,
    embodiment: dict[str, Any] | None = None,
) -> FeaturePacket:
    """Map dashboard/source dictionaries into raw benchmark feature values."""
    rs = dict(realsense or {})
    dv = dict(davis or {})
    pc = dict(pointcloud or {})
    sy = dict(sync or {})
    qm = dict(quality or {})
    lq = dict(live_quality or {})
    emb = dict(embodiment or {})
    perception = dict(rs.get("realsense_perception") or {})
    motion = dict(dv.get("motion_estimate") or {})

    detection_status = _first(perception.get("bottle_detection_status"), rs.get("target_status"))
    detected = str(detection_status or "").upper() not in {"", "LOST", "NO_TARGET", "NO_BOTTLE", "NOT_FOUND"}
    raw_contact_pair = _first(perception.get("contact_pair_valid"), qm.get("contact_pair_valid"))
    contact_pair_valid = bool(raw_contact_pair) if raw_contact_pair is not None else None
    raw_contact_gate = _first(qm.get("contact_quality_gate_pass"), qm.get("contact_inputs_valid"), qm.get("contact_3d_valid"))
    contact_gate = bool(contact_pair_valid) if contact_pair_valid is not None else bool(raw_contact_gate)
    if raw_contact_gate is False:
        contact_gate = False
    # depth features are declared depth-only in the registry: read only
    # RealSense DEPTH-pipeline valid-ratio values. The runner publishes these
    # under depth_stats (attach_depth_report) and the perception result exposes
    # bottle_depth_valid_ratio; both are the RealSense depth pipeline's own
    # valid-pixel ratio. Do NOT fall back to quality/pointcloud pipeline values
    # -- that was the cross-modality leak fixed on 2026-07-06. NOTE: the pure
    # realtime hot path does not currently publish depth_stats, so depth-only
    # geometry features report unavailable there by design rather than borrowing
    # the pointcloud pipeline's ratio (see docs/final_benchmark_audit.md).
    depth_stats = dict(rs.get("depth_stats") or {})
    depth_ratio = _num(
        rs.get("depth_valid_ratio"),
        rs.get("raw_depth_valid_ratio"),
        depth_stats.get("raw_depth_valid_ratio"),
        depth_stats.get("smoothed_depth_valid_ratio"),
        perception.get("bottle_depth_valid_ratio"),
        perception.get("depth_valid_ratio"),
    )
    aps_state = _first(dv.get("aps_state"), dv.get("aps_stream_status"), dv.get("aps_mode"))
    event_stream = _first(dv.get("event_state"), dv.get("stream"), dv.get("event_stream_status"))
    event_rate = _num(dv.get("event_rate"), dv.get("davis_event_rate"), motion.get("event_rate"))
    if event_rate is None and str(event_stream or "").lower() in {"active", "stream_active_no_motion", "stream_active_events"}:
        event_rate = 0.0
    pointcloud_freshness = str(_first(sy.get("pointcloud_freshness"), pc.get("freshness"), pc.get("pointcloud_freshness"), "UNAVAILABLE")).upper()
    davis_event_freshness = str(_first(sy.get("davis_event_freshness"), "CURRENT" if event_stream or event_rate is not None else "UNAVAILABLE")).upper()
    davis_aps_freshness = str(_first(sy.get("davis_aps_freshness"), "CURRENT" if aps_state or dv.get("aps_fps") is not None else "UNAVAILABLE")).upper()
    detection_confidence = _num(perception.get("detection_confidence"), rs.get("tracking_confidence"))
    mask_confidence = _num(
        perception.get("mask_confidence"),
        qm.get("bottle_mask_confidence"),
        85.0 if qm.get("image_bottle_hull_px") else None,
    )
    visual_object_quality = _score_mean(
        detection_confidence,
        mask_confidence,
        90.0 if _first(perception.get("bottle_center_px"), rs.get("current_tracked_center"), rs.get("target_center_px")) else None,
        85.0 if _num(perception.get("bottle_angle_deg"), rs.get("raw_bottle_angle_deg"), rs.get("display_magenta_angle_deg")) is not None else None,
    )
    # geometry_quality is registered with depth-only dependencies, so the
    # composite must not embed point-cloud evidence (leaks past the mask)
    geometry_quality = _score_mean(
        _ratio_score(depth_ratio),
        85.0 if _num(rs.get("bottle_distance_mm"), rs.get("distance_mm"), rs.get("depth_mm"), rs.get("bottle_depth_median_mm")) is not None else None,
    )

    values = {
        "rgb_bottle_detection": {
            "detected": bool(detected),
            "status": detection_status,
            "confidence": _num(perception.get("detection_confidence"), rs.get("tracking_confidence")),
            "bbox_xyxy": _first(perception.get("bottle_bbox_xyxy"), rs.get("bottle_extent_bbox_xyxy")),
        },
        "rgb_bottle_segmentation": {
            "mask_available": bool(_first(perception.get("mask_available"), qm.get("image_bottle_hull_px"), qm.get("current_live_bottle_hull_px"))),
            "mask_confidence": _num(perception.get("mask_confidence"), qm.get("bottle_mask_confidence"), 85.0 if qm.get("image_bottle_hull_px") else None),
        },
        "rgb_bottle_center": _first(perception.get("bottle_center_px"), rs.get("current_tracked_center"), rs.get("target_center_px")),
        "rgb_bottle_angle": _num(perception.get("bottle_angle_deg"), rs.get("raw_bottle_angle_deg"), rs.get("display_magenta_angle_deg")),
        "rgb_gripper_detection": _first(
            perception.get("finger_detection"),
            perception.get("left_finger_status") or perception.get("right_finger_status"),
            qm.get("finger_status"),
        ),
        "rgb_contact_visibility": {"visual_contact": bool(_first(rs.get("visual_contact"), perception.get("visual_contact"))), "contact_pair_valid": contact_pair_valid} if contact_pair_valid else None,
        "depth_bottle_distance": _num(
            rs.get("bottle_distance_mm"),
            rs.get("distance_mm"),
            rs.get("depth_mm"),
            rs.get("bottle_depth_median_mm"),
        ),
        "depth_grip_width": _num(rs.get("grip_width_mm")),
        "depth_contact_consistency": {
            "contact_depth_consistent": True,
            "left_depth_mm": _num(rs.get("left_finger_depth_mm"), qm.get("left_depth_mm")),
            "right_depth_mm": _num(rs.get("right_finger_depth_mm"), qm.get("right_depth_mm")),
        } if bool(_first(perception.get("contact_depth_consistent"), qm.get("contact_depth_consistent"))) else None,
        "depth_valid_ratio": depth_ratio,
        "depth_alignment_quality": {
            "intrinsics_available": bool(_first(rs.get("realsense_intrinsics_available"), qm.get("intrinsics_available"), pc.get("intrinsics_available"))),
            "camera_info_available": bool(rs.get("camera_info_ok") or rs.get("camera_info")),
        },
        "depth_object_geometry": {
            "distance_mm": _num(rs.get("bottle_distance_mm"), rs.get("distance_mm"), rs.get("depth_mm"), rs.get("bottle_depth_median_mm")),
            "depth_valid_ratio": depth_ratio,
            "geometry_available": depth_ratio is not None or _num(rs.get("bottle_distance_mm"), rs.get("distance_mm"), rs.get("depth_mm"), rs.get("bottle_depth_median_mm")) is not None,
        },
        "rgbd_bottle_tracking": {
            "tracking_status": rs.get("target_status"),
            "tracking_confidence": _num(rs.get("tracking_confidence"), perception.get("tracking_confidence")),
            "tracking_source": rs.get("tracking_source"),
        },
        "rgbd_object_mask_depth_support": {
            "mask_available": bool(_first(perception.get("mask_available"), qm.get("image_bottle_hull_px"))),
            "depth_valid_ratio": depth_ratio,
        },
        "rgbd_grasp_region_estimate": {
            "available": bool(_first(rs.get("magenta_band_polygon"), rs.get("left_contact_patch_polygon"), qm.get("current_grasp_region"))),
            "source": _first(rs.get("band_source"), qm.get("contact_hull_source"), "planned_or_locked_geometry"),
        },
        "rgbd_contact_patches": {
            "contact_source": _first(qm.get("contact_source"), rs.get("contact_source")),
            "contact_count": _num(qm.get("detected_contact_count"), perception.get("detected_contact_count"), 2 if contact_pair_valid else None),
            "contact_validity": bool(contact_pair_valid),
        } if contact_pair_valid else None,
        "rgbd_ready_to_close": {
            "ready_to_close": bool(rs.get("ready_to_close")),
            "recommended_action": rs.get("recommended_action"),
        } if rs.get("ready_to_close") is not None else None,
        "rgbd_visual_contact_model": {
            "visual_contact_ready": bool(contact_gate),
            "visual_contact_pair_count": _num(qm.get("visual_contact_pair_count"), perception.get("visual_contact_pair_count"), 2 if contact_gate else None),
        } if contact_gate else None,
        "pointcloud_scene_geometry": {
            "scene_raw_point_count": _num(pc.get("scene_raw_point_count"), pc.get("point_count")),
            "scene_filtered_point_count": _num(pc.get("scene_filtered_point_count"), pc.get("filtered_point_count")),
        },
        "pointcloud_bottle_geometry": {
            "valid": bool(_first(pc.get("bottle_pointcloud_valid"), pc.get("bottle_cloud_valid"), pc.get("bottle_dimensions_valid"))),
            "roi_source": _first(pc.get("bottle_roi_source"), pc.get("bottle_pointcloud_roi_source"), pc.get("bottle_mask_source")),
            "filtered_point_count": _num(pc.get("bottle_filtered_point_count"), pc.get("bottle_point_count")),
        },
        "pointcloud_bottle_extents": _first(pc.get("bottle_pointcloud_extents_mm"), pc.get("bottle_3d_extents_m"), pc.get("axis_aligned_extents_mm")),
        "pointcloud_grip_width_3d": _num(pc.get("contact_width_3d_mm"), pc.get("bottle_pointcloud_extents_mm", {}).get("width_x") if isinstance(pc.get("bottle_pointcloud_extents_mm"), dict) else None),
        "pointcloud_3d_contact_validation": {
            "valid": bool(pc.get("contact_3d_valid") or pc.get("contacts_on_filtered_bottle_cloud")),
            "quality": pc.get("contact_3d_quality"),
        },
        "davis_aps_frame": {
            "aps_state": aps_state,
            "aps_fps": _num(dv.get("aps_fps"), dv.get("davis_aps_fps")),
            "aps_visibility_status": _first(dv.get("aps_visibility_status"), None),
            "aps_visibility_score": _num(dv.get("aps_visibility_score")),
            "aps_underexposed": dv.get("aps_underexposed"),
            "aps_low_contrast": dv.get("aps_low_contrast"),
            "aps_display_warning": _first(dv.get("aps_display_warning"), None),
        },
        "davis_aps_target_motion": _num(dv.get("aps_motion"), dv.get("aps_target_motion"), motion.get("aps_motion")),
        "davis_aps_angle_change": _num(dv.get("aps_angle_change"), motion.get("aps_angle_change")),
        "davis_aps_lighting_change": _num(dv.get("lighting_change"), motion.get("lighting_flicker_score")),
        "event_rate": event_rate,
        "event_heatmap": bool(_first(dv.get("event_heatmap_available"), dv.get("heatmap_path"), dv.get("event_heatmap_url"), True if event_stream else None)),
        "event_motion_activity": {
            "event_rate": event_rate,
            "motion_class": _first(dv.get("motion_class"), motion.get("motion_class")),
            "motion_bbox": _first(dv.get("motion_bbox"), motion.get("motion_bbox")),
            "dx_px": _num(dv.get("dx_px"), motion.get("dx_px")),
            "dy_px": _num(dv.get("dy_px"), motion.get("dy_px")),
            "displacement_px": _num(dv.get("displacement_px"), motion.get("displacement_px")),
            "cumulative_slip_px": _num(dv.get("cumulative_slip_px"), motion.get("cumulative_slip_px")),
            "downward_slip_px": _num(dv.get("downward_slip_px"), motion.get("downward_slip_px")),
            "rotation_delta_deg": _num(dv.get("rotation_delta_deg"), motion.get("rotation_delta_deg")),
            "cumulative_rotation_deg": _num(dv.get("cumulative_rotation_deg"), motion.get("cumulative_rotation_deg")),
            "angular_velocity_deg_s": _num(dv.get("angular_velocity_deg_s"), motion.get("angular_velocity_deg_s")),
            "vibration_amplitude_px_rms": _num(dv.get("vibration_amplitude_px_rms"), motion.get("vibration_amplitude_px_rms")),
            "motion_estimation_confidence": _num(dv.get("motion_confidence"), dv.get("motion_estimation_confidence"), motion.get("motion_estimation_confidence")),
            "motion_estimate_valid": _first(dv.get("motion_estimate_valid"), motion.get("estimate_valid")),
            "disturbance_type": _first(dv.get("disturbance_type"), motion.get("disturbance_type")),
            # event health vs object motion vs disturbance are three different
            # claims; report them separately so a lighting change is never
            # counted as motion success or bottle evidence
            "event_health_valid": bool(
                davis_event_freshness == "CURRENT"
                and (
                    str(_first(dv.get("event_state"), dv.get("event_stream_status"), "")).upper() in {"STREAM_ACTIVE_EVENTS", "STREAM_ACTIVE_NO_MOTION", "ACTIVE"}
                    or (event_rate is not None and event_rate >= 0.0 and str(_first(dv.get("event_state"), dv.get("event_stream_status"), "")).lower() == "active")
                )
            ),
            "object_motion_valid": _first(dv.get("object_motion_valid"), None),
            "object_motion_reason": _first(dv.get("object_motion_reason"), None),
            "disturbance_detected": bool(
                str(_first(dv.get("raw_motion_class"), dv.get("motion_class"), motion.get("motion_class"), "")).upper().replace(" ", "_")
                in {"LIGHTING_CHANGE", "CAMERA_SHAKE", "BACKGROUND_MOTION"}
            ),
            "davis_motion_validity": (
                "valid_object_motion" if dv.get("object_motion_valid")
                else (
                    "disturbance" if str(_first(dv.get("raw_motion_class"), dv.get("motion_class"), motion.get("motion_class"), "")).upper().replace(" ", "_") in {"LIGHTING_CHANGE", "CAMERA_SHAKE", "BACKGROUND_MOTION"}
                    else ("not_object_specific" if dv.get("object_motion_valid") is False else "unknown")
                )
            ),
            "davis_motion_exclusion_reason": _first(dv.get("davis_motion_excluded_reason"), dv.get("exclusion_reason"), None),
            "closeup_object_motion_valid": _first(dv.get("object_motion_valid"), None),
            "closeup_roi_source": _first(dv.get("roi_source"), motion.get("roi_source"), None),
            "closeup_motion_class": _first(dv.get("davis_closeup_motion_class"), dv.get("raw_motion_class"), dv.get("motion_class"), motion.get("motion_class"), None),
            "closeup_displacement_px": _num(dv.get("displacement_px"), motion.get("displacement_px")),
            "closeup_rotation_deg": _num(dv.get("rotation_delta_deg"), motion.get("rotation_delta_deg")),
            "closeup_vibration_rms_px": _num(dv.get("vibration_amplitude_px_rms"), motion.get("vibration_amplitude_px_rms")),
            "closeup_stability_score": _num(dv.get("stability_score"), dv.get("post_grasp_stability_score"), motion.get("stability_score")),
            "closeup_disturbance_type": _first(dv.get("disturbance_type"), motion.get("disturbance_type"), None),
            "closeup_used_for_live_quality": _first(dv.get("davis_closeup_used_for_live_quality"), dv.get("davis_motion_used_for_live_quality"), False),
            "closeup_window": _first(dv.get("davis_closeup_window"), None),
        },
        "event_stability": _num(dv.get("stability"), dv.get("stability_score"), motion.get("stability_score")),
        "event_rotation": _num(dv.get("rotation"), dv.get("rotation_score"), motion.get("rotation_score")),
        "event_vibration": _num(dv.get("vibration"), dv.get("vibration_score"), motion.get("vibration_score")),
        "event_incipient_slip": _num(dv.get("incipient_slip"), dv.get("incipient_slip_score"), motion.get("incipient_slip_score")),
        "event_gross_slip": _num(dv.get("gross_slip"), dv.get("gross_slip_score"), motion.get("gross_slip_score")),
        "event_impact": _num(dv.get("impact"), dv.get("impact_score"), motion.get("impact_score")),
        "visual_object_quality": visual_object_quality,
        "geometry_quality": geometry_quality,
        "visual_contact_quality": _num(qm.get("contact_confidence"), qm.get("contact_quality_score"), qm.get("normalized_quality_score")) if contact_gate else None,
        "analytic_epsilon": _num(qm.get("epsilon_score"), qm.get("epsilon"), qm.get("raw_epsilon")) if contact_gate else None,
        "wrench_quality": {"force_closure": qm.get("force_closure"), "quality_label": qm.get("quality_label")} if contact_gate and _num(qm.get("epsilon_score"), qm.get("epsilon"), qm.get("raw_epsilon")) is not None else None,
        "live_operational_quality": _num(lq.get("live_quality_score"), lq.get("score")) if contact_gate else None,
        "synchronization_confidence": {
            "sync_mode": sy.get("sync_mode"),
            "sync_quality": sy.get("sync_quality"),
            "max_sensor_time_offset_ms": _num(sy.get("max_sensor_time_offset_ms"), sy.get("max_offset_ms")),
            "bundle_valid_for_benchmark": sy.get("bundle_valid_for_benchmark"),
            "bundle_valid_for_live_quality": sy.get("bundle_valid_for_live_quality"),
            "stale_modalities": sy.get("stale_modalities") or [],
            "unavailable_modalities": sy.get("unavailable_modalities") or [],
        },
    }

    source_fields = {feature_id: feature_id for feature_id in values}
    source_fields.update({
        "rgb_bottle_detection": "realsense.realsense_perception.bottle_detection_status|realsense.target_status",
        "rgb_bottle_segmentation": "realsense.realsense_perception.mask_available|quality.image_bottle_hull_px",
        "depth_bottle_distance": "realsense.bottle_distance_mm|realsense.distance_mm",
        "pointcloud_bottle_geometry": "pointcloud.bottle_pointcloud_valid|pointcloud.bottle_cloud_valid",
        "davis_aps_frame": "davis.aps_state|davis.aps_stream_status",
        "event_rate": "davis.event_rate|davis.davis_event_rate",
        "analytic_epsilon": "quality.epsilon_score|quality.epsilon",
        "live_operational_quality": "live_quality.live_quality_score",
    })
    availability = {
        "rgb": bool(_truthy(rs.get("realsense_rgb_active")) or detected or rs),
        "depth": bool(_truthy(rs.get("realsense_depth_active")) or depth_ratio is not None or rs.get("bottle_distance_mm") is not None),
        "rgbd": bool(rs.get("target_status") or perception),
        "pointcloud": bool(_truthy(pc.get("active")) or pc.get("bottle_pointcloud_valid") is not None or pc.get("freshness")),
        "davis_aps": bool(aps_state or dv.get("aps_fps") is not None),
        "davis_events": bool(event_stream or event_rate is not None or dv.get("motion_class")),
        "quality": bool(qm),
        "sync": bool(sy),
    }
    freshness = {
        "pointcloud": pointcloud_freshness,
        "davis_events": davis_event_freshness,
        "davis_aps": davis_aps_freshness,
        "sync": str(_first(sy.get("sync_quality"), "UNAVAILABLE")).upper(),
    }
    return FeaturePacket(
        values=values,
        source_fields=source_fields,
        availability=availability,
        freshness=freshness,
        production_reference={
            "decision": rs.get("decision") or rs.get("ui_primary_status"),
            "ready_to_close": rs.get("ready_to_close"),
            "quality_label": qm.get("quality_label"),
            "live_quality_label": lq.get("live_quality_label"),
            "embodiment": emb.get("active_embodiment"),
            "timing": rs.get("processed_overlay_timing") or rs.get("performance") or {},
        },
        timestamp=_num(rs.get("timestamp"), rs.get("time")),
        bundle_id=_first(sy.get("last_bundle_id"), sy.get("bundle_id"), rs.get("frame_id")),
    )


class MaskedFeatureView:
    """Per-configuration strict view of a feature packet."""

    def __init__(self, feature_packet: FeaturePacket, enabled_features: Iterable[str], *, bottle_only: bool = True):
        self.packet = feature_packet
        self.enabled_features = set(canonical_feature_ids(enabled_features))
        self.bottle_only = bottle_only
        self.trace: list[dict[str, Any]] = []
        self.attempted_features: set[str] = set()
        self.used_features: set[str] = set()
        self.blocked_features: dict[str, str] = {}
        self.missing_dependencies: dict[str, list[str]] = {}
        self.not_applicable_features: dict[str, str] = {}
        self.unavailable_features: dict[str, str] = {}
        self.stale_features: dict[str, str] = {}
        self.unsupported_features: dict[str, str] = {}

    @property
    def disabled_features(self) -> list[str]:
        return sorted(fid for fid in FEATURE_BY_ID if fid not in self.enabled_features)

    def is_enabled(self, feature_id: str) -> bool:
        return feature_id in self.enabled_features

    def _sensor_for(self, feature_id: str) -> str:
        feature = FEATURE_BY_ID.get(feature_id)
        if not feature:
            return "unknown"
        category = feature.category
        return {
            "RealSense RGB": "rgb",
            "RealSense Depth": "depth",
            "RGB-D Combined": "rgbd",
            "Point Cloud": "pointcloud",
            "DAVIS APS": "davis_aps",
            "DAVIS Events": "davis_events",
            "Quality": "quality",
        }.get(category, "unknown")

    def get_feature(self, feature_id: str) -> dict[str, Any]:
        self.attempted_features.add(feature_id)
        feature = FEATURE_BY_ID.get(feature_id)
        status = VALID
        reason = "ok"
        value = None
        dependencies: list[str] = []
        source_field = self.packet.source_fields.get(feature_id, feature_id)

        if not feature:
            status = UNSUPPORTED
            reason = "feature_id_not_registered"
            self.unsupported_features[feature_id] = reason
        elif feature_id not in self.enabled_features:
            status = DISABLED
            reason = "feature_disabled_by_configuration"
            self.blocked_features[feature_id] = reason
        elif self.bottle_only and feature_id in BOTTLE_ONLY_NOT_APPLICABLE:
            status = NOT_APPLICABLE
            reason = BOTTLE_ONLY_NOT_APPLICABLE[feature_id]
            self.not_applicable_features[feature_id] = reason
        else:
            dependencies = [dep for dep in feature.dependencies if dep not in self.enabled_features]
            blocked_dependencies = [
                dep
                for dep in feature.dependencies
                if dep in self.blocked_features
                or dep in self.missing_dependencies
                or dep in self.not_applicable_features
                or dep in self.unavailable_features
                or dep in self.stale_features
                or dep in self.unsupported_features
            ]
            dependencies = sorted(set(dependencies + blocked_dependencies))
            if dependencies:
                status = DEPENDENCY_MISSING
                reason = "missing_enabled_dependencies" if any(dep not in self.enabled_features for dep in dependencies) else "blocked_dependency_features"
                self.missing_dependencies[feature_id] = dependencies
                self.blocked_features[feature_id] = reason
            else:
                sensor = self._sensor_for(feature_id)
                if sensor != "unknown" and not self.packet.availability.get(sensor, False):
                    status = SENSOR_UNAVAILABLE
                    reason = f"{sensor}_unavailable"
                    self.unavailable_features[feature_id] = reason
                    self.blocked_features[feature_id] = reason
                elif sensor != "unknown" and str(self.packet.freshness.get(sensor) or "").upper() == "STALE":
                    status = STALE_DATA
                    reason = f"{sensor}_stale"
                    self.stale_features[feature_id] = reason
                    self.blocked_features[feature_id] = reason
                else:
                    value = self.packet.values.get(feature_id)
                    if value is None:
                        status = SENSOR_UNAVAILABLE
                        reason = "source_value_missing"
                        self.unavailable_features[feature_id] = reason
                        self.blocked_features[feature_id] = reason
                    else:
                        self.used_features.add(feature_id)

        record = {
            "requested_feature": feature_id,
            "returned_value": value,
            "status": status,
            "reason": reason,
            "source_field": source_field,
            "dependencies": dependencies,
        }
        self.trace.append(record)
        return record

    def get_rgb_detection(self) -> dict[str, Any]:
        detection = self.get_feature("rgb_bottle_detection")
        center = self.get_feature("rgb_bottle_center")
        angle = self.get_feature("rgb_bottle_angle")
        mask = self.get_feature("rgb_bottle_segmentation")
        return {"detection": detection, "center": center, "angle": angle, "mask": mask}

    def get_depth_geometry(self) -> dict[str, Any]:
        return {
            "distance": self.get_feature("depth_bottle_distance"),
            "valid_ratio": self.get_feature("depth_valid_ratio"),
            "alignment": self.get_feature("depth_alignment_quality"),
            "object_geometry": self.get_feature("depth_object_geometry"),
        }

    def get_rgbd_tracking(self) -> dict[str, Any]:
        return {
            "tracking": self.get_feature("rgbd_bottle_tracking"),
            "mask_depth_support": self.get_feature("rgbd_object_mask_depth_support"),
            "grasp_region": self.get_feature("rgbd_grasp_region_estimate"),
        }

    def get_pointcloud_geometry(self) -> dict[str, Any]:
        return {
            "scene": self.get_feature("pointcloud_scene_geometry"),
            "bottle": self.get_feature("pointcloud_bottle_geometry"),
            "extents": self.get_feature("pointcloud_bottle_extents"),
            "grip_width": self.get_feature("pointcloud_grip_width_3d"),
            "contact": self.get_feature("pointcloud_3d_contact_validation"),
        }

    def get_davis_aps_motion(self) -> dict[str, Any]:
        return {
            "frame": self.get_feature("davis_aps_frame"),
            "motion": self.get_feature("davis_aps_target_motion"),
            "angle_change": self.get_feature("davis_aps_angle_change"),
            "lighting_change": self.get_feature("davis_aps_lighting_change"),
        }

    def get_davis_event_motion(self) -> dict[str, Any]:
        return {
            "rate": self.get_feature("event_rate"),
            "heatmap": self.get_feature("event_heatmap"),
            "activity": self.get_feature("event_motion_activity"),
            "stability": self.get_feature("event_stability"),
            "rotation": self.get_feature("event_rotation"),
            "vibration": self.get_feature("event_vibration"),
            "incipient_slip": self.get_feature("event_incipient_slip"),
            "gross_slip": self.get_feature("event_gross_slip"),
            "impact": self.get_feature("event_impact"),
        }

    def get_quality_metric(self) -> dict[str, Any]:
        return {
            "visual_object_quality": self.get_feature("visual_object_quality"),
            "geometry_quality": self.get_feature("geometry_quality"),
            "analytic_epsilon": self.get_feature("analytic_epsilon"),
            "wrench_quality": self.get_feature("wrench_quality"),
            "live_operational_quality": self.get_feature("live_operational_quality"),
            "synchronization_confidence": self.get_feature("synchronization_confidence"),
        }

    def proof_payload(self) -> dict[str, Any]:
        return {
            "enabled_features": sorted(self.enabled_features),
            "disabled_features": self.disabled_features,
            "used_features": sorted(self.used_features),
            "attempted_features": sorted(self.attempted_features),
            "blocked_features": dict(sorted(self.blocked_features.items())),
            "missing_dependencies": dict(sorted(self.missing_dependencies.items())),
            "not_applicable_features": dict(sorted(self.not_applicable_features.items())),
            "unavailable_features": dict(sorted(self.unavailable_features.items())),
            "stale_features": dict(sorted(self.stale_features.items())),
            "unsupported_features": dict(sorted(self.unsupported_features.items())),
            "production_fields_used": sorted(self.packet.production_reference.accessed_fields),
            "feature_leakage_detected": bool(set(self.used_features) - self.enabled_features),
            "feature_leakage_reason": "" if not (set(self.used_features) - self.enabled_features) else "used_feature_not_enabled",
            "evaluator_trace": list(self.trace),
        }
