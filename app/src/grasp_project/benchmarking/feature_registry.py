"""Registry of benchmarkable sensing features.

These entries describe existing signals that the dashboard can compare. The
registry is metadata only: it does not start sensors, alter tracking, or
recompute production grasp logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkFeature:
    feature_id: str
    display_name: str
    category: str
    provider: str
    dependencies: tuple[str, ...]
    output_fields: tuple[str, ...]
    stable_or_experimental: str
    enabled_by_default: bool
    description: str
    limitation: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dependencies"] = list(self.dependencies)
        payload["output_fields"] = list(self.output_fields)
        return payload


FEATURE_CATEGORIES: tuple[str, ...] = (
    "RealSense RGB",
    "RealSense Depth",
    "RGB-D Combined",
    "Point Cloud",
    "DAVIS APS",
    "DAVIS Events",
    "Quality",
)


FEATURES: tuple[BenchmarkFeature, ...] = (
    BenchmarkFeature("rgb_bottle_detection", "Bottle detection", "RealSense RGB", "RealSense", (), ("bottle_detected", "detection_confidence", "bbox_xyxy"), "stable", True, "RGB appearance-based bottle candidate evidence from the existing pipeline.", "Cannot validate distance or contact depth by itself."),
    BenchmarkFeature("rgb_bottle_segmentation", "Segmentation", "RealSense RGB", "RealSense", ("rgb_bottle_detection",), ("mask_available", "mask_confidence"), "stable", True, "Image-space contour or mask evidence for the bottle body.", "Unavailable when the current detector falls back to bbox-only evidence."),
    BenchmarkFeature("rgb_bottle_center", "Bottle center", "RealSense RGB", "RealSense", ("rgb_bottle_detection",), ("center_px",), "stable", True, "Image-space bottle center from the current perception result.", "2D center alone is not a grasp-depth measurement."),
    BenchmarkFeature("rgb_bottle_angle", "Bottle angle", "RealSense RGB", "RealSense", ("rgb_bottle_detection",), ("angle_deg",), "stable", True, "Bottle orientation estimate from existing contour/oriented-box outputs.", "Angle may be unavailable for bbox-only fallback."),
    BenchmarkFeature("rgb_gripper_detection", "Gripper detection", "RealSense RGB", "RealSense", ("rgb_bottle_detection",), ("left_finger_status", "right_finger_status"), "stable", True, "Existing image-space finger visibility and identity evidence.", "Does not imply physical force contact."),
    BenchmarkFeature("rgb_contact_visibility", "Contact visibility", "RealSense RGB", "RealSense", ("rgb_gripper_detection",), ("visual_contact", "contact_pair_valid"), "stable", True, "Visual evidence that contact patches are visible near the bottle.", "Requires depth/geometry to become a validated contact pair."),
    BenchmarkFeature("depth_bottle_distance", "Distance", "RealSense Depth", "RealSense", (), ("distance_mm", "bottle_depth_mm"), "stable", True, "Aligned depth estimate for the bottle or target region.", "Depth-only cannot identify a stationary bottle appearance without a prior ROI or segmentation."),
    BenchmarkFeature("depth_grip_width", "Grip width", "RealSense Depth", "RealSense", ("depth_bottle_distance", "depth_valid_ratio"), ("grip_width_mm",), "stable", True, "Grip width or separation derived from existing depth/contact geometry.", "Depends on valid contact geometry or a locked target."),
    BenchmarkFeature("depth_contact_consistency", "Contact depth", "RealSense Depth", "RealSense", ("depth_bottle_distance",), ("contact_depth_consistent", "left_finger_depth_mm", "right_finger_depth_mm"), "stable", True, "Depth agreement for left/right contact evidence.", "Invalid when one contact depth is missing."),
    BenchmarkFeature("depth_valid_ratio", "Depth validity", "RealSense Depth", "RealSense", (), ("depth_valid_ratio", "bottle_depth_valid_ratio"), "stable", True, "Valid-depth ratio in the current bottle/contact region.", "Does not identify object class by itself."),
    BenchmarkFeature("depth_alignment_quality", "Depth alignment quality", "RealSense Depth", "RealSense", ("depth_valid_ratio",), ("realsense_intrinsics_available", "snapshot_consistent"), "stable", True, "RGB/depth/intrinsics availability and frame consistency.", "Only reports existing alignment health."),
    BenchmarkFeature("depth_object_geometry", "Depth object geometry", "RealSense Depth", "RealSense", ("depth_valid_ratio",), ("distance_mm", "depth_valid_ratio", "geometry_available"), "stable", True, "Bottle-only depth geometry support from aligned depth values.", "Depth geometry alone does not identify a stationary bottle."),
    BenchmarkFeature("rgbd_bottle_tracking", "Bottle tracking", "RGB-D Combined", "RealSense", ("rgb_bottle_detection", "depth_bottle_distance"), ("tracking_status", "tracking_confidence", "tracking_source"), "stable", True, "Existing RGB-D post-lock tracking status.", "Benchmark masks do not alter the tracker source."),
    BenchmarkFeature("rgbd_object_mask_depth_support", "Mask + depth support", "RGB-D Combined", "RealSense", ("rgb_bottle_segmentation", "depth_valid_ratio"), ("mask_available", "depth_valid_ratio"), "stable", True, "Bottle-only check that image mask support and depth support exist together.", "This is object support, not physical gripper contact."),
    BenchmarkFeature("rgbd_grasp_region_estimate", "Planned grasp region", "RGB-D Combined", "RealSense", ("rgbd_bottle_tracking",), ("grasp_region_available", "grasp_region_source"), "stable", True, "Planned/current grasp region estimate for bottle-only visualization.", "Does not prove contact or closure."),
    BenchmarkFeature("rgbd_contact_patches", "Contact patches", "RGB-D Combined", "RealSense", ("rgbd_bottle_tracking", "depth_contact_consistency"), ("contact_source", "contact_count", "contact_validity"), "stable", True, "Current left/right contact patch evidence from the production pipeline.", "Requires the production contact model to have current contact inputs."),
    BenchmarkFeature("rgbd_ready_to_close", "Ready to close", "RGB-D Combined", "RealSense", ("rgbd_contact_patches", "depth_grip_width"), ("ready_to_close", "recommended_action"), "stable", True, "Read-only production ready-to-close outcome.", "Benchmark cards never control the robot."),
    BenchmarkFeature("rgbd_visual_contact_model", "Visual contact model", "RGB-D Combined", "RealSense", ("rgbd_contact_patches", "rgb_contact_visibility"), ("visual_contact_ready", "visual_contact_pair_count"), "stable", True, "Validated visual/contact model result from existing outputs.", "No force or pressure sensing is implied."),
    BenchmarkFeature("pointcloud_scene_geometry", "Scene geometry", "Point Cloud", "RealSense point cloud", ("depth_valid_ratio",), ("scene_raw_point_count", "scene_filtered_point_count"), "experimental", False, "Current sampled scene point cloud from existing RGB-D data.", "Requires RealSense depth and intrinsics."),
    BenchmarkFeature("pointcloud_bottle_geometry", "Bottle geometry", "Point Cloud", "RealSense point cloud", ("pointcloud_scene_geometry", "rgbd_bottle_tracking"), ("bottle_filtered_point_count", "bottle_dimensions_valid"), "experimental", False, "Filtered bottle crop/geometry from current point-cloud outputs.", "Quality depends on ROI/mask freshness."),
    BenchmarkFeature("pointcloud_3d_contact_validation", "3D contact validation", "Point Cloud", "RealSense point cloud", ("pointcloud_bottle_geometry", "rgbd_contact_patches"), ("contact_3d_quality", "contact_points_on_bottle_cloud"), "experimental", False, "Checks whether contact patches are consistent with the bottle cloud.", "Stale point clouds are excluded."),
    BenchmarkFeature("pointcloud_grip_width_3d", "3D width estimate", "Point Cloud", "RealSense point cloud", ("pointcloud_bottle_geometry",), ("contact_width_3d_mm", "bottle_width_mm"), "experimental", False, "3D bottle/contact width estimate when current point-cloud geometry supports it.", "Contact-specific width is unavailable without contact points."),
    BenchmarkFeature("pointcloud_bottle_extents", "Bottle extents", "Point Cloud", "RealSense point cloud", ("pointcloud_bottle_geometry",), ("bottle_pointcloud_extents_mm",), "experimental", False, "Metric bottle dimensions from filtered point cloud.", "May degrade with occlusion or bbox fallback."),
    BenchmarkFeature("multiview_reconstruction_support", "Multi-view support", "Point Cloud", "Multi-view", ("pointcloud_bottle_geometry",), ("scan_status", "accepted_keyframes", "coverage_percent"), "experimental", False, "Multi-view scan state and reconstruction support.", "Depends on active scan state and valid point-cloud input."),
    BenchmarkFeature("davis_aps_frame", "APS frame", "DAVIS APS", "DAVIS", (), ("aps_state", "aps_fps"), "experimental", False, "Latest APS frame state from the dashboard-owned DAVIS backend.", "Not treated as RGB unless color is verified."),
    BenchmarkFeature("davis_aps_target_motion", "APS motion", "DAVIS APS", "DAVIS", ("davis_aps_frame",), ("aps_motion",), "experimental", False, "Frame-based DAVIS APS motion context when available.", "Lower temporal resolution than event stream."),
    BenchmarkFeature("davis_aps_angle_change", "APS angle change", "DAVIS APS", "DAVIS", ("davis_aps_target_motion",), ("aps_angle_change",), "experimental", False, "APS-derived angle/motion context when available.", "Not a substitute for RealSense geometry."),
    BenchmarkFeature("davis_aps_lighting_change", "Lighting change", "DAVIS APS", "DAVIS", ("davis_aps_frame",), ("lighting_change",), "experimental", False, "APS context for lighting-change warnings.", "Does not classify grasp stability alone."),
    BenchmarkFeature("event_rate", "Event rate", "DAVIS Events", "DAVIS", (), ("event_rate",), "stable", False, "DAVIS event activity from the existing motion backend.", "Stationary scenes may correctly show no events."),
    BenchmarkFeature("event_heatmap", "Event heatmap", "DAVIS Events", "DAVIS", ("event_rate",), ("event_heatmap",), "stable", False, "Recent event heatmap from the dashboard-owned DAVIS backend.", "Visualization only."),
    BenchmarkFeature("event_motion_activity", "Motion activity", "DAVIS Events", "DAVIS", ("event_rate",), ("event_rate", "motion_class", "motion_bbox"), "stable", False, "DAVIS event-rate and motion-activity summary.", "Motion activity does not identify a stationary bottle."),
    BenchmarkFeature("event_impact", "Impact", "DAVIS Events", "DAVIS", ("event_rate",), ("impact_score",), "stable", False, "Impact-like transient motion score.", "Benchmark-only; does not recompute epsilon."),
    BenchmarkFeature("event_incipient_slip", "Incipient slip", "DAVIS Events", "DAVIS", ("event_rate",), ("incipient_slip",), "stable", False, "Early slip/micro-motion risk score.", "Requires calibrated baseline for stronger interpretation."),
    BenchmarkFeature("event_gross_slip", "Gross slip", "DAVIS Events", "DAVIS", ("event_rate",), ("gross_slip",), "stable", False, "Gross slip risk score.", "Requires active event stream and valid ROI/full-frame context."),
    BenchmarkFeature("event_rotation", "Rotation", "DAVIS Events", "DAVIS", ("event_rate",), ("rotation",), "stable", False, "Rotation/spin-like event pattern score.", "Camera shake can be a confounder if not gated."),
    BenchmarkFeature("event_vibration", "Vibration", "DAVIS Events", "DAVIS", ("event_rate",), ("vibration",), "stable", False, "High-frequency event disturbance score.", "Lighting flicker must be interpreted separately."),
    BenchmarkFeature("event_stability", "Stability", "DAVIS Events", "DAVIS", ("event_rate",), ("stability", "decision"), "stable", False, "DAVIS post-grasp stability summary.", "Does not control production close/lift by itself."),
    BenchmarkFeature("visual_object_quality", "Bottle visual quality", "Quality", "Quality", ("rgb_bottle_detection", "rgb_bottle_segmentation", "rgb_bottle_angle"), ("mask_stability", "center_stability", "angle_stability", "profile_agreement"), "stable", True, "Bottle-only visual quality from mask, center, angle, and profile agreement.", "Not a contact or force-closure metric."),
    BenchmarkFeature("geometry_quality", "Bottle geometry quality", "Quality", "Quality", ("depth_valid_ratio", "depth_object_geometry"), ("depth_validity", "distance_stability", "dimension_plausibility", "pointcloud_freshness"), "stable", True, "Bottle-only geometry quality from depth and optional point-cloud support.", "Not a contact or force-closure metric."),
    BenchmarkFeature("visual_contact_quality", "Visual contact quality", "Quality", "Quality", ("rgbd_visual_contact_model",), ("contact_confidence", "contact_quality_gate_status"), "stable", True, "Contact quality gate from current visual/depth evidence.", "Cannot replace force sensing."),
    BenchmarkFeature("analytic_epsilon", "Epsilon", "Quality", "Quality", ("visual_contact_quality",), ("epsilon_score", "epsilon_value"), "stable", True, "Read-only epsilon/wrench result from the current matrix output.", "Unavailable when contact geometry is missing or disabled."),
    BenchmarkFeature("wrench_quality", "Wrench quality", "Quality", "Quality", ("analytic_epsilon",), ("force_closure", "quality_label"), "stable", True, "Current wrench-space quality label and force-closure status.", "No epsilon math is changed by benchmarking."),
    BenchmarkFeature("live_operational_quality", "Live quality", "Quality", "Quality", ("visual_contact_quality", "synchronization_confidence",), ("live_quality_score", "live_quality_label"), "experimental", True, "Operational quality monitor result from existing status outputs.", "Event penalties are included only when event features are enabled."),
    BenchmarkFeature("synchronization_confidence", "Sync confidence", "Quality", "Synchronization", (), ("sync_quality", "max_sensor_time_offset_ms"), "stable", True, "Software timestamp bundle/sync quality.", "No hardware trigger is implied."),
)


FEATURE_BY_ID = {feature.feature_id: feature for feature in FEATURES}

PRESET_DEFINITIONS: dict[str, dict] = {
    "rgb_only": {"display_name": "RGB Only", "feature_ids": [f.feature_id for f in FEATURES if f.category == "RealSense RGB"]},
    "depth_only": {"display_name": "Depth Only", "feature_ids": [f.feature_id for f in FEATURES if f.category == "RealSense Depth"]},
    "rgb_depth": {"display_name": "RGB + Depth", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"RealSense RGB", "RealSense Depth", "RGB-D Combined"}]},
    "pointcloud_only": {"display_name": "Point Cloud Only", "feature_ids": [f.feature_id for f in FEATURES if f.category == "Point Cloud"]},
    "davis_aps_only": {"display_name": "DAVIS APS Only", "feature_ids": [f.feature_id for f in FEATURES if f.category == "DAVIS APS"]},
    "davis_events_only": {"display_name": "DAVIS Events Only", "feature_ids": [f.feature_id for f in FEATURES if f.category == "DAVIS Events"]},
    "aps_events": {"display_name": "DAVIS APS + Events", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"DAVIS APS", "DAVIS Events"}]},
    "rgbd_pointcloud": {"display_name": "RGB-D + Point Cloud", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"RealSense RGB", "RealSense Depth", "RGB-D Combined", "Point Cloud"}]},
    "rgbd_davis_events": {"display_name": "RGB-D + DAVIS Events", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"RealSense RGB", "RealSense Depth", "RGB-D Combined", "DAVIS Events"}]},
    "rgbd_pointcloud_davis_events": {"display_name": "RGB-D + Point Cloud + DAVIS Events", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"RealSense RGB", "RealSense Depth", "RGB-D Combined", "Point Cloud", "DAVIS Events"}]},
    "pointcloud_davis_events": {"display_name": "Point Cloud + DAVIS Events", "feature_ids": [f.feature_id for f in FEATURES if f.category in {"Point Cloud", "DAVIS Events"}]},
    "full_multimodal": {"display_name": "Full Multimodal", "feature_ids": [f.feature_id for f in FEATURES]},
    "custom": {"display_name": "Custom", "feature_ids": []},
}


def get_feature(feature_id: str) -> BenchmarkFeature | None:
    return FEATURE_BY_ID.get(feature_id)


def all_features() -> list[BenchmarkFeature]:
    return list(FEATURES)


def features_by_category() -> dict[str, list[dict]]:
    return {
        category: [feature.to_dict() for feature in FEATURES if feature.category == category]
        for category in FEATURE_CATEGORIES
    }


def preset_feature_ids(preset_id: str) -> list[str]:
    return list((PRESET_DEFINITIONS.get(preset_id) or {}).get("feature_ids") or [])


def preset_payload() -> list[dict]:
    return [
        {"preset_id": preset_id, **payload, "feature_ids": list(payload.get("feature_ids") or [])}
        for preset_id, payload in PRESET_DEFINITIONS.items()
    ]


def feature_registry_payload() -> dict:
    return {
        "categories": list(FEATURE_CATEGORIES),
        "features": [feature.to_dict() for feature in FEATURES],
        "features_by_category": features_by_category(),
        "presets": preset_payload(),
    }


def default_comparison_configurations() -> list[dict]:
    defaults = ("rgb_only", "rgb_depth", "davis_events_only", "full_multimodal")
    return [
        {
            "configuration_id": preset_id,
            "display_name": PRESET_DEFINITIONS[preset_id]["display_name"],
            "preset_id": preset_id,
            "feature_ids": preset_feature_ids(preset_id),
        }
        for preset_id in defaults
    ]


def _with_dependencies(feature_ids: Iterable[str]) -> list[str]:
    selected = set(canonical_feature_ids(feature_ids))
    changed = True
    while changed:
        changed = False
        for feature_id in list(selected):
            for dependency in FEATURE_BY_ID[feature_id].dependencies:
                if dependency in FEATURE_BY_ID and dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return canonical_feature_ids(selected)


def default_run_configurations(duration_sec: float | int | None = None) -> list[dict]:
    defaults = (
        "rgb_only",
        "depth_only",
        "rgb_depth",
        "pointcloud_only",
        "rgbd_pointcloud",
        "davis_aps_only",
        "davis_events_only",
        "aps_events",
        "rgbd_davis_events",
        "rgbd_pointcloud_davis_events",
        "full_multimodal",
    )
    configs = []
    for preset_id in defaults:
        preset = PRESET_DEFINITIONS[preset_id]
        feature_ids = _with_dependencies(preset_feature_ids(preset_id))
        config = {
            "configuration_id": preset_id,
            "display_name": preset["display_name"],
            "preset_id": preset_id,
            "feature_ids": feature_ids,
            "enabled_features": feature_ids,
            "disabled_features": [feature.feature_id for feature in FEATURES if feature.feature_id not in set(feature_ids)],
            "dependencies": sorted({dep for fid in feature_ids for dep in FEATURE_BY_ID[fid].dependencies}),
            "expected_outputs": sorted({field for fid in feature_ids for field in FEATURE_BY_ID[fid].output_fields}),
            "limitations": [FEATURE_BY_ID[fid].limitation for fid in feature_ids if FEATURE_BY_ID[fid].limitation],
        }
        if duration_sec is not None:
            config["duration_sec"] = float(duration_sec)
        configs.append(config)
    return configs


def canonical_feature_ids(feature_ids: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for feature_id in feature_ids:
        if feature_id in FEATURE_BY_ID and feature_id not in seen:
            seen.add(feature_id)
            ordered.append(feature_id)
    return ordered
