"""Deterministic component replay on verified observations; no camera or labels."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np

from grasp_project.davis.event_stream import EventPostGraspMonitor
from grasp_project.davis.motion_estimation import DavisMotionEstimator
from grasp_project.davis.motion_gui import select_event_analysis_window
from grasp_project.quality.live_quality_monitor import (
    BaselineGraspQuality,
    LiveGraspObservation,
    LiveGraspQualityMonitor,
)
from grasp_project.recording.rgbd_trace import replay_rgbd_trace
from grasp_project.recording.session import _file_digest, read_recording

ABLATIONS = {
    "reference": "Recorded inputs and production policy",
    "events_disabled": "DAVIS unavailable; other recorded inputs unchanged",
    "sync_diagnostics_removed": "Remove cross-camera sync diagnostics; preserve native event freshness",
    "phase_conditioning_removed": "Apply production post-grasp monitoring branch to every recorded phase; offline control only",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _implementation():
    package = Path(__file__).resolve().parents[1]
    owners = (
        "recording/replay.py",
        "davis/event_stream.py",
        "davis/motion_estimation.py",
        "davis/motion_gui.py",
        "quality/live_quality_monitor.py",
    )
    return {
        "source_sha256": {owner: _file_digest(package / owner) for owner in owners},
        "python": sys.version,
        "platform": sys.platform,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def _ablate(observation, name):
    if name == "events_disabled":
        return replace(
            observation,
            event_state="UNAVAILABLE",
            davis_event_freshness="UNAVAILABLE",
            davis_events_valid_for_live_quality=False,
            davis_motion_usable_for_grasp=False,
        )
    if name == "sync_diagnostics_removed":
        return replace(
            observation,
            sync_quality=None,
            davis_realsense_sync_quality=None,
            sync_delta_sec=None,
            sync_age_sec=None,
        )
    if name == "phase_conditioning_removed":
        return replace(
            observation, grasp_stage="POST_GRASP_MONITORING", post_grasp_active=True
        )
    return observation


def replay_recording(recording_dir, *, ablation="reference") -> dict:
    """Replay native DAVIS normalization/estimation and recorded quality inputs.

    A raw ROS image is preserved but not fed through the stateful RGB-D tracker.
    UUID/wall-clock diagnostic fields from the estimator are replaced with the
    recording's stable ID/arrival; numeric inference and classification are not
    changed. This does not reproduce dashboard scheduling or wall-clock speed.
    """
    if ablation not in ABLATIONS:
        raise ValueError("unknown replay ablation")
    recording = read_recording(recording_dir)
    manifest = recording.manifest
    config = manifest["config"]
    quality_configs = {}
    for record in recording.records:
        if record.get("kind") == "live_quality_inputs":
            effective = record["metadata"].get("effective_config")
            if not isinstance(effective, dict):
                raise ValueError(
                    "quality inputs need an effective configuration object"
                )
            quality_configs[_digest(effective)] = effective
    if len(quality_configs) > 1:
        raise ValueError(
            "quality configuration changed within recording; use separate frozen configurations"
        )
    motion_config = config.get("davis_motion", {})
    estimator = DavisMotionEstimator(motion_config)
    mapper = object.__new__(
        EventPostGraspMonitor
    )  # Only pure normalization; never initialize/open a backend.
    buffer = np.empty((0, 4), dtype=float)
    quality_monitor = None
    quality_gap = False
    quality_epoch = None
    quality_input_index = None
    normalization_epoch = None
    outputs, unsupported = [], []
    pending_gap = False
    configuration = {
        "ablation": ablation,
        "description": ABLATIONS[ablation],
        "recorded_config_sha256": manifest["config_sha256"],
        "effective_quality_config": next(iter(quality_configs.values()), None),
        "effective_davis_motion_config": estimator.config,
        "effective_event_window_selection": config.get("event_window_selection"),
        "replay_implementation": _implementation(),
        "replay_schema": "grasp_component_replay_v1",
    }
    config_hash = _digest(configuration)
    for record in recording.records:
        if record["record_type"] == "drop":
            accounting_notice = record["count"] is None and record["reason"] in {
                "backend_packet_loss_not_observable",
                "ros_header_has_no_source_sequence",
            }
            if (
                record["stream"] in {"davis_events", "davis_poll"}
                and not accounting_notice
            ):
                pending_gap = True
                buffer = np.empty((0, 4), dtype=float)
                estimator.reset()
            if record["stream"] == "live_quality_inputs" and not accounting_notice:
                quality_monitor = None
                quality_gap = True
            continue
        kind, metadata = record["kind"], record["metadata"]
        common = {
            "trial_id": record["trial_id"],
            "observation_id": record["observation_id"],
            "configuration_id": ablation,
            "config_sha256": config_hash,
            "kind": kind,
        }
        if kind == "davis_poll":
            epoch = metadata.get("normalization_epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("DAVIS replay needs a recorded normalization epoch")
            if normalization_epoch is not None and epoch != normalization_epoch:
                if epoch < normalization_epoch:
                    raise ValueError("normalization epoch decreased")
                mapper = object.__new__(EventPostGraspMonitor)
                buffer = np.empty((0, 4), dtype=float)
                estimator.reset()
                pending_gap = False
            normalization_epoch = epoch
            if (
                record["native_clock"]["domain"] != "davis_device"
                or record["native_clock"]["unit"] != "us"
            ):
                raise ValueError(
                    "DAVIS replay requires explicitly declared native device microseconds"
                )
            native = record["arrays"].get("events")
            if native is None or native.ndim != 2 or native.shape[1] < 4:
                raise ValueError(
                    "davis_poll must contain at least four native event columns"
                )
            arrival = metadata.get(
                "normalization_reference_monotonic_sec",
                record["host_arrival_monotonic_ns"] / 1_000_000_000.0,
            )
            if (
                not isinstance(arrival, (int, float))
                or isinstance(arrival, bool)
                or not np.isfinite(arrival)
            ):
                raise ValueError(
                    "recorded normalization reference must be finite host monotonic seconds"
                )
            if record["source_sequence_gap"]:
                pending_gap = True
                buffer = np.empty((0, 4), dtype=float)
                estimator.reset()
            previous_resets = getattr(mapper, "event_timestamp_reanchor_count", 0)
            normalized = mapper.normalize_native_event_array(native[:, :4], arrival)
            if getattr(mapper, "event_timestamp_reanchor_count", 0) != previous_resets:
                pending_gap = True
                buffer = np.empty((0, 4), dtype=float)
                estimator.reset()
            if len(normalized):
                buffer = np.vstack([buffer, normalized])
            # Keep all recorded events needed by both current/previous adaptive windows.
            roi = metadata.get("roi_xywh")
            window = config.get("event_window_selection")
            if not isinstance(window, dict) or set(window) != {
                "primary_window_sec",
                "fallback_window_sec",
                "min_primary_events",
            }:
                raise ValueError(
                    "DAVIS replay needs all recorded adaptive window settings"
                )
            selected, duration, mode, end = select_event_analysis_window(
                buffer, arrival, roi, **window
            )
            estimate = estimator.estimate(
                buffer,
                now=arrival,
                analysis_end=end,
                motion_window_ms=duration * 1000,
                roi_xywh=roi,
                roi_source="recorded_roi" if roi else "full_frame",
            )
            raw = estimate.to_dict()
            raw["estimate_id"] = f"{record['recording_id']}:{record['observation_id']}"
            raw["host_timestamp"] = record["host_arrival_epoch_ns"] / 1_000_000_000.0
            valid = (
                estimate.estimate_valid is True
                and not pending_gap
                and ablation != "events_disabled"
            )
            reasons = (
                []
                if valid
                else list(
                    dict.fromkeys(
                        filter(
                            None,
                            [
                                estimate.invalid_reason,
                                "recorded_observation_gap" if pending_gap else None,
                                "events_disabled"
                                if ablation == "events_disabled"
                                else None,
                            ],
                        )
                    )
                )
            )
            if not valid and not reasons:
                reasons = ["motion_estimate_not_valid"]
            outputs.append(
                {
                    **common,
                    "prediction": {
                        "label": estimate.motion_class if valid else None,
                        "score": estimate.motion_estimation_confidence
                        if valid
                        else None,
                        "decision": None,
                        "valid": valid,
                        "reasons": reasons,
                    },
                    "evidence": {
                        "davis_events": {
                            "status": "valid" if valid else "unavailable",
                            "reasons": reasons,
                        }
                    },
                    "raw_estimate": raw,
                    "analysis_window": {
                        "seconds": duration,
                        "selection": mode,
                        "selected_event_count": len(selected),
                    },
                }
            )
        elif kind == "live_quality_inputs":
            epoch = metadata.get("state_epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError("quality replay needs a recorded state epoch")
            index = metadata.get("epoch_input_index")
            if type(index) is not int or index < 0:
                raise ValueError("quality replay needs a recorded epoch input index")
            if quality_epoch is None and (
                index != 0
                or metadata.get("initial_state")
                not in {"constructor_reset", "epoch_reset"}
            ):
                raise ValueError("quality recording begins with unknown prior state")
            if quality_epoch is not None and epoch != quality_epoch:
                if epoch < quality_epoch:
                    raise ValueError("quality state epoch decreased")
                quality_monitor = None
                quality_gap = False
                quality_input_index = None
                if index != 0 or metadata.get("initial_state") != "epoch_reset":
                    raise ValueError(
                        "quality reset epoch must begin with its first input"
                    )
            if quality_input_index is not None:
                if index <= quality_input_index:
                    raise ValueError("quality epoch input index did not increase")
                if index > quality_input_index + 1:
                    quality_gap = True
                    quality_monitor = None
            quality_input_index = index
            quality_epoch = epoch
            if record["source_sequence_gap"]:
                quality_monitor = None
                quality_gap = True
            effective = metadata["effective_config"]
            if quality_monitor is None:
                quality_monitor = LiveGraspQualityMonitor(effective)
            else:
                quality_monitor.config = effective
            baseline_fields = {item.name for item in fields(BaselineGraspQuality)}
            observation_fields = {item.name for item in fields(LiveGraspObservation)}
            if (
                not isinstance(metadata.get("baseline"), dict)
                or not isinstance(metadata.get("observation"), dict)
                or set(metadata["baseline"]) != baseline_fields
                or set(metadata["observation"]) != observation_fields
            ):
                raise ValueError(
                    "quality replay requires complete recorded dataclass inputs"
                )
            timestamp = metadata["observation"]["timestamp"]
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not np.isfinite(timestamp)
            ):
                raise ValueError(
                    "quality observation timestamp must be explicitly finite"
                )
            if type(metadata["baseline"]["valid"]) is not bool:
                raise ValueError("baseline validity must be a boolean")
            baseline = BaselineGraspQuality(**metadata["baseline"])
            observation = _ablate(
                LiveGraspObservation(**metadata["observation"]), ablation
            )
            result = quality_monitor.update(baseline, observation)
            raw = result.to_dict()
            used = result.davis_motion_used_for_live_quality is True
            freshness = str(observation.davis_event_freshness or "").upper()
            status = (
                "valid"
                if used
                else "stale"
                if freshness in {"STALE", "AGING"}
                else "unavailable"
                if freshness in {"", "UNAVAILABLE"}
                else "excluded"
            )
            reasons = (
                []
                if used
                else [
                    result.davis_motion_live_quality_reason
                    or result.davis_closeup_exclusion_reason
                    or "events_not_used"
                ]
            )
            valid = baseline.valid and not quality_gap
            invalid_reasons = (
                [baseline.failure_reason or "invalid_baseline"]
                if not baseline.valid
                else []
            ) + (["recorded_observation_gap"] if quality_gap else [])
            outputs.append(
                {
                    **common,
                    "prediction": {
                        "label": result.decision if valid else None,
                        "score": result.live_quality_score,
                        "decision": result.decision,
                        "valid": valid,
                        "reasons": invalid_reasons,
                    },
                    "evidence": {
                        "davis_events": {"status": status, "reasons": reasons}
                    },
                    "raw_quality": raw,
                }
            )
        elif kind != "rgbd_trace":
            unsupported.append(
                {
                    "kind": kind,
                    "observation_id": record["observation_id"],
                    "reason": "preserved_raw_input_no_component_replay_adapter",
                }
            )
    return {
        "schema_version": "grasp_component_replay_v1",
        "recording_id": manifest["recording_id"],
        "trial_id": manifest["trial_id"],
        "configuration": {**configuration, "sha256": config_hash},
        "source_revision": manifest["revision"],
        "drop_accounting": manifest["summary"],
        "observations": outputs,
        "unsupported_observations": unsupported,
        "rgbd_trace": replay_rgbd_trace(recording.records),
        "physical_validation": False,
        "ground_truth_generated": False,
        "scope": "canonical_DAVIS_normalization_and_estimator_and_live_quality_core; not_full_RGBD_tracker_dashboard_or_hardware_timing",
    }
