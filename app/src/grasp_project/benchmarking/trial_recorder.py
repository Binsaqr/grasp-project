"""Small file-backed trial recorder for Benchmark Lab.

The recorder stores manual labels separately from predictions and writes only
JSON/CSV/HTML summaries and key snapshots. It never records raw video streams.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


GROUND_TRUTH_SCHEMA_VERSION = "benchmark_ground_truth_v2"
ACCURACY_METRIC_FIELDS = {
    "detection": ("bottle_present", ("bottle_present", "bottle_detected")),
    "pose_classification": ("pose_class", ("pose_class", "object_pose_class")),
    "clearance": (
        "feasible_grasp_region",
        ("feasible_grasp_region", "grasp_region_feasible", "clearance_feasible"),
    ),
    "decision": ("expected_decision", ("decision", "expected_decision")),
}
SPECIALIZED_APPLICABILITY_FIELDS = {
    "pose_angle": "pose_angle_deg",
    "identity": "target_identity",
    "target_persistence": "target_persistent",
    "selected_band": "selected_band",
    "body_fraction": "selected_body_fraction",
    "band_center": "band_center_px",
}
_NON_APPLICABLE_TOKENS = {
    "",
    "N/A",
    "NA",
    "NOT_APPLICABLE",
    "NOT_TESTED",
    "UNAVAILABLE",
    "UNKNOWN",
    "UNKNOWN_ORIENTATION",
    "STALE",
}


def _utc_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{int((time.time() % 1) * 1000):03d}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if token in {"TRUE", "YES", "Y", "1", "PRESENT", "FEASIBLE"}:
        return True
    if token in {"FALSE", "NO", "N", "0", "ABSENT", "INFEASIBLE"}:
        return False
    return None


def _category_or_none(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return None if token in _NON_APPLICABLE_TOKENS else token


def _float_or_none(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _point_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    point = [_float_or_none(item) for item in value]
    return None if any(item is None for item in point) else [float(item) for item in point]


def _first(mapping: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _normalize_prediction_field(raw: Any, normalizer) -> Any:
    """Normalize a prediction while preserving explicit field gates."""
    if not isinstance(raw, dict):
        return normalizer(raw)
    current = None
    applicable = None
    while isinstance(raw, dict):
        if "current" in raw:
            current = (current is not False) and raw.get("current") is True
        if "applicable" in raw:
            applicable = (applicable is not False) and raw.get("applicable") is True
        raw = raw.get("value")
    value = normalizer(raw)
    wrapped = {"value": value}
    if current is not None:
        wrapped["current"] = current
    if applicable is not None:
        wrapped["applicable"] = applicable
    return wrapped


def normalize_ground_truth_payload(payload: dict | None) -> dict:
    """Return the supported manual-label fields without inventing missing truth."""
    payload = dict(payload or {})
    return {
        "bottle_present": _bool_or_none(payload.get("bottle_present")),
        "pose_class": _category_or_none(
            _first(payload, ("pose_class", "object_pose_class"))
        ),
        "clutter_class": _category_or_none(payload.get("clutter_class")),
        "feasible_grasp_region": _bool_or_none(
            _first(
                payload,
                (
                    "feasible_grasp_region",
                    "grasp_region_feasible",
                    "clearance_feasible",
                ),
            )
        ),
        "expected_decision": _category_or_none(payload.get("expected_decision")),
        "motion_label": _category_or_none(
            _first(payload, ("motion_label", "disturbance_motion_label"))
        ),
        "target_identity": _category_or_none(
            _first(
                payload,
                (
                    "target_identity",
                    "target_id",
                    "bottle_target_id",
                    "expected_target_id",
                    "locked_bottle_target_id",
                ),
            )
        ),
        "pose_angle_deg": _float_or_none(
            _first(
                payload,
                (
                    "pose_angle_deg",
                    "object_pose_angle_deg",
                    "tilt_angle_deg",
                    "camera_frame_tilt_deg",
                ),
            )
        ),
        "selected_band": _category_or_none(
            _first(
                payload,
                (
                    "selected_band",
                    "grasp_band",
                    "selected_band_id",
                    "band_id",
                    "selected_candidate_id",
                ),
            )
        ),
        "selected_body_fraction": _float_or_none(
            _first(
                payload,
                (
                    "selected_body_fraction",
                    "selected_grasp_body_fraction",
                    "body_fraction",
                    "locked_body_fraction",
                    "locked_grasp_fraction",
                ),
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        "target_persistent": _bool_or_none(
            _first(
                payload,
                (
                    "target_persistent",
                    "target_persistence",
                    "identity_persistent",
                    "identity_should_persist",
                ),
            )
        ),
        "band_center_px": _point_or_none(
            _first(
                payload,
                (
                    "band_center_px",
                    "grasp_band_center_px",
                    "clamp_center_px",
                    "displayed_band_center_px",
                    "displayed_magenta_center",
                    "display_magenta_center",
                ),
            )
        ),
    }


def normalize_prediction_payload(payload: dict | None) -> dict:
    """Normalize only prediction fields that can be paired with manual truth."""
    payload = dict(payload or {})
    return {
        "bottle_present": _normalize_prediction_field(
            _first(payload, ("bottle_present", "bottle_detected")), _bool_or_none
        ),
        "pose_class": _normalize_prediction_field(
            _first(payload, ("pose_class", "object_pose_class")), _category_or_none
        ),
        "feasible_grasp_region": _normalize_prediction_field(
            _first(
                payload,
                (
                    "feasible_grasp_region",
                    "grasp_region_feasible",
                    "clearance_feasible",
                ),
            ),
            _bool_or_none,
        ),
        "decision": _normalize_prediction_field(
            _first(payload, ("decision", "expected_decision")), _category_or_none
        ),
        "target_identity": _normalize_prediction_field(
            _first(
                payload,
                (
                    "target_identity",
                    "target_id",
                    "bottle_target_id",
                    "selected_bottle_target_id",
                    "locked_bottle_target_id",
                ),
            ),
            _category_or_none,
        ),
        "pose_angle_deg": _normalize_prediction_field(
            _first(
                payload,
                (
                    "pose_angle_deg",
                    "object_pose_angle_deg",
                    "tilt_angle_deg",
                    "camera_frame_tilt_deg",
                ),
            ),
            _float_or_none,
        ),
        "selected_band": _normalize_prediction_field(
            _first(
                payload,
                (
                    "selected_band",
                    "grasp_band",
                    "selected_band_id",
                    "band_id",
                    "selected_candidate_id",
                ),
            ),
            _category_or_none,
        ),
        "selected_body_fraction": _normalize_prediction_field(
            _first(
                payload,
                (
                    "selected_body_fraction",
                    "selected_grasp_body_fraction",
                    "body_fraction",
                    "locked_body_fraction",
                    "locked_grasp_fraction",
                ),
            ),
            lambda value: _float_or_none(value, minimum=0.0, maximum=1.0),
        ),
        "target_persistent": _normalize_prediction_field(
            _first(
                payload,
                (
                    "target_persistent",
                    "target_persistence",
                    "target_identity_preserved",
                    "identity_tracking_ok",
                ),
            ),
            _bool_or_none,
        ),
        "band_center_px": _normalize_prediction_field(
            _first(
                payload,
                (
                    "band_center_px",
                    "grasp_band_center_px",
                    "clamp_center_px",
                    "displayed_band_center_px",
                    "displayed_magenta_center",
                    "display_magenta_center",
                ),
            ),
            _point_or_none,
        ),
        "frame_id": payload.get("frame_id"),
        "configuration_id": payload.get("configuration_id"),
    }


def _truth_field_applicable(metric: str, truth: dict) -> bool:
    field = ACCURACY_METRIC_FIELDS[metric][0]
    value = truth.get(field)
    if value is None:
        return False
    if metric in {"pose_classification", "clearance"}:
        return truth.get("bottle_present") is True
    return True


def _prediction_field_value(metric: str, prediction: dict) -> tuple[Any, bool, bool]:
    aliases = ACCURACY_METRIC_FIELDS[metric][1]
    raw = _first(prediction, aliases)
    field_current = True
    field_applicable = True
    if isinstance(raw, dict):
        field_current = raw.get("current") is not False
        field_applicable = raw.get("applicable") is not False
        raw = raw.get("value")
    if metric in {"detection", "clearance"}:
        value = _bool_or_none(raw)
    else:
        value = _category_or_none(raw)
    return value, field_current, field_applicable


def _prediction_value(prediction: dict, field: str, normalizer) -> tuple[Any, bool, bool]:
    raw = prediction.get(field)
    field_current = True
    field_applicable = True
    if isinstance(raw, dict):
        field_current = raw.get("current") is not False
        field_applicable = raw.get("applicable") is not False
        raw = raw.get("value")
    return normalizer(raw), field_current, field_applicable


def _mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def _median(values: list[float]) -> float | None:
    return None if not values else statistics.median(values)


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _angle_error_deg(predicted: float, expected: float) -> float:
    return abs((float(predicted) - float(expected) + 180.0) % 360.0 - 180.0)


def _measurement_status(
    pair_count: int,
    counts: dict,
    *,
    labels: list,
    structured_count: int,
) -> tuple[str, str]:
    if pair_count:
        return "VALID", f"{pair_count} current applicable truth/prediction pairs."
    if not labels:
        return "NOT_TESTED", "No ground-truth labels were recorded."
    if structured_count == 0:
        return "NOT_TESTED", "Only freeform labels were recorded; no structured truth was available."
    if counts.get("truth_label_count", 0) == 0:
        return "NOT_APPLICABLE", "No applicable structured truth was recorded for this metric."
    if counts.get("applicable_truth_label_count", 0) == 0:
        return "NOT_APPLICABLE", "Structured truth explicitly marked this metric not applicable."
    if counts.get("current_truth_label_count", 0) == 0:
        return "NON_CURRENT", "Relevant truth exists, but no evidence snapshot was explicitly current."
    if counts.get("non_current_label_count", 0) > 0:
        return "NON_CURRENT", "Relevant prediction evidence was explicitly non-current."
    return "NOT_TESTED", "No current applicable prediction was paired with the structured truth."


def evaluate_ground_truth_labels(
    ground_truth: dict | list | None,
    *,
    declared: bool = False,
) -> dict:
    """Evaluate label availability and paired accuracy without fail-open inference.

    A field contributes only when its manual truth is applicable, its paired
    prediction is present/applicable, and the prediction snapshot is explicitly
    current. Freeform labels remain useful trial notes but never become accuracy.
    """
    if isinstance(ground_truth, dict):
        labels = list(ground_truth.get("labels") or [])
    elif isinstance(ground_truth, list):
        labels = list(ground_truth)
    else:
        labels = []

    metric_counts = {
        metric: {
            "correct_count": 0,
            "applicable_label_count": 0,
            "current_truth_label_count": 0,
            "non_current_label_count": 0,
            "missing_or_nonapplicable_prediction_count": 0,
        }
        for metric in ACCURACY_METRIC_FIELDS
    }
    specialized_counts = {
        name: {
            "truth_label_count": 0,
            "applicable_truth_label_count": 0,
            "current_truth_label_count": 0,
            "non_current_label_count": 0,
            "missing_or_nonapplicable_prediction_count": 0,
            "applicable_pair_count": 0,
        }
        for name in SPECIALIZED_APPLICABILITY_FIELDS
    }
    detection_confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    pose_errors: list[float] = []
    identity_matches: list[bool] = []
    persistence_matches: list[bool] = []
    selected_band_matches: list[bool] = []
    body_fraction_errors: list[float] = []
    band_center_errors: list[float] = []
    identity_records: list[dict] = []
    band_records: list[dict] = []
    structured_count = 0
    current_structured_count = 0
    used_label_indices: set[int] = set()

    for index, entry_value in enumerate(labels):
        entry = dict(entry_value or {}) if isinstance(entry_value, dict) else {}
        truth_raw = entry.get("ground_truth")
        if not isinstance(truth_raw, dict):
            continue
        truth = normalize_ground_truth_payload(truth_raw)
        if not any(value is not None for value in truth.values()):
            continue
        structured_count += 1
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        evidence_current = evidence.get("current") is True
        if evidence_current:
            current_structured_count += 1
        applicability = (
            evidence.get("applicability")
            if isinstance(evidence.get("applicability"), dict)
            else {}
        )
        prediction = normalize_prediction_payload(
            entry.get("prediction")
            if isinstance(entry.get("prediction"), dict)
            else None
        )

        for metric, counts in metric_counts.items():
            if not _truth_field_applicable(metric, truth):
                continue
            if applicability.get(metric) is False:
                continue
            counts["current_truth_label_count"] += int(evidence_current)
            if not evidence_current:
                counts["non_current_label_count"] += 1
                continue
            predicted, field_current, field_applicable = _prediction_field_value(
                metric,
                prediction,
            )
            if not field_current:
                counts["non_current_label_count"] += 1
                continue
            if not field_applicable or predicted is None:
                counts["missing_or_nonapplicable_prediction_count"] += 1
                continue
            truth_value = truth[ACCURACY_METRIC_FIELDS[metric][0]]
            counts["applicable_label_count"] += 1
            counts["correct_count"] += int(predicted == truth_value)
            if metric == "detection":
                if truth_value is True and predicted is True:
                    detection_confusion["true_positive"] += 1
                elif truth_value is False and predicted is False:
                    detection_confusion["true_negative"] += 1
                elif truth_value is False and predicted is True:
                    detection_confusion["false_positive"] += 1
                elif truth_value is True and predicted is False:
                    detection_confusion["false_negative"] += 1
            used_label_indices.add(index)

        def paired_specialized(
            name: str,
            truth_field: str,
            prediction_field: str,
            normalizer,
            *,
            require_bottle: bool = True,
            group: str | None = None,
        ) -> tuple[Any, Any] | None:
            truth_value = truth.get(truth_field)
            if truth_value is None or (require_bottle and truth.get("bottle_present") is not True):
                return None
            counts = specialized_counts[name]
            counts["truth_label_count"] += 1
            if applicability.get(name) is False or (group and applicability.get(group) is False):
                return None
            counts["applicable_truth_label_count"] += 1
            if not evidence_current:
                counts["non_current_label_count"] += 1
                return None
            counts["current_truth_label_count"] += 1
            predicted_value, field_current, field_applicable = _prediction_value(
                prediction,
                prediction_field,
                normalizer,
            )
            if not field_current:
                counts["non_current_label_count"] += 1
                return None
            if not field_applicable or predicted_value is None:
                counts["missing_or_nonapplicable_prediction_count"] += 1
                return None
            counts["applicable_pair_count"] += 1
            used_label_indices.add(index)
            return truth_value, predicted_value

        pose_pair = paired_specialized(
            "pose_angle",
            "pose_angle_deg",
            "pose_angle_deg",
            _float_or_none,
        )
        if pose_pair:
            pose_errors.append(_angle_error_deg(pose_pair[1], pose_pair[0]))

        identity_pair = paired_specialized(
            "identity",
            "target_identity",
            "target_identity",
            _category_or_none,
        )
        if identity_pair:
            identity_matches.append(identity_pair[0] == identity_pair[1])
            identity_records.append(
                {
                    "truth": identity_pair[0],
                    "prediction": identity_pair[1],
                    "persistent": truth.get("target_persistent"),
                }
            )

        persistence_pair = paired_specialized(
            "target_persistence",
            "target_persistent",
            "target_persistent",
            _bool_or_none,
        )
        if persistence_pair:
            persistence_matches.append(persistence_pair[0] == persistence_pair[1])

        selected_band_pair = paired_specialized(
            "selected_band",
            "selected_band",
            "selected_band",
            _category_or_none,
            group="band",
        )
        if selected_band_pair:
            selected_band_matches.append(selected_band_pair[0] == selected_band_pair[1])

        body_fraction_pair = paired_specialized(
            "body_fraction",
            "selected_body_fraction",
            "selected_body_fraction",
            lambda value: _float_or_none(value, minimum=0.0, maximum=1.0),
            group="band",
        )
        if body_fraction_pair:
            body_fraction_errors.append(abs(body_fraction_pair[1] - body_fraction_pair[0]))

        band_center_pair = paired_specialized(
            "band_center",
            "band_center_px",
            "band_center_px",
            _point_or_none,
            group="band",
        )
        if band_center_pair:
            band_center_errors.append(
                math.hypot(
                    band_center_pair[1][0] - band_center_pair[0][0],
                    band_center_pair[1][1] - band_center_pair[0][1],
                )
            )

        if body_fraction_pair or band_center_pair:
            band_records.append(
                {
                    "target_identity": truth.get("target_identity"),
                    "persistent": truth.get("target_persistent"),
                    "truth_fraction": None if not body_fraction_pair else body_fraction_pair[0],
                    "predicted_fraction": None if not body_fraction_pair else body_fraction_pair[1],
                    "truth_center": None if not band_center_pair else band_center_pair[0],
                    "predicted_center": None if not band_center_pair else band_center_pair[1],
                }
            )

    accuracy_metrics = {}
    for metric, counts in metric_counts.items():
        applicable_count = counts["applicable_label_count"]
        valid = applicable_count > 0
        if valid:
            status = "VALID"
            reason = (
                f"{applicable_count} current applicable label/prediction pairs."
            )
        elif not labels:
            status = "NOT_TESTED"
            reason = "No ground-truth labels were recorded."
        elif structured_count == 0:
            status = "NOT_TESTED"
            reason = "Only freeform labels were recorded; no structured truth was available."
        elif current_structured_count == 0:
            status = "NON_CURRENT"
            reason = "Structured labels exist, but no prediction snapshot was explicitly current."
        elif counts["non_current_label_count"] > 0:
            status = "NON_CURRENT"
            reason = f"Relevant {metric.replace('_', ' ')} prediction evidence was explicitly non-current."
        elif counts["current_truth_label_count"] == 0:
            status = "NOT_APPLICABLE"
            reason = f"No current applicable {metric.replace('_', ' ')} truth labels were recorded."
        else:
            status = "NOT_TESTED"
            reason = f"No current applicable {metric.replace('_', ' ')} prediction evidence was paired with the labels."
        accuracy_metrics[metric] = {
            **counts,
            "valid": valid,
            "status": status,
            "accuracy_percent": (
                round(100.0 * counts["correct_count"] / applicable_count, 2)
                if applicable_count
                else None
            ),
            "reason": reason,
        }

    detection_pairs = metric_counts["detection"]["applicable_label_count"]
    true_positive = detection_confusion["true_positive"]
    false_positive = detection_confusion["false_positive"]
    false_negative = detection_confusion["false_negative"]
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    detection_status = accuracy_metrics["detection"]["status"]
    detection_summary = {
        "valid": detection_pairs > 0,
        "status": detection_status,
        "pair_count": detection_pairs,
        "confusion": detection_confusion,
        "precision_valid": precision_denominator > 0,
        "precision_percent": (
            round(100.0 * true_positive / precision_denominator, 2)
            if precision_denominator
            else None
        ),
        "recall_valid": recall_denominator > 0,
        "recall_percent": (
            round(100.0 * true_positive / recall_denominator, 2)
            if recall_denominator
            else None
        ),
        "accuracy_percent": accuracy_metrics["detection"]["accuracy_percent"],
        "reason": accuracy_metrics["detection"]["reason"],
    }

    pose_counts = specialized_counts["pose_angle"]
    pose_status, pose_reason = _measurement_status(
        len(pose_errors), pose_counts, labels=labels, structured_count=structured_count
    )
    pose_summary = {
        **pose_counts,
        "valid": bool(pose_errors),
        "status": pose_status,
        "mean_absolute_error_deg": _rounded(_mean(pose_errors)),
        "median_absolute_error_deg": _rounded(_median(pose_errors)),
        "max_absolute_error_deg": _rounded(max(pose_errors) if pose_errors else None),
        "reason": pose_reason,
    }

    identity_switch_opportunities = 0
    identity_switches = 0
    for previous, current in zip(identity_records, identity_records[1:]):
        explicitly_same_target = bool(
            previous["truth"] == current["truth"]
            and previous["persistent"] is True
            and current["persistent"] is True
        )
        if not explicitly_same_target:
            continue
        identity_switch_opportunities += 1
        identity_switches += int(previous["prediction"] != current["prediction"])

    identity_counts = specialized_counts["identity"]
    persistence_counts = specialized_counts["target_persistence"]
    identity_pair_count = len(identity_matches)
    persistence_pair_count = len(persistence_matches)
    identity_status, identity_reason = _measurement_status(
        identity_pair_count + persistence_pair_count,
        {
            "truth_label_count": identity_counts["truth_label_count"]
            + persistence_counts["truth_label_count"],
            "applicable_truth_label_count": identity_counts[
                "applicable_truth_label_count"
            ]
            + persistence_counts["applicable_truth_label_count"],
            "current_truth_label_count": identity_counts["current_truth_label_count"]
            + persistence_counts["current_truth_label_count"],
            "non_current_label_count": identity_counts["non_current_label_count"]
            + persistence_counts["non_current_label_count"],
        },
        labels=labels,
        structured_count=structured_count,
    )
    identity_summary = {
        "valid": bool(identity_pair_count or persistence_pair_count),
        "status": identity_status,
        "identity_pair_count": identity_pair_count,
        "identity_correct_count": sum(identity_matches),
        "identity_accuracy_valid": identity_pair_count > 0,
        "identity_accuracy_percent": (
            round(100.0 * sum(identity_matches) / identity_pair_count, 2)
            if identity_pair_count
            else None
        ),
        "persistence_pair_count": persistence_pair_count,
        "persistence_correct_count": sum(persistence_matches),
        "persistence_accuracy_valid": persistence_pair_count > 0,
        "persistence_accuracy_percent": (
            round(100.0 * sum(persistence_matches) / persistence_pair_count, 2)
            if persistence_pair_count
            else None
        ),
        "identity_switch_proxy_valid": identity_switch_opportunities > 0,
        "identity_switch_proxy_opportunities": identity_switch_opportunities,
        "identity_switch_proxy_count": identity_switches,
        "identity_switch_proxy_rate_percent": (
            round(100.0 * identity_switches / identity_switch_opportunities, 2)
            if identity_switch_opportunities
            else None
        ),
        "identity_switch_proxy_definition": (
            "prediction identity changed between consecutive valid identity-paired "
            "labels whose truth identity was unchanged and target_persistent was true"
        ),
        "reason": identity_reason,
    }

    body_fraction_drift_errors: list[float] = []
    band_center_drift_errors: list[float] = []
    for previous, current in zip(band_records, band_records[1:]):
        explicitly_same_target = bool(
            previous["target_identity"] is not None
            and previous["target_identity"] == current["target_identity"]
            and previous["persistent"] is True
            and current["persistent"] is True
        )
        if not explicitly_same_target:
            continue
        if all(
            value is not None
            for value in (
                previous["truth_fraction"],
                previous["predicted_fraction"],
                current["truth_fraction"],
                current["predicted_fraction"],
            )
        ):
            truth_delta = current["truth_fraction"] - previous["truth_fraction"]
            predicted_delta = current["predicted_fraction"] - previous["predicted_fraction"]
            body_fraction_drift_errors.append(abs(predicted_delta - truth_delta))
        if all(
            value is not None
            for value in (
                previous["truth_center"],
                previous["predicted_center"],
                current["truth_center"],
                current["predicted_center"],
            )
        ):
            truth_dx = current["truth_center"][0] - previous["truth_center"][0]
            truth_dy = current["truth_center"][1] - previous["truth_center"][1]
            predicted_dx = current["predicted_center"][0] - previous["predicted_center"][0]
            predicted_dy = current["predicted_center"][1] - previous["predicted_center"][1]
            band_center_drift_errors.append(
                math.hypot(predicted_dx - truth_dx, predicted_dy - truth_dy)
            )

    band_pair_count = (
        len(selected_band_matches) + len(body_fraction_errors) + len(band_center_errors)
    )
    band_combined_counts = {
        "truth_label_count": sum(
            specialized_counts[name]["truth_label_count"]
            for name in ("selected_band", "body_fraction", "band_center")
        ),
        "current_truth_label_count": sum(
            specialized_counts[name]["current_truth_label_count"]
            for name in ("selected_band", "body_fraction", "band_center")
        ),
        "applicable_truth_label_count": sum(
            specialized_counts[name]["applicable_truth_label_count"]
            for name in ("selected_band", "body_fraction", "band_center")
        ),
        "non_current_label_count": sum(
            specialized_counts[name]["non_current_label_count"]
            for name in ("selected_band", "body_fraction", "band_center")
        ),
    }
    band_status, band_reason = _measurement_status(
        band_pair_count,
        band_combined_counts,
        labels=labels,
        structured_count=structured_count,
    )
    band_summary = {
        "valid": band_pair_count > 0,
        "status": band_status,
        "selected_band_pair_count": len(selected_band_matches),
        "selected_band_accuracy_valid": bool(selected_band_matches),
        "selected_band_accuracy_percent": (
            round(100.0 * sum(selected_band_matches) / len(selected_band_matches), 2)
            if selected_band_matches
            else None
        ),
        "body_fraction_error_valid": bool(body_fraction_errors),
        "body_fraction_pair_count": len(body_fraction_errors),
        "mean_absolute_body_fraction_error": _rounded(_mean(body_fraction_errors), 6),
        "median_absolute_body_fraction_error": _rounded(_median(body_fraction_errors), 6),
        "max_absolute_body_fraction_error": _rounded(
            max(body_fraction_errors) if body_fraction_errors else None, 6
        ),
        "band_center_error_valid": bool(band_center_errors),
        "band_center_pair_count": len(band_center_errors),
        "mean_band_center_error_px": _rounded(_mean(band_center_errors)),
        "median_band_center_error_px": _rounded(_median(band_center_errors)),
        "max_band_center_error_px": _rounded(
            max(band_center_errors) if band_center_errors else None
        ),
        "body_fraction_drift_valid": bool(body_fraction_drift_errors),
        "body_fraction_drift_pair_count": len(body_fraction_drift_errors),
        "mean_body_fraction_drift_error": _rounded(
            _mean(body_fraction_drift_errors), 6
        ),
        "max_body_fraction_drift_error": _rounded(
            max(body_fraction_drift_errors) if body_fraction_drift_errors else None,
            6,
        ),
        "band_center_drift_valid": bool(band_center_drift_errors),
        "band_center_drift_pair_count": len(band_center_drift_errors),
        "mean_band_center_drift_error_px": _rounded(
            _mean(band_center_drift_errors)
        ),
        "max_band_center_drift_error_px": _rounded(
            max(band_center_drift_errors) if band_center_drift_errors else None
        ),
        "drift_definition": (
            "difference between predicted and truth displacement across adjacent "
            "valid band-paired labels for the same explicitly persistent truth identity"
        ),
        "reason": band_reason,
    }

    structured_metrics = {
        "detection": detection_summary,
        "pose_angle_error": pose_summary,
        "identity": identity_summary,
        "band": band_summary,
    }
    structured_metrics_valid = any(
        metric.get("valid") is True for metric in structured_metrics.values()
    )

    valid_metric_count = sum(
        metric["valid"] for metric in accuracy_metrics.values()
    )
    if not labels:
        state = "DECLARED_WITHOUT_LABELS" if declared else "NOT_RECORDED"
    elif structured_count == 0:
        state = "UNSTRUCTURED_LABELS_ONLY"
    elif current_structured_count == 0:
        state = "NON_CURRENT"
    elif valid_metric_count == 0 and not structured_metrics_valid:
        state = "NO_APPLICABLE_CURRENT_PAIRS"
    elif valid_metric_count == len(accuracy_metrics):
        state = "AVAILABLE"
    else:
        state = "AVAILABLE_WITH_LIMITATIONS"

    labels_used = len(used_label_indices)
    accuracy_valid = valid_metric_count > 0
    if accuracy_valid:
        accuracy_reason = (
            f"{labels_used} current structured labels contributed "
            f"{sum(item['applicable_label_count'] for item in accuracy_metrics.values())} "
            "applicable field comparisons."
        )
    elif state == "DECLARED_WITHOUT_LABELS":
        accuracy_reason = "Ground truth was declared, but zero label records were supplied."
    elif state == "UNSTRUCTURED_LABELS_ONLY":
        accuracy_reason = "Recorded labels were freeform notes and are not accuracy ground truth."
    elif state == "NON_CURRENT":
        accuracy_reason = "Structured labels exist, but their prediction evidence is not explicitly current."
    elif structured_metrics_valid:
        accuracy_reason = (
            "Current structured measurement evidence is available, but no legacy "
            "categorical accuracy field was paired."
        )
    elif state == "NO_APPLICABLE_CURRENT_PAIRS":
        accuracy_reason = "No current applicable ground-truth/prediction field pairs were available."
    else:
        accuracy_reason = "No ground-truth labels were attached to this benchmark run."

    return {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "ground_truth_declared": bool(declared or labels),
        "ground_truth_recorded": bool(labels),
        "ground_truth_state": state,
        "ground_truth_label_count": len(labels),
        "structured_ground_truth_label_count": structured_count,
        "current_ground_truth_label_count": current_structured_count,
        "ground_truth_labels_used": labels_used,
        "ground_truth_applicable_comparison_count": sum(
            item["applicable_label_count"] for item in accuracy_metrics.values()
        ),
        "ground_truth_available": bool(accuracy_valid or structured_metrics_valid),
        "has_ground_truth": bool(accuracy_valid or structured_metrics_valid),
        "accuracy_metrics_valid": accuracy_valid,
        "all_accuracy_metrics_valid": valid_metric_count == len(accuracy_metrics),
        "valid_accuracy_metric_count": valid_metric_count,
        "accuracy_metrics_reason": accuracy_reason,
        "accuracy_metrics": accuracy_metrics,
        "structured_metrics_valid": structured_metrics_valid,
        "structured_measurement_pair_count": (
            detection_pairs
            + len(pose_errors)
            + identity_pair_count
            + persistence_pair_count
            + band_pair_count
        ),
        "structured_metrics": structured_metrics,
    }


def _ground_truth_document(labels: list[dict], source: str = "manual_operator_labels") -> dict:
    evaluation = evaluate_ground_truth_labels(labels, declared=bool(labels))
    return {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "source": source,
        "labels": labels,
        **evaluation,
    }


class BenchmarkTrialRecorder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.experiments_dir = self.root / "experiments"
        self.current_path = self.root / "current_trial.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def current_state(self) -> dict:
        state = _read_json(self.current_path, default={}) or {}
        if not state:
            return {
                "active": False,
                "experiment_id": None,
                "trial_id": None,
                "trial_dir": None,
                "label_count": 0,
                "last_label": None,
                "ground_truth_state": "NOT_RECORDED",
                "structured_ground_truth_label_count": 0,
                "current_ground_truth_label_count": 0,
                "ground_truth_labels_used": 0,
                "accuracy_metrics_valid": False,
                "structured_metrics_valid": False,
            }
        trial_dir = Path(state.get("trial_dir") or "")
        labels_path = trial_dir / "labels.jsonl"
        label_count = 0
        last_label = None
        if labels_path.exists():
            lines = [line for line in labels_path.read_text().splitlines() if line.strip()]
            label_count = len(lines)
            if lines:
                try:
                    last_label = json.loads(lines[-1]).get("label")
                except Exception:
                    last_label = None
        ground_truth = _read_json(trial_dir / "ground_truth.json", default={}) or {}
        evaluation = evaluate_ground_truth_labels(
            ground_truth,
            declared=bool(ground_truth.get("ground_truth_declared") or label_count),
        )
        return {
            **state,
            "label_count": label_count,
            "last_label": last_label,
            "ground_truth_state": evaluation["ground_truth_state"],
            "structured_ground_truth_label_count": evaluation[
                "structured_ground_truth_label_count"
            ],
            "current_ground_truth_label_count": evaluation[
                "current_ground_truth_label_count"
            ],
            "ground_truth_labels_used": evaluation["ground_truth_labels_used"],
            "accuracy_metrics_valid": evaluation["accuracy_metrics_valid"],
            "structured_metrics_valid": evaluation["structured_metrics_valid"],
        }

    def start_trial(self, configurations: list[dict] | None = None, metadata: dict | None = None) -> dict:
        experiment_id = _utc_id("experiment")
        trial_id = _utc_id("trial")
        experiment_dir = self.experiments_dir / experiment_id
        trial_dir = experiment_dir / "trials" / trial_id
        snapshot_dir = trial_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        experiment = {
            "experiment_id": experiment_id,
            "created_at": now,
            "purpose": "Benchmark Lab display/report-only comparison",
            "raw_video_recording": False,
        }
        manifest = {
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "started_at": now,
            "active": True,
            "metadata": metadata or {},
            "benchmark_only": True,
            "production_control": False,
        }
        _write_json(experiment_dir / "experiment.json", experiment)
        _write_json(experiment_dir / "configurations.json", {"configurations": configurations or []})
        _write_json(trial_dir / "manifest.json", manifest)
        _write_json(trial_dir / "ground_truth.json", _ground_truth_document([]))
        state = {
            "active": True,
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "experiment_dir": str(experiment_dir),
            "trial_dir": str(trial_dir),
            "started_at": now,
        }
        _write_json(self.current_path, state)
        return self.current_state()

    def stop_trial(self, summary: dict | None = None) -> dict:
        state = self.current_state()
        if not state.get("trial_dir"):
            return state
        state["active"] = False
        state["stopped_at"] = time.time()
        _write_json(self.current_path, state)
        manifest_path = Path(state["trial_dir"]) / "manifest.json"
        manifest = _read_json(manifest_path, default={}) or {}
        manifest.update({"active": False, "stopped_at": state["stopped_at"], "summary": summary or {}})
        _write_json(manifest_path, manifest)
        return self.current_state()

    def ensure_trial(self, configurations: list[dict] | None = None) -> dict:
        state = self.current_state()
        if state.get("trial_dir"):
            return state
        return self.start_trial(configurations=configurations)

    def record_label(
        self,
        label: str | dict,
        benchmark_status: dict | None = None,
        note: str | None = None,
        *,
        ground_truth: dict | None = None,
        prediction: dict | None = None,
        evidence_current: bool | None = None,
    ) -> dict:
        state = self.ensure_trial((benchmark_status or {}).get("configurations") or [])
        trial_dir = Path(state["trial_dir"])
        label_payload = dict(label) if isinstance(label, dict) else {}
        label_name = str(
            label_payload.get("label")
            or (label if isinstance(label, str) else "structured_scene_label")
        ).strip()
        truth_input = ground_truth
        if truth_input is None and label_payload:
            truth_input = label_payload.get("ground_truth")
            if truth_input is None:
                truth_input = label_payload
        normalized_truth = normalize_ground_truth_payload(truth_input)
        structured = any(value is not None for value in normalized_truth.values())

        production = (benchmark_status or {}).get("production_context") or {}
        cards = (benchmark_status or {}).get("comparison_cards") or []
        card = cards[0] if cards else {}
        source_status = (benchmark_status or {}).get("source_status") or {}
        realsense = source_status.get("realsense") or {}
        object_state = realsense.get("object_state") or {}
        target_lock = realsense.get("target_lock") or {}
        target_clamp = (
            realsense.get("target_clamp")
            or target_lock.get("target_clamp")
            or {}
        )
        task_plan = realsense.get("selected_target_task_plan") or {}
        grasp_memory = realsense.get("grasp_memory") or target_lock.get("grasp_memory") or {}
        identity_current = realsense.get("identity_measurement_current") is True
        target_identity = _first(
            realsense,
            ("locked_bottle_target_id", "selected_bottle_target_id"),
        ) or _first(target_lock, ("bottle_target_id", "target_id"))
        selected_body_fraction = _first(
            realsense,
            ("locked_body_fraction", "selected_grasp_body_fraction"),
        )
        if selected_body_fraction is None:
            selected_body_fraction = _first(
                target_lock,
                ("locked_body_fraction", "locked_grasp_fraction"),
            )
        if selected_body_fraction is None:
            selected_body_fraction = _first(
                task_plan,
                ("selected_grasp_body_fraction", "locked_body_fraction"),
            )
        derived_prediction = {
            "bottle_present": card.get("bottle_detected"),
            "pose_class": card.get("pose_class") or production.get("pose_class"),
            "feasible_grasp_region": card.get("feasible_grasp_region")
            if card.get("feasible_grasp_region") is not None
            else production.get("feasible_grasp_region"),
            "decision": production.get("decision") or card.get("decision"),
            "target_identity": {
                "value": target_identity,
                "current": identity_current,
            },
            "pose_angle_deg": _first(
                object_state,
                ("tilt_angle_deg", "camera_frame_tilt_deg"),
            ),
            "selected_band": _first(
                realsense,
                (
                    "selected_band",
                    "selected_band_id",
                    "selected_candidate_id",
                    "selected_grasp_preference_tier",
                ),
            )
            or _first(
                grasp_memory,
                ("selected_candidate_id", "selected_band_id", "selected_band"),
            )
            or _first(
                task_plan,
                (
                    "selected_candidate_id",
                    "selected_band_id",
                    "selected_band",
                    "selected_grasp_preference_tier",
                ),
            ),
            "selected_body_fraction": selected_body_fraction,
            "target_persistent": {
                "value": _first(
                    realsense,
                    ("target_identity_preserved", "identity_tracking_ok"),
                ),
                "current": identity_current,
            },
            "band_center_px": _first(
                realsense,
                (
                    "displayed_magenta_center",
                    "display_magenta_center",
                    "band_center_px",
                ),
            )
            or _first(target_clamp, ("clamp_center_px", "band_center_px")),
            "frame_id": production.get("frame_id"),
            "configuration_id": card.get("configuration_id"),
        }
        explicit_prediction = prediction or label_payload.get("prediction") or {}
        derived_prediction.update(
            {
                key: value
                for key, value in dict(explicit_prediction).items()
                if value is not None
            }
        )
        normalized_prediction = normalize_prediction_payload(derived_prediction)

        evidence_payload = (
            label_payload.get("evidence")
            if isinstance(label_payload.get("evidence"), dict)
            else {}
        )
        if evidence_current is None:
            evidence_current = evidence_payload.get("current")
        derived_applicability = {
            metric: _truth_field_applicable(metric, normalized_truth)
            for metric in ACCURACY_METRIC_FIELDS
        }
        derived_applicability.update(
            {
                name: bool(
                    normalized_truth.get(field) is not None
                    and normalized_truth.get("bottle_present") is True
                )
                for name, field in SPECIALIZED_APPLICABILITY_FIELDS.items()
            }
        )
        derived_applicability["band"] = any(
            derived_applicability[name]
            for name in ("selected_band", "body_fraction", "band_center")
        )
        explicit_applicability = (
            evidence_payload.get("applicability")
            if isinstance(evidence_payload.get("applicability"), dict)
            else {}
        )
        applicability = {
            metric: bool(
                applicable and explicit_applicability.get(metric) is not False
            )
            for metric, applicable in derived_applicability.items()
        }
        if not structured:
            label_state = "UNSTRUCTURED_NOT_APPLICABLE_FOR_ACCURACY"
        elif evidence_current is not True:
            label_state = "NON_CURRENT_EVIDENCE"
        elif not any(applicability.values()):
            label_state = "NOT_APPLICABLE_FOR_ACCURACY"
        else:
            label_state = "CURRENT_STRUCTURED_LABEL"
        entry = {
            "timestamp": time.time(),
            "label": label_name,
            "note": note or "",
            "source": "manual_operator",
            "prediction_snapshot_id": (benchmark_status or {}).get("production_context", {}).get("frame_id"),
            "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
            "label_state": label_state,
            "ground_truth": normalized_truth if structured else None,
            "prediction": normalized_prediction if structured else None,
            "evidence": {
                "current": evidence_current is True,
                "current_reason": (
                    "explicit_current_snapshot"
                    if evidence_current is True
                    else "prediction_snapshot_not_explicitly_current"
                ),
                "applicability": applicability,
            },
        }
        _append_jsonl(trial_dir / "labels.jsonl", entry)
        ground_truth = _read_json(trial_dir / "ground_truth.json", default={"labels": []}) or {"labels": []}
        labels = list(ground_truth.get("labels") or [])
        labels.append(entry)
        _write_json(trial_dir / "ground_truth.json", _ground_truth_document(labels))
        if benchmark_status:
            _append_jsonl(trial_dir / "benchmark_results.jsonl", {
                "timestamp": time.time(),
                "comparison_cards": benchmark_status.get("comparison_cards") or [],
                "production_context": benchmark_status.get("production_context") or {},
            })
            _append_jsonl(trial_dir / "decisions.jsonl", {
                "timestamp": time.time(),
                "production_decision": (benchmark_status.get("production_context") or {}).get("decision"),
                "benchmark_decisions": [
                    {"configuration": card.get("display_name"), "decision": card.get("decision")}
                    for card in benchmark_status.get("comparison_cards") or []
                ],
            })
            _append_jsonl(trial_dir / "synchronized_features.jsonl", {
                "timestamp": time.time(),
                "bundle_id": (benchmark_status.get("comparison_cards") or [{}])[0].get("same_bundle_id"),
                "features": [card.get("enabled_features") for card in benchmark_status.get("comparison_cards") or []],
            })
        return self.current_state()

    def undo_last_label(self) -> dict:
        state = self.current_state()
        if not state.get("trial_dir"):
            return state
        trial_dir = Path(state["trial_dir"])
        labels_path = trial_dir / "labels.jsonl"
        lines = []
        if labels_path.exists():
            lines = [line for line in labels_path.read_text().splitlines() if line.strip()]
        if lines:
            labels_path.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
        ground_truth_path = trial_dir / "ground_truth.json"
        ground_truth = _read_json(ground_truth_path, default={"labels": []}) or {"labels": []}
        labels = list(ground_truth.get("labels") or [])
        if labels:
            _write_json(
                ground_truth_path,
                _ground_truth_document(
                    labels[:-1],
                    source=str(ground_truth.get("source") or "manual_operator_labels"),
                ),
            )
        return self.current_state()

    def export_report(self, benchmark_status: dict | None = None) -> dict:
        state = self.ensure_trial((benchmark_status or {}).get("configurations") or [])
        trial_dir = Path(state["trial_dir"])
        ground_truth = _read_json(trial_dir / "ground_truth.json", default={}) or {}
        ground_truth_evaluation = evaluate_ground_truth_labels(
            ground_truth,
            declared=bool(ground_truth.get("ground_truth_declared")),
        )
        report = {
            "experiment_id": state.get("experiment_id"),
            "trial_id": state.get("trial_id"),
            "generated_at": time.time(),
            "benchmark_only": True,
            "production_control": False,
            "benchmark_status": benchmark_status or {},
            "ground_truth": ground_truth,
            "ground_truth_summary": ground_truth_evaluation,
            "accuracy_metrics_valid": ground_truth_evaluation[
                "accuracy_metrics_valid"
            ],
            "structured_metrics_valid": ground_truth_evaluation[
                "structured_metrics_valid"
            ],
            "accuracy_metrics_reason": ground_truth_evaluation[
                "accuracy_metrics_reason"
            ],
        }
        _write_json(trial_dir / "report.json", report)
        _write_json(trial_dir / "benchmark_summary.json", report)

        rows = (benchmark_status or {}).get("metrics_rows") or []
        config_csv = trial_dir / "configuration_results.csv"
        with config_csv.open("w", newline="") as handle:
            fieldnames = sorted({key for row in rows for key in row.keys()} or {"configuration"})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        modality_csv = trial_dir / "modality_comparison.csv"
        with modality_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["configuration", "enabled_feature_count", "decision", "confidence"])
            for card in (benchmark_status or {}).get("comparison_cards") or []:
                writer.writerow([card.get("display_name"), len(card.get("enabled_features") or []), card.get("decision"), card.get("confidence")])

        contrib_csv = trial_dir / "feature_contributions.csv"
        with contrib_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["configuration", "feature", "effect", "message"])
            for card in (benchmark_status or {}).get("comparison_cards") or []:
                for contribution in card.get("contributions") or []:
                    writer.writerow([card.get("display_name"), contribution.get("feature"), contribution.get("effect"), contribution.get("message")])

        timeline_png = trial_dir / "quality_timeline.png"
        if not timeline_png.exists():
            timeline_png.write_bytes(b"")

        html_path = trial_dir / "benchmark_report.html"
        cards = (benchmark_status or {}).get("comparison_cards") or []
        body = "\n".join(
            f"<tr><td>{html.escape(str(card.get('display_name')))}</td><td>{html.escape(str(card.get('decision')))}</td><td>{html.escape(str(card.get('confidence')))}</td><td>{html.escape(str(card.get('main_limitation')))}</td></tr>"
            for card in cards
        )
        html_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>Benchmark Report</title></head>"
            "<body><h1>Benchmark Lab Report</h1><p>Display/report-only. Production controls unchanged.</p>"
            "<table border='1'><thead><tr><th>Configuration</th><th>Decision</th><th>Confidence</th><th>Limitation</th></tr></thead>"
            f"<tbody>{body}</tbody></table></body></html>"
        )
        return {
            **self.current_state(),
            "report_json": str(trial_dir / "benchmark_summary.json"),
            "configuration_results_csv": str(config_csv),
            "modality_comparison_csv": str(modality_csv),
            "feature_contributions_csv": str(contrib_csv),
            "quality_timeline_png": str(timeline_png),
            "benchmark_report_html": str(html_path),
        }
