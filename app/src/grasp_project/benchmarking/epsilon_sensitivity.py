"""Offline geometric epsilon controls; never infer physical grasp performance."""

from __future__ import annotations

import copy
import math
import statistics

from grasp_project.realsense.epsilon_wrench import compute_epsilon_quality

SCHEMA = "grasp_epsilon_sensitivity/v1"


def _finite(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _vector(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_finite(item) for item in value)
    )


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sample_errors(sample, kind):
    if not isinstance(sample, dict):
        return ["sample_must_be_an_object"]
    errors = []
    if not _text(sample.get("sample_id")):
        errors.append("missing_sample_id")
    source = sample.get("source")
    if not isinstance(source, dict) or source.get("kind") != kind:
        errors.append("source_kind_must_match_collection")
    elif kind == "constructed":
        if not _text(source.get("description")):
            errors.append("missing_construction_description")
    else:
        for key in ("recording_id", "artifact", "contact_source"):
            if not _text(source.get(key)):
                errors.append(f"missing_source_{key}")
        digest = source.get("artifact_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            errors.append("invalid_source_artifact_sha256")
    if not _vector(sample.get("object_centroid")):
        errors.append("invalid_object_centroid")
    radius = sample.get("object_radius")
    # The live implementation falls back for a zero radius. An offline study
    # must retain the invalid case rather than silently change its geometry.
    if not _finite(radius) or radius < 1e-9:
        errors.append("object_radius_must_be_at_least_1e-9")
    contacts = sample.get("contacts")
    if not isinstance(contacts, list) or len(contacts) < 2:
        errors.append("need_at_least_two_contacts")
    else:
        for index, contact in enumerate(contacts):
            if not isinstance(contact, dict) or not _vector(contact.get("point")):
                errors.append(f"invalid_contact_point:{index}")
            normal = contact.get("inward_normal") if isinstance(contact, dict) else None
            if not _vector(normal) or math.hypot(*normal) < 1e-9:
                errors.append(f"invalid_contact_normal:{index}")
    return errors


def _evaluate(sample, mu, samples_per_contact):
    try:
        result = compute_epsilon_quality(
            sample["contacts"],
            sample["object_centroid"],
            sample["object_radius"],
            mu=mu,
            samples_per_contact=samples_per_contact,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return {
            "status": "invalid_geometry",
            "epsilon": None,
            "force_closure": None,
            "reasons": [str(exc)],
        }
    if result.get("error"):
        status = (
            "unavailable" if "scipy is required" in result["error"] else "degenerate"
        )
        return {
            "status": status,
            "epsilon": None,
            "force_closure": None,
            "reasons": [result["error"].splitlines()[0]],
        }
    epsilon = result.get("epsilon")
    if not _finite(epsilon):
        return {
            "status": "invalid_geometry",
            "epsilon": None,
            "force_closure": None,
            "reasons": ["nonfinite_epsilon"],
        }
    return {
        "status": "valid",
        "epsilon": float(epsilon),
        "force_closure": bool(result["force_closure"]),
        "reasons": [],
        "num_wrench_points": result["num_wrench_points"],
    }


def _rotate(vector, angle):
    x, y = vector
    return [
        math.cos(angle) * x - math.sin(angle) * y,
        math.sin(angle) * x + math.cos(angle) * y,
    ]


def _control_samples(sample):
    """Deterministic controls, preserving every original sample unchanged."""
    translated = copy.deepcopy(sample)
    scaled = copy.deepcopy(sample)
    rotated = copy.deepcopy(sample)
    radius_only = copy.deepcopy(sample)
    normal_tilt = copy.deepcopy(sample)
    translation_factors = [17.0, -11.0]
    translation = [v * sample["object_radius"] for v in translation_factors]
    factor, angle, tilt = 3.0, 0.7, 0.25
    translated["object_centroid"] = [
        v + d for v, d in zip(sample["object_centroid"], translation)
    ]
    scaled["object_centroid"] = [factor * v for v in sample["object_centroid"]]
    scaled["object_radius"] *= factor
    rotated["object_centroid"] = _rotate(sample["object_centroid"], angle)
    for index, contact in enumerate(sample["contacts"]):
        translated["contacts"][index]["point"] = [
            v + d for v, d in zip(contact["point"], translation)
        ]
        scaled["contacts"][index]["point"] = [factor * v for v in contact["point"]]
        rotated["contacts"][index]["point"] = _rotate(contact["point"], angle)
        rotated["contacts"][index]["inward_normal"] = _rotate(
            contact["inward_normal"], angle
        )
    radius_only["object_radius"] *= 2.0
    normal_tilt["contacts"][0]["inward_normal"] = _rotate(
        sample["contacts"][0]["inward_normal"], tilt
    )
    return [
        (
            "translation",
            translated,
            True,
            {"translation_in_radius_units": translation_factors},
        ),
        ("uniform_scale", scaled, True, {"factor": factor}),
        ("rotation", rotated, True, {"angle_rad": angle}),
        ("radius_only", radius_only, False, {"factor": 2.0}),
        ("first_normal_tilt", normal_tilt, False, {"angle_rad": tilt}),
    ]


def _distribution(rows, baseline, tolerance):
    valid = [row["reference"] for row in rows if row["reference"]["status"] == "valid"]
    values = [item["epsilon"] for item in valid]
    result = {
        "sample_count": len(rows),
        "valid_count": len(values),
        "invalid_or_unavailable_count": len(rows) - len(values),
        "force_closure_count": sum(item["force_closure"] for item in valid),
        "epsilon": {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "population_std": None,
            "range": None,
            "near_constant": None,
        },
        "constant_baseline": {
            "value": baseline,
            "selection": "fixed_user_parameter_not_fitted",
            "population_std": 0.0 if values else None,
            "mean_absolute_difference_from_epsilon": None,
            "root_mean_square_difference_from_epsilon": None,
            "comparison_target": "computed_epsilon_not_independent_ground_truth",
        },
    }
    if values:
        spread = max(values) - min(values)
        differences = [v - baseline for v in values]
        difference_scale = max(abs(v) for v in differences)
        rms_difference = (
            difference_scale
            * math.sqrt(
                statistics.mean((v / difference_scale) ** 2 for v in differences)
            )
            if difference_scale
            else 0.0
        )
        result["epsilon"] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "population_std": statistics.pstdev(values) if len(values) > 1 else None,
            "range": spread,
            "near_constant": spread <= tolerance if len(values) > 1 else None,
        }
        result["constant_baseline"].update(
            {
                "mean_absolute_difference_from_epsilon": statistics.mean(
                    abs(v - baseline) for v in values
                ),
                "root_mean_square_difference_from_epsilon": rms_difference,
            }
        )
    return result


def analyze_epsilon_sensitivity(
    constructed_samples=(),
    recorded_samples=(),
    *,
    reference_mu=0.5,
    mu_values=(0.1, 0.5, 1.0),
    samples_per_contact=8,
    constant_baseline=0.05,
    tolerance=1e-9,
):
    """Compare geometry estimates and controls; provenance remains user-declared.

    Recorded rows are never expanded into synthetic "recordings". Controls and
    friction sweeps remain separate from each collection's reference distribution.
    Invalid observations retain null epsilon, including degenerate wrench hulls.
    """
    if not isinstance(constructed_samples, (list, tuple)) or not isinstance(
        recorded_samples, (list, tuple)
    ):
        raise TypeError("sample collections must be lists or tuples")
    if not _finite(reference_mu) or reference_mu < 0:
        raise ValueError("reference_mu must be finite and nonnegative")
    if (
        not isinstance(mu_values, (list, tuple))
        or not mu_values
        or any(not _finite(mu) or mu < 0 for mu in mu_values)
    ):
        raise ValueError("mu_values must contain finite nonnegative numbers")
    if (
        not isinstance(samples_per_contact, int)
        or isinstance(samples_per_contact, bool)
        or samples_per_contact < 2
    ):
        raise ValueError("samples_per_contact must be an integer >= 2")
    if not _finite(constant_baseline) or constant_baseline < 0:
        raise ValueError("constant_baseline must be finite and nonnegative")
    if not _finite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    report = {
        "schema": SCHEMA,
        "analysis_scope": "offline_planar_geometry_only",
        "physical_validation": False,
        "provenance_verification": "user_declared_not_artifact_verified",
        "canonical_metric": "grasp_project.realsense.epsilon_wrench.compute_epsilon_quality",
        "parameters": {
            "reference_mu": reference_mu,
            "mu_values": list(mu_values),
            "samples_per_contact": samples_per_contact,
            "constant_baseline": constant_baseline,
            "absolute_tolerance": tolerance,
        },
        "limitations": [
            "Constructed invariance is a geometric control, not a recorded grasp distribution.",
            "Recorded results recompute supplied contact models; provenance declarations are not verified here.",
            "Baseline differences are not accuracy, success rates, or sample adequacy evidence.",
            "No independent physical outcomes or hardware measurements are evaluated.",
        ],
    }
    for kind, samples in (
        ("constructed", constructed_samples),
        ("recorded", recorded_samples),
    ):
        rows = []
        ids = [
            sample.get("sample_id") for sample in samples if isinstance(sample, dict)
        ]
        for sample in samples:
            errors = _sample_errors(sample, kind)
            sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
            if _text(sample_id) and ids.count(sample_id) > 1:
                errors.append("duplicate_sample_id")
            row = {
                "sample_id": sample_id,
                "source": copy.deepcopy(sample.get("source"))
                if isinstance(sample, dict)
                else None,
            }
            if errors:
                row.update(
                    {
                        "reference": {
                            "status": "invalid_input",
                            "epsilon": None,
                            "force_closure": None,
                            "reasons": errors,
                        },
                        "friction_sweep": [],
                        "constructed_controls": [],
                    }
                )
                rows.append(row)
                continue
            reference = _evaluate(sample, reference_mu, samples_per_contact)
            row.update(
                {
                    "reference": reference,
                    "friction_sweep": [],
                    "constructed_controls": [],
                }
            )
            row["constant_baseline"] = {
                "value": constant_baseline,
                "epsilon_minus_baseline": reference["epsilon"] - constant_baseline
                if reference["status"] == "valid"
                else None,
            }
            for mu in mu_values:
                row["friction_sweep"].append(
                    {"mu": mu, **_evaluate(sample, mu, samples_per_contact)}
                )
            if kind == "constructed":
                for (
                    name,
                    transformed,
                    invariant,
                    transform_parameters,
                ) in _control_samples(sample):
                    result = _evaluate(transformed, reference_mu, samples_per_contact)
                    comparable = reference["status"] == result["status"] == "valid"
                    delta = (
                        result["epsilon"] - reference["epsilon"] if comparable else None
                    )
                    row["constructed_controls"].append(
                        {
                            "name": name,
                            "parameters": transform_parameters,
                            "expected_geometric_invariance": invariant,
                            "epsilon_delta": delta,
                            "invariance_passed": (
                                abs(delta) <= tolerance
                                and result["force_closure"]
                                == reference["force_closure"]
                            )
                            if invariant and comparable
                            else None,
                            **result,
                        }
                    )
            rows.append(row)
        report[kind] = {
            "interpretation": "constructed_geometry_controls"
            if kind == "constructed"
            else "supplied_recorded_contact_distribution",
            "samples": rows,
            "distribution": _distribution(rows, constant_baseline, tolerance),
        }
        if kind == "recorded":
            contact_sources = sorted(
                {
                    row["source"]["contact_source"]
                    for row in rows
                    if isinstance(row["source"], dict)
                    and _text(row["source"].get("contact_source"))
                }
            )
            report[kind]["mixed_contact_sources"] = len(contact_sources) > 1
            report[kind]["contact_source_distributions"] = {
                source: _distribution(
                    [
                        row
                        for row in rows
                        if isinstance(row["source"], dict)
                        and row["source"].get("contact_source") == source
                    ],
                    constant_baseline,
                    tolerance,
                )
                for source in contact_sources
            }
    return report
