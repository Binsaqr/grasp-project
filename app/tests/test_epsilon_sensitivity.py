"""Synthetic geometry regressions; these fixtures are not physical validation."""

import copy
import hashlib
import json
import math
import sys

import pytest

from grasp_project.benchmarking import epsilon_sensitivity as sensitivity
from scripts.epsilon_sensitivity import main


def sample(sample_id="antipodal", kind="constructed"):
    source = {"kind": "constructed", "description": "analytic antipodal test fixture"}
    if kind == "recorded":
        # Exercise the recorded schema only. No camera or physical result exists.
        source = {
            "kind": "recorded",
            "recording_id": "schema-fixture-only",
            "artifact": "fixture/epsilon_result.json",
            "artifact_sha256": "a" * 64,
            "contact_source": "synthetic schema fixture; no physical validation",
        }
    return {
        "sample_id": sample_id,
        "source": source,
        "contacts": [
            {"point": [-1.0, 0.0], "inward_normal": [1.0, 0.0]},
            {"point": [1.0, 0.0], "inward_normal": [-1.0, 0.0]},
        ],
        "object_centroid": [0.0, 0.0],
        "object_radius": 1.0,
    }


def test_constructed_controls_are_invariant_only_for_joint_geometry_transforms():
    original = sample()
    before = copy.deepcopy(original)
    report = sensitivity.analyze_epsilon_sensitivity([original])
    row = report["constructed"]["samples"][0]
    assert original == before
    assert row["reference"]["status"] == "valid"
    assert row["reference"]["force_closure"] is True
    # Analytic support at this friction gives epsilon near 1/sqrt(10).
    # The discrete cone chords conservatively approximate that value.
    assert row["reference"]["epsilon"] == pytest.approx(1 / math.sqrt(10), abs=0.002)
    controls = {item["name"]: item for item in row["constructed_controls"]}
    for name in ("translation", "uniform_scale", "rotation"):
        assert controls[name]["expected_geometric_invariance"] is True
        assert controls[name]["invariance_passed"] is True
        assert controls[name]["epsilon_delta"] == pytest.approx(0.0, abs=1e-12)
    for name in ("radius_only", "first_normal_tilt"):
        assert controls[name]["expected_geometric_invariance"] is False
        assert controls[name]["invariance_passed"] is None
        assert abs(controls[name]["epsilon_delta"]) > 0.05
    assert report["physical_validation"] is False
    assert report["recorded"]["distribution"]["sample_count"] == 0


def test_friction_sweep_changes_geometry_estimates_and_exposes_zero_friction_degeneracy():
    row = sensitivity.analyze_epsilon_sensitivity(
        [sample()], mu_values=[0.0, 0.1, 0.5, 1.0]
    )["constructed"]["samples"][0]
    zero, *positive = row["friction_sweep"]
    assert zero["status"] == "degenerate"
    assert zero["epsilon"] is None
    assert zero["force_closure"] is None
    values = [item["epsilon"] for item in positive]
    assert 0 < values[0] < values[1] < values[2]


def test_translation_is_scaled_to_small_geometry_instead_of_losing_precision():
    value = sample()
    value["object_radius"] = 1e-9
    value["contacts"][0]["point"] = [-1e-9, 0]
    value["contacts"][1]["point"] = [1e-9, 0]
    row = sensitivity.analyze_epsilon_sensitivity([value])["constructed"]["samples"][0]
    translation = row["constructed_controls"][0]
    assert translation["parameters"] == {"translation_in_radius_units": [17, -11]}
    assert translation["invariance_passed"] is True


def test_recorded_distribution_is_separate_and_constant_baseline_is_not_fitted():
    closing = sample("closing", "recorded")
    nonclosing = sample("nonclosing", "recorded")
    for contact in nonclosing["contacts"]:
        contact["inward_normal"] = [0.0, 1.0]
    report = sensitivity.analyze_epsilon_sensitivity(
        [sample()], [closing, nonclosing], constant_baseline=0.2
    )
    rows = report["recorded"]["samples"]
    assert all(row["constructed_controls"] == [] for row in rows)
    assert rows[1]["reference"]["status"] == "valid"
    assert rows[1]["reference"]["force_closure"] is False
    assert rows[1]["reference"]["epsilon"] == 0
    summary = report["recorded"]["distribution"]
    assert summary["sample_count"] == 2
    assert summary["force_closure_count"] == 1
    assert summary["epsilon"]["population_std"] > 0.1
    assert summary["epsilon"]["near_constant"] is False
    baseline = summary["constant_baseline"]
    expected = [row["reference"]["epsilon"] for row in rows]
    assert baseline["value"] == 0.2
    assert baseline["population_std"] == 0
    assert baseline["selection"] == "fixed_user_parameter_not_fitted"
    assert baseline["mean_absolute_difference_from_epsilon"] == pytest.approx(
        sum(abs(v - 0.2) for v in expected) / 2
    )
    assert baseline["root_mean_square_difference_from_epsilon"] == pytest.approx(
        math.sqrt(sum((v - 0.2) ** 2 for v in expected) / 2)
    )
    assert (
        baseline["comparison_target"] == "computed_epsilon_not_independent_ground_truth"
    )
    assert report["constructed"]["distribution"]["sample_count"] == 1
    assert report["provenance_verification"] == "user_declared_not_artifact_verified"


def test_identical_contact_models_are_flagged_near_constant_without_sample_adequacy_claim():
    report = sensitivity.analyze_epsilon_sensitivity(
        recorded_samples=[sample("first", "recorded"), sample("second", "recorded")]
    )
    summary = report["recorded"]["distribution"]
    assert summary["epsilon"]["near_constant"] is True
    assert summary["epsilon"]["population_std"] == 0
    assert "sample adequacy" in " ".join(report["limitations"])
    single = sensitivity.analyze_epsilon_sensitivity([sample()])["constructed"][
        "distribution"
    ]
    assert single["epsilon"]["near_constant"] is None
    assert single["epsilon"]["population_std"] is None


@pytest.mark.parametrize(
    "count, baseline_value",
    [(4, 1e308), (6, sys.float_info.max), (9, sys.float_info.max)],
)
def test_finite_large_constant_does_not_overflow_rms_intermediate(
    count, baseline_value
):
    report = sensitivity.analyze_epsilon_sensitivity(
        recorded_samples=[sample(str(index), "recorded") for index in range(count)],
        constant_baseline=baseline_value,
    )
    baseline = report["recorded"]["distribution"]["constant_baseline"]
    assert baseline["root_mean_square_difference_from_epsilon"] == pytest.approx(
        baseline_value
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"source": {"kind": "recorded"}}, "source_kind_must_match_collection"),
        ({"source": {"kind": "constructed"}}, "missing_construction_description"),
        ({"object_centroid": [float("nan"), 0]}, "invalid_object_centroid"),
        ({"object_centroid": [0, 0, 0]}, "invalid_object_centroid"),
        ({"object_radius": 0}, "object_radius_must_be_at_least_1e-9"),
        ({"object_radius": -1}, "object_radius_must_be_at_least_1e-9"),
        ({"object_radius": 1e-10}, "object_radius_must_be_at_least_1e-9"),
        ({"object_radius": True}, "object_radius_must_be_at_least_1e-9"),
        ({"contacts": []}, "need_at_least_two_contacts"),
        ({"sample_id": ""}, "missing_sample_id"),
    ],
)
def test_invalid_inputs_are_null_not_zero_or_force_closure(change, reason):
    value = sample()
    value.update(change)
    summary = sensitivity.analyze_epsilon_sensitivity([value])["constructed"]
    row = summary["samples"][0]
    assert row["reference"]["status"] == "invalid_input"
    assert reason in row["reference"]["reasons"]
    assert row["reference"]["epsilon"] is None
    assert row["reference"]["force_closure"] is None
    assert row["friction_sweep"] == []
    assert summary["distribution"]["valid_count"] == 0
    assert summary["distribution"]["epsilon"]["mean"] is None


def test_zero_normals_and_coincident_points_retain_invalid_or_degenerate_status():
    zero = sample("zero")
    zero["contacts"][0]["inward_normal"] = [0, 0]
    coincident = sample("coincident")
    for contact in coincident["contacts"]:
        contact["point"] = [0, 0]
    rows = sensitivity.analyze_epsilon_sensitivity([zero, coincident])["constructed"][
        "samples"
    ]
    assert rows[0]["reference"]["status"] == "invalid_input"
    assert "invalid_contact_normal:0" in rows[0]["reference"]["reasons"]
    assert rows[1]["reference"]["status"] == "degenerate"
    assert rows[1]["reference"]["epsilon"] is None
    assert all(
        control["invariance_passed"] is None
        for control in rows[1]["constructed_controls"]
    )


def test_recorded_provenance_is_required_and_duplicate_ids_are_excluded():
    missing = sample(kind="recorded")
    missing["source"] = {"kind": "recorded"}
    row = sensitivity.analyze_epsilon_sensitivity(recorded_samples=[missing])[
        "recorded"
    ]["samples"][0]
    assert set(row["reference"]["reasons"]) == {
        "missing_source_recording_id",
        "missing_source_artifact",
        "missing_source_contact_source",
        "invalid_source_artifact_sha256",
    }
    rows = sensitivity.analyze_epsilon_sensitivity([sample(), sample()])["constructed"][
        "samples"
    ]
    assert all("duplicate_sample_id" in row["reference"]["reasons"] for row in rows)


def test_different_recorded_contact_sources_are_flagged_and_reported_separately():
    planner, diagnostic = (
        sample("planner", "recorded"),
        sample("diagnostic", "recorded"),
    )
    planner["source"]["contact_source"] = "planner_authoritative"
    diagnostic["source"]["contact_source"] = "visual_diagnostic"
    for contact in diagnostic["contacts"]:
        contact["inward_normal"] = [0, 1]
    report = sensitivity.analyze_epsilon_sensitivity(
        recorded_samples=[planner, diagnostic]
    )["recorded"]
    assert report["mixed_contact_sources"] is True
    groups = report["contact_source_distributions"]
    assert groups["planner_authoritative"]["force_closure_count"] == 1
    assert groups["visual_diagnostic"]["force_closure_count"] == 0
    assert all(group["sample_count"] == 1 for group in groups.values())


def test_missing_scipy_is_unavailable_not_a_valid_zero(monkeypatch):
    monkeypatch.setattr(
        sensitivity,
        "compute_epsilon_quality",
        lambda *args, **kwargs: {
            "epsilon": 0.0,
            "force_closure": False,
            "error": "scipy is required for the true ConvexHull epsilon metric.",
        },
    )
    row = sensitivity.analyze_epsilon_sensitivity([sample()])["constructed"]["samples"][
        0
    ]
    assert row["reference"]["status"] == "unavailable"
    assert row["reference"]["epsilon"] is None


@pytest.mark.parametrize(
    "parameters",
    [
        {"reference_mu": -0.1},
        {"reference_mu": float("nan")},
        {"mu_values": []},
        {"mu_values": [False]},
        {"mu_values": [float("inf")]},
        {"samples_per_contact": 1},
        {"samples_per_contact": 8.0},
        {"constant_baseline": -1},
        {"tolerance": -1},
        {"tolerance": True},
    ],
)
def test_invalid_parameters_fail_explicitly(parameters):
    with pytest.raises(ValueError):
        sensitivity.analyze_epsilon_sensitivity([sample()], **parameters)


def test_cli_writes_deterministic_hash_attributed_report(tmp_path):
    manifest = tmp_path / "samples.json"
    output = tmp_path / "sensitivity.json"
    manifest.write_text(
        json.dumps(
            {"schema": "grasp_epsilon_samples/v1", "constructed_samples": [sample()]}
        )
    )
    assert main([str(manifest), "--output", str(output)]) == 0
    first = output.read_bytes()
    report = json.loads(first)
    assert (
        report["input_manifest"]["sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert main([str(manifest), "--output", str(output)]) == 0
    assert output.read_bytes() == first


def test_cli_retains_invalid_rows_and_returns_nonzero(tmp_path):
    manifest, output = tmp_path / "samples.json", tmp_path / "sensitivity.json"
    value = sample()
    value["object_radius"] = 0
    manifest.write_text(
        json.dumps(
            {"schema": "grasp_epsilon_samples/v1", "constructed_samples": [value]}
        )
    )
    assert main([str(manifest), "--output", str(output)]) == 1
    assert (
        json.loads(output.read_text())["constructed"]["distribution"][
            "invalid_or_unavailable_count"
        ]
        == 1
    )


@pytest.mark.parametrize("passed", [False, None])
def test_cli_returns_failure_for_failed_or_unavailable_expected_invariance(
    tmp_path, monkeypatch, passed
):
    from scripts import epsilon_sensitivity as cli

    report = sensitivity.analyze_epsilon_sensitivity([sample()])
    report["constructed"]["samples"][0]["constructed_controls"][0][
        "invariance_passed"
    ] = passed
    monkeypatch.setattr(
        cli, "analyze_epsilon_sensitivity", lambda *args, **kwargs: report
    )
    manifest, output = tmp_path / "samples.json", tmp_path / "report.json"
    manifest.write_text(json.dumps({"schema": "grasp_epsilon_samples/v1"}))
    assert main([str(manifest), "--output", str(output)]) == 1
    assert output.exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema":"grasp_epsilon_samples/v1","parameters":{"reference_mu":0.1,"reference_mu":0.9}}',
        '{"schema":"grasp_epsilon_samples/v1","parameters":{"reference_mu":NaN}}',
        '{"schema":"grasp_epsilon_samples/v1","parameters":{"reference_mu":Infinity}}',
    ],
)
def test_cli_rejects_ambiguous_or_nonfinite_json(tmp_path, raw):
    manifest, output = tmp_path / "samples.json", tmp_path / "report.json"
    manifest.write_text(raw)
    with pytest.raises(SystemExit) as exc:
        main([str(manifest), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_cli_empty_manifest_reports_absence_with_nonzero_exit(tmp_path):
    manifest, output = tmp_path / "samples.json", tmp_path / "report.json"
    manifest.write_text(json.dumps({"schema": "grasp_epsilon_samples/v1"}))
    assert main([str(manifest), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["recorded"]["distribution"]["epsilon"]["mean"] is None
    assert report["physical_validation"] is False


def test_cli_does_not_overwrite_the_input_manifest(tmp_path):
    manifest = tmp_path / "samples.json"
    raw = json.dumps(
        {"schema": "grasp_epsilon_samples/v1", "constructed_samples": [sample()]}
    )
    manifest.write_text(raw)
    with pytest.raises(SystemExit) as exc:
        main([str(manifest), "--output", str(manifest)])
    assert exc.value.code == 2
    assert manifest.read_text() == raw


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema": "grasp_epsilon_samples/v1", "parameter": {}},
        {"schema": "grasp_epsilon_samples/v1", "parameters": {"typo_mu": 0.5}},
        {"schema": "grasp_epsilon_samples/v1", "parameters": []},
    ],
)
def test_cli_rejects_wrong_schema_or_misspelled_configuration(tmp_path, payload):
    manifest = tmp_path / "samples.json"
    output = tmp_path / "report.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as exc:
        main([str(manifest), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()
