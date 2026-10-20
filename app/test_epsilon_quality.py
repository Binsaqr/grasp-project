import os

from epsilon_quality import (
    ConvexHull,
    SCIPY_INSTALL_MESSAGE,
    compute_epsilon_quality,
    save_all_wrench_space_projections,
    save_wrench_space_projection,
)


def require_scipy():
    if ConvexHull is None:
        print(SCIPY_INSTALL_MESSAGE)
        return False
    return True


def test_good_antipodal_grasp():
    if not require_scipy():
        return

    contacts = [
        {"point": (50, 100), "inward_normal": (1, 0)},
        {"point": (150, 100), "inward_normal": (-1, 0)},
    ]

    result = compute_epsilon_quality(
        contacts,
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
        samples_per_contact=12,
    )
    save_wrench_space_projection(result, "output/wrench_space_projection.png")

    assert result["num_wrench_points"] > 0
    assert result["hull_equations"]
    assert result["force_closure"] is True
    assert result["epsilon"] > 0.0
    assert os.path.exists("output/wrench_space_projection.png")


def test_bad_non_opposing_grasp():
    if not require_scipy():
        return

    contacts = [
        {"point": (90, 100), "inward_normal": (1, 0)},
        {"point": (110, 100), "inward_normal": (1, 0)},
    ]

    good = compute_epsilon_quality(
        [
            {"point": (50, 100), "inward_normal": (1, 0)},
            {"point": (150, 100), "inward_normal": (-1, 0)},
        ],
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
        samples_per_contact=12,
    )
    bad = compute_epsilon_quality(
        contacts,
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
        samples_per_contact=12,
    )

    assert bad["epsilon"] <= good["epsilon"]
    assert bad["force_closure"] is False or bad["epsilon"] < good["epsilon"]


def test_asymmetric_contact_geometry():
    if not require_scipy():
        return

    contacts = [
        {"point": (45, 82), "inward_normal": (0.95, 0.22)},
        {"point": (168, 128), "inward_normal": (-0.85, -0.40)},
    ]

    result = compute_epsilon_quality(
        contacts,
        object_centroid=(105, 104),
        object_radius=68,
        mu=0.6,
        samples_per_contact=12,
    )
    save_all_wrench_space_projections(result, "output", mu=0.6)

    assert result["num_wrench_points"] == 24
    assert result["hull_equations"]
    assert result["weakest_direction"] is not None
    assert os.path.exists("output/wrench_space_fx_fy.png")
    assert os.path.exists("output/wrench_space_fx_tau.png")
    assert os.path.exists("output/wrench_space_fy_tau.png")


def test_bad_normals_reduce_quality():
    if not require_scipy():
        return

    good = compute_epsilon_quality(
        [
            {"point": (50, 100), "inward_normal": (1, 0)},
            {"point": (150, 100), "inward_normal": (-1, 0)},
        ],
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
        samples_per_contact=12,
    )
    bad_normals = compute_epsilon_quality(
        [
            {"point": (50, 100), "inward_normal": (0, 1)},
            {"point": (150, 100), "inward_normal": (0, 1)},
        ],
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
        samples_per_contact=12,
    )

    assert bad_normals["epsilon"] <= good["epsilon"]
    assert bad_normals["force_closure"] is False or bad_normals["epsilon"] < good["epsilon"]


def test_missing_contacts_graceful():
    result = compute_epsilon_quality(
        [],
        object_centroid=(100, 100),
        object_radius=50,
        mu=0.5,
    )

    assert result["epsilon"] == 0.0
    assert result["force_closure"] is False
    assert result["error"]


def test_current_pipeline_outputs_if_available():
    if not require_scipy():
        return

    expected_inputs = [
        "output/captured_frame.png",
        "output/depth_frame.png",
        "output/camera_intrinsics.json",
    ]
    missing = [path for path in expected_inputs if not os.path.exists(path)]
    if missing:
        print("Skipping current pipeline sample test; missing:", missing)
        return

    import subprocess

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epsilon-test")

    subprocess.run(
        ["sam_env/bin/python", "scripts/hybrid_opencv_sam.py"],
        check=True,
        env=env,
    )

    assert os.path.exists("output/contact_model.png")
    assert os.path.exists("output/quality_matrix.png")
    assert os.path.exists("output/wrench_space_projection.png")
    assert os.path.exists("output/wrench_space_fx_fy.png")
    assert os.path.exists("output/wrench_space_fx_tau.png")
    assert os.path.exists("output/wrench_space_fy_tau.png")
    assert os.path.exists("output/best_grasp_overlay.png")


if __name__ == "__main__":
    test_good_antipodal_grasp()
    test_bad_non_opposing_grasp()
    test_asymmetric_contact_geometry()
    test_bad_normals_reduce_quality()
    test_missing_contacts_graceful()
    test_current_pipeline_outputs_if_available()
    print("epsilon quality tests completed")
