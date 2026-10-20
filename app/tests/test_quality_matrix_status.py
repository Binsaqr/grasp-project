"""Camera-free checks for quality matrix helper behavior."""

import numpy as np

from epsilon_quality import compute_epsilon_quality, sample_friction_cone, unit_vector


def test_unit_vector_normalizes_nonzero_vector():
    out = unit_vector([3.0, 4.0])
    assert np.allclose(out, [0.6, 0.8])


def test_friction_cone_samples_unit_vectors():
    samples = sample_friction_cone([1.0, 0.0], mu=0.5, samples_per_contact=4)
    assert len(samples) == 4
    assert all(np.isclose(np.linalg.norm(v), 1.0) for v in samples)


def test_compute_epsilon_quality_handles_missing_contacts():
    result = compute_epsilon_quality([], [0.0, 0.0], 1.0)
    assert result["force_closure"] is False
    assert result["epsilon"] == 0.0
    assert result["error"]

