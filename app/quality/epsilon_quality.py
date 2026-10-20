"""Planar wrench-space epsilon metric API.

The implementation is re-exported from the existing tested module to preserve
behavior during the staged refactor.
"""

from epsilon_quality import (  # noqa: F401
    Contact2D,
    build_wrench_points,
    compute_epsilon_quality,
    rotate_2d,
    sample_friction_cone,
    save_all_wrench_space_projections,
    save_epsilon_result_json,
    save_wrench_space_projection,
    unit_vector,
)

