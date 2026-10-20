"""Epsilon/wrench-space grasp quality helpers.

Purpose: compute planar contact wrench samples, convex-hull epsilon quality,
force-closure status, and related quality visualizations.
Main demo: RealSense quality matrix path in the final dashboard.
Phase note: migrated in Phase 3; top-level `epsilon_quality.py` is a
compatibility wrapper.
"""

import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - exercised when scipy is absent
    ConvexHull = None
    QhullError = Exception


SCIPY_INSTALL_MESSAGE = (
    "scipy is required for the true ConvexHull epsilon metric. "
    "Install it with one of:\n"
    "  pip install scipy\n"
    "  sudo apt install python3-scipy"
)


@dataclass
class Contact2D:
    point: np.ndarray
    inward_normal: np.ndarray


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def rotate_2d(vector, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y = vector
    return np.array([c * x - s * y, s * x + c * y], dtype=float)


def _as_contact(contact):
    point = np.asarray(contact["point"], dtype=float)
    inward_normal = unit_vector(contact["inward_normal"])
    return Contact2D(point=point, inward_normal=inward_normal)


def sample_friction_cone(inward_normal, mu=0.5, samples_per_contact=8):
    inward_normal = unit_vector(inward_normal)
    if np.linalg.norm(inward_normal) < 1e-9:
        return []

    half_angle = math.atan(max(0.0, float(mu)))
    samples = max(2, int(samples_per_contact))
    angles = np.linspace(-half_angle, half_angle, samples)
    return [unit_vector(rotate_2d(inward_normal, angle)) for angle in angles]


def build_wrench_points(
    contacts,
    object_centroid,
    object_radius,
    mu=0.5,
    samples_per_contact=8,
):
    centroid = np.asarray(object_centroid, dtype=float)
    radius = float(object_radius) if object_radius else 0.0
    if radius < 1e-9:
        radius = 1.0

    wrenches = []
    normalized_contacts = []

    for raw_contact in contacts:
        contact = _as_contact(raw_contact)
        normalized_contacts.append(contact)

        r = (contact.point - centroid) / radius
        for force in sample_friction_cone(contact.inward_normal, mu, samples_per_contact):
            fx, fy = force
            tau = r[0] * fy - r[1] * fx
            wrenches.append([float(fx), float(fy), float(tau)])

    return np.asarray(wrenches, dtype=float), normalized_contacts


def compute_epsilon_quality(
    contacts,
    object_centroid,
    object_radius,
    mu=0.5,
    samples_per_contact=8,
    tolerance=1e-9,
):
    """
    Compute a planar 2D grasp epsilon metric in 3D wrench space [Fx, Fy, Tau].

    Inputs use image-plane XY coordinates and inward object surface normals.
    """
    base_result = {
        "epsilon": 0.0,
        "force_closure": False,
        "wrench_points": [],
        "hull_vertices": [],
        "hull_simplices": [],
        "hull_equations": [],
        "weakest_direction": None,
        "closest_face_distance": None,
        "closest_face_signed_distance": None,
        "closest_face_equation": None,
        "closest_hull_point": None,
        "error": None,
        "mu": float(mu),
        "num_wrench_points": 0,
    }

    if ConvexHull is None:
        base_result["error"] = SCIPY_INSTALL_MESSAGE
        print(SCIPY_INSTALL_MESSAGE)
        return base_result

    if contacts is None or len(contacts) < 2:
        base_result["error"] = "Need at least two contacts for epsilon quality."
        return base_result

    try:
        wrenches, _ = build_wrench_points(
            contacts,
            object_centroid,
            object_radius,
            mu=mu,
            samples_per_contact=samples_per_contact,
        )
    except Exception as exc:
        base_result["error"] = f"Could not build wrench points: {exc}"
        return base_result

    base_result["wrench_points"] = wrenches.tolist()
    base_result["num_wrench_points"] = int(len(wrenches))

    if len(wrenches) < 4:
        base_result["error"] = "Need at least four wrench points for a 3D hull."
        return base_result

    try:
        hull = ConvexHull(wrenches)
    except QhullError as exc:
        base_result["error"] = f"ConvexHull failed: {exc}"
        return base_result

    equations = np.asarray(hull.equations, dtype=float)
    base_result["hull_vertices"] = hull.vertices.tolist()
    base_result["hull_simplices"] = hull.simplices.tolist()
    base_result["hull_equations"] = equations.tolist()

    normals = equations[:, :3]
    offsets = equations[:, 3]
    normal_norms = np.linalg.norm(normals, axis=1)
    valid_faces = normal_norms > 1e-12

    if not np.any(valid_faces):
        base_result["error"] = "Convex hull has no valid faces."
        return base_result

    valid_normals = normals[valid_faces]
    valid_offsets = offsets[valid_faces]
    valid_norms = normal_norms[valid_faces]
    signed_distances = valid_offsets / valid_norms
    closest_index = int(np.argmin(np.abs(signed_distances)))
    closest_normal = unit_vector(valid_normals[closest_index])
    closest_point = (
        -valid_offsets[closest_index]
        / max(float(np.dot(valid_normals[closest_index], valid_normals[closest_index])), 1e-12)
    ) * valid_normals[closest_index]

    base_result["weakest_direction"] = closest_normal.tolist()
    base_result["closest_face_distance"] = float(abs(signed_distances[closest_index]))
    base_result["closest_face_signed_distance"] = float(signed_distances[closest_index])
    base_result["closest_face_equation"] = equations[valid_faces][closest_index].tolist()
    base_result["closest_hull_point"] = closest_point.tolist()

    signed_origin_values = valid_offsets
    force_closure = bool(np.all(signed_origin_values <= tolerance))
    base_result["force_closure"] = force_closure

    if not force_closure:
        return base_result

    distances = -signed_origin_values / valid_norms
    min_index = int(np.argmin(distances))
    epsilon = max(0.0, float(distances[min_index]))

    weakest = unit_vector(valid_normals[min_index])
    epsilon_touch_point = (
        -valid_offsets[min_index]
        / max(float(np.dot(valid_normals[min_index], valid_normals[min_index])), 1e-12)
    ) * valid_normals[min_index]

    base_result["epsilon"] = epsilon
    base_result["weakest_direction"] = weakest.tolist()
    base_result["closest_face_distance"] = epsilon
    base_result["closest_face_signed_distance"] = float(signed_origin_values[min_index] / valid_norms[min_index])
    base_result["closest_face_equation"] = equations[valid_faces][min_index].tolist()
    base_result["closest_hull_point"] = epsilon_touch_point.tolist()
    return base_result


def _projection_text(result, wrenches, mu):
    epsilon = float(result.get("epsilon", 0.0) or 0.0)
    force_closure = bool(result.get("force_closure", False))
    error = result.get("error")
    weakest = result.get("weakest_direction")

    text = (
        f"epsilon: {epsilon:.4f}\n"
        f"force closure: {force_closure}\n"
        f"wrench points: {int(result.get('num_wrench_points', len(wrenches)))}\n"
        f"mu: {float(result.get('mu', mu)):.3f}"
    )
    if weakest is not None:
        text += (
            "\nweak dir: "
            f"[{weakest[0]:.2f}, {weakest[1]:.2f}, {weakest[2]:.2f}]"
        )
    if not force_closure and not error:
        text += "\nOrigin outside 3D wrench hull"
    if force_closure:
        text += "\nEpsilon is distance to closest 3D face"
    if error:
        text += f"\nstatus: {error.splitlines()[0]}"
    return text


def save_wrench_space_projection(
    result,
    output_path="output/wrench_space_projection.png",
    mu=0.5,
    axes=(0, 1),
    labels=("Fx", "Fy"),
    title="Wrench Space Projection: Fx vs Fy",
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    wrenches = np.asarray(result.get("wrench_points", []), dtype=float)
    epsilon = float(result.get("epsilon", 0.0) or 0.0)
    weakest = result.get("weakest_direction")
    closest_hull_point = result.get("closest_hull_point")
    force_closure = bool(result.get("force_closure", False))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="0.75", linewidth=1)
    ax.axvline(0, color="0.75", linewidth=1)
    ax.scatter([0], [0], c="red", s=90, marker="+", label="origin", zorder=4)

    if wrenches.size:
        points_2d = wrenches[:, axes]
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=32, c="tab:blue", label="wrenches")

        if ConvexHull is not None and len(points_2d) >= 3:
            try:
                hull_2d = ConvexHull(points_2d)
                for simplex in hull_2d.simplices:
                    pts = points_2d[simplex]
                    ax.plot(pts[:, 0], pts[:, 1], color="tab:orange", linewidth=2)
            except QhullError:
                pass

    if epsilon > 0:
        circle = plt.Circle(
            (0, 0),
            epsilon,
            color="tab:green",
            fill=False,
            linewidth=2,
            label="2D projected epsilon circle",
        )
        ax.add_patch(circle)

    if closest_hull_point is not None:
        closest_hull_point = np.asarray(closest_hull_point, dtype=float)
        if closest_hull_point.shape[0] >= 3 and np.all(np.isfinite(closest_hull_point)):
            touch = closest_hull_point[list(axes)]
            ax.plot(
                [0, touch[0]],
                [0, touch[1]],
                color="tab:green",
                linewidth=2.5,
                label="epsilon touch line",
                zorder=5,
            )
            ax.scatter(
                [touch[0]],
                [touch[1]],
                c="tab:green",
                s=140,
                marker="o",
                edgecolors="black",
                linewidths=1.2,
                label="closest hull face / epsilon touch point",
                zorder=6,
            )
            ax.annotate(
                "closest hull face /\nepsilon touch point",
                xy=(touch[0], touch[1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                color="tab:green",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "tab:green"},
            )

    if weakest is not None:
        weakest = np.asarray(weakest, dtype=float)
        arrow = weakest[list(axes)]
        norm = np.linalg.norm(arrow)
        if norm > 1e-9:
            arrow = arrow / norm
            scale = max(epsilon, 0.25)
            ax.arrow(
                0,
                0,
                arrow[0] * scale,
                arrow[1] * scale,
                color="tab:red",
                width=0.006,
                head_width=0.05,
                length_includes_head=True,
                label="weak disturbance direction",
            )

    text = _projection_text(result, wrenches, mu)
    if force_closure:
        text += "\nEpsilon is the 3D distance from origin to the closest hull face."
        text += "\nMarker shows projected touch point."
    else:
        text += "\nOrigin is outside the 3D wrench hull. Epsilon = 0."
    text += "\nCircle is only a 2D projection."
    text += "\n2D projection of 3D planar wrench space [Fx, Fy, Tau]"

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_wrench_space_projections(result, output_dir="output", mu=0.5):
    paths = {
        "fx_fy": os.path.join(output_dir, "wrench_space_fx_fy.png"),
        "fx_tau": os.path.join(output_dir, "wrench_space_fx_tau.png"),
        "fy_tau": os.path.join(output_dir, "wrench_space_fy_tau.png"),
    }
    save_wrench_space_projection(
        result,
        paths["fx_fy"],
        mu,
        axes=(0, 1),
        labels=("Fx", "Fy"),
        title="Wrench Space Projection: Fx vs Fy",
    )
    save_wrench_space_projection(
        result,
        paths["fx_tau"],
        mu,
        axes=(0, 2),
        labels=("Fx", "Tau"),
        title="Wrench Space Projection: Fx vs Tau",
    )
    save_wrench_space_projection(
        result,
        paths["fy_tau"],
        mu,
        axes=(1, 2),
        labels=("Fy", "Tau"),
        title="Wrench Space Projection: Fy vs Tau",
    )
    save_wrench_space_projection(
        result,
        os.path.join(output_dir, "wrench_space_projection.png"),
        mu,
        axes=(0, 1),
        labels=("Fx", "Fy"),
        title="Wrench Space Projection: Fx vs Fy",
    )
    return paths


def epsilon_ball_radius_from_hull_equations(result, tolerance=1e-9):
    equations = np.asarray((result or {}).get("hull_equations", []), dtype=float)
    if equations.size == 0:
        return 0.0, False, "missing_hull_equations"
    try:
        equations = equations.reshape(-1, 4)
    except ValueError:
        return 0.0, False, "invalid_hull_equations"
    normals = equations[:, :3]
    offsets = equations[:, 3]
    normal_norms = np.linalg.norm(normals, axis=1)
    valid = normal_norms > 1e-12
    if not np.any(valid):
        return 0.0, False, "degenerate_wrench_hull"
    origin_values = offsets[valid]
    if not np.all(origin_values <= tolerance):
        return 0.0, False, "origin_outside_hull"
    distances = -origin_values / normal_norms[valid]
    if distances.size == 0 or not np.all(np.isfinite(distances)):
        return 0.0, False, "degenerate_wrench_hull"
    return max(0.0, float(np.min(distances))), True, None


def _project_wrench_3d(points, elev_deg=23.0, azim_deg=-43.0):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    rz = np.array(
        [
            [math.cos(az), -math.sin(az), 0.0],
            [math.sin(az), math.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(el), -math.sin(el)],
            [0.0, math.sin(el), math.cos(el)],
        ],
        dtype=float,
    )
    rotated = pts @ rz.T @ rx.T
    return rotated[:, :2], rotated[:, 2]


def _draw_projected_axis(ax, axis_vector, label, radius, color="#4b5563"):
    end3 = np.asarray(axis_vector, dtype=float) * float(radius)
    end2, _ = _project_wrench_3d([end3])
    ax.arrow(
        0.0,
        0.0,
        float(end2[0, 0]),
        float(end2[0, 1]),
        color=color,
        linewidth=1.6,
        length_includes_head=True,
        head_width=max(radius * 0.035, 0.02),
        head_length=max(radius * 0.055, 0.03),
        zorder=8,
    )
    ax.text(float(end2[0, 0]) * 1.08, float(end2[0, 1]) * 1.08, label, color=color, fontsize=12, weight="bold")


def save_wrench_space_3d_hull(
    result,
    output_path="output/wrench_space_3d_hull.png",
    mu=0.5,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_start = None
    try:
        import time

        plot_start = time.perf_counter()
    except Exception:
        plot_start = None

    wrenches = np.asarray((result or {}).get("wrench_points", []), dtype=float)
    simplices = np.asarray((result or {}).get("hull_simplices", []), dtype=int)
    weakest = (result or {}).get("weakest_direction")
    force_closure = bool((result or {}).get("force_closure", False))
    epsilon_radius, origin_inside, radius_failure = epsilon_ball_radius_from_hull_equations(result)
    if not force_closure:
        epsilon_radius = 0.0
        origin_inside = False
        radius_failure = radius_failure or "origin_outside_hull"

    status = "updated"
    failure_reason = None
    if wrenches.size == 0:
        status = "invalid"
        failure_reason = "missing_wrench_points"
    else:
        try:
            wrenches = wrenches.reshape(-1, 3)
        except ValueError:
            status = "invalid"
            failure_reason = "invalid_wrench_points"
            wrenches = np.zeros((0, 3), dtype=float)
    if status == "updated" and (simplices.size == 0 or simplices.size % 3 != 0):
        status = "invalid"
        failure_reason = "degenerate_wrench_hull"

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_title("Grasp Wrench Space Convex Hull", pad=18)
    ax.axhline(0, color="0.88", linewidth=1.0, zorder=0)
    ax.axvline(0, color="0.88", linewidth=1.0, zorder=0)
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    if status == "updated":
        faces = []
        for simplex in simplices.reshape(-1, 3):
            if np.all(simplex >= 0) and np.all(simplex < len(wrenches)):
                faces.append(wrenches[simplex])
        if faces:
            projected_faces = []
            for face in faces:
                face_2d, face_depth = _project_wrench_3d(face)
                projected_faces.append((float(np.mean(face_depth)), face_2d))
            for _, face_2d in sorted(projected_faces, key=lambda item: item[0]):
                ax.add_patch(
                    Polygon(
                        face_2d,
                        closed=True,
                        facecolor=(0.45, 0.45, 0.45, 0.22),
                        edgecolor=(0.08, 0.08, 0.08, 0.58),
                        linewidth=0.85,
                        zorder=2,
                    )
                )
        else:
            status = "invalid"
            failure_reason = "degenerate_wrench_hull"
        wrench_2d, wrench_depth = _project_wrench_3d(wrenches)
        order = np.argsort(wrench_depth)
        ax.scatter(
            wrench_2d[order, 0],
            wrench_2d[order, 1],
            s=24,
            c="#2563eb",
            edgecolors="white",
            linewidths=0.35,
            label="wrench points",
            zorder=5,
        )

    ax.scatter([0], [0], c="#111827", s=82, marker="o", label="origin", zorder=9)

    if status == "updated" and force_closure and origin_inside and epsilon_radius > 0.0:
        u = np.linspace(0, 2 * np.pi, 80)
        circles = [
            np.column_stack([epsilon_radius * np.cos(u), epsilon_radius * np.sin(u), np.zeros_like(u)]),
            np.column_stack([epsilon_radius * np.cos(u), np.zeros_like(u), epsilon_radius * np.sin(u)]),
            np.column_stack([np.zeros_like(u), epsilon_radius * np.cos(u), epsilon_radius * np.sin(u)]),
        ]
        all_circle = []
        for circle in circles:
            circle_2d, _ = _project_wrench_3d(circle)
            all_circle.append(circle_2d)
            ax.plot(circle_2d[:, 0], circle_2d[:, 1], color="#16a34a", linewidth=1.25, alpha=0.75, zorder=6)
        sphere_pts = np.vstack(all_circle)
        ax.add_patch(
            Polygon(
                sphere_pts,
                closed=True,
                facecolor=(0.13, 0.77, 0.37, 0.18),
                edgecolor="none",
                zorder=4,
            )
        )
    elif status == "updated":
        ax.text(
            0.04,
            0.92,
            "origin outside hull / no force closure" if not force_closure else f"epsilon ball unavailable: {radius_failure}",
            transform=ax.transAxes,
            color="#b42318",
            fontsize=11,
            weight="bold",
        )

    if status == "updated" and weakest is not None:
        weak = np.asarray(weakest, dtype=float)
        if weak.size >= 3 and np.all(np.isfinite(weak[:3])):
            weak = weak[:3]
            norm = float(np.linalg.norm(weak))
            if norm > 1e-9:
                weak = weak / norm
                scale = max(epsilon_radius, 0.28)
                weak_2d, _ = _project_wrench_3d([weak * scale])
                ax.arrow(
                    0.0,
                    0.0,
                    float(weak_2d[0, 0]),
                    float(weak_2d[0, 1]),
                    color="#dc2626",
                    linewidth=2.4,
                    head_width=max(scale * 0.08, 0.025),
                    head_length=max(scale * 0.10, 0.035),
                    length_includes_head=True,
                    label="weak disturbance direction",
                    zorder=10,
                )

    if status != "updated":
        ax.text(
            0.12,
            0.50,
            f"Wrench hull unavailable: {failure_reason or 'invalid_wrench_hull'}",
            transform=ax.transAxes,
            color="#b42318",
            fontsize=13,
            weight="bold",
        )

    plot_points = wrenches if len(wrenches) else np.zeros((1, 3), dtype=float)
    finite = plot_points[np.all(np.isfinite(plot_points), axis=1)] if len(plot_points) else np.zeros((1, 3), dtype=float)
    if finite.size == 0:
        finite = np.zeros((1, 3), dtype=float)
    lo = np.minimum(np.min(finite, axis=0), -epsilon_radius)
    hi = np.maximum(np.max(finite, axis=0), epsilon_radius)
    axis_radius = max(float(np.max(hi - lo)) * 0.62, float(epsilon_radius) * 1.25, 0.35)
    _draw_projected_axis(ax, [1, 0, 0], "Fx", axis_radius)
    _draw_projected_axis(ax, [0, 1, 0], "Fy", axis_radius)
    _draw_projected_axis(ax, [0, 0, 1], "Tau", axis_radius)
    projected_bounds, _ = _project_wrench_3d(
        np.vstack(
            [
                finite,
                np.eye(3) * axis_radius,
                -np.eye(3) * axis_radius,
            ]
        )
    )
    pad = max(axis_radius * 0.22, 0.12)
    ax.set_xlim(float(np.min(projected_bounds[:, 0])) - pad, float(np.max(projected_bounds[:, 0])) + pad)
    ax.set_ylim(float(np.min(projected_bounds[:, 1])) - pad, float(np.max(projected_bounds[:, 1])) + pad)
    caption = (
        f"epsilon radius: {epsilon_radius:.4f} | "
        f"force closure: {bool(force_closure and origin_inside)} | "
        f"mu: {float((result or {}).get('mu', mu)):.2f}"
    )
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=10, color="#344054")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    compute_ms = None
    if plot_start is not None:
        try:
            import time

            compute_ms = round((time.perf_counter() - plot_start) * 1000.0, 2)
        except Exception:
            compute_ms = None
    try:
        simplex_count = int(len(simplices.reshape(-1, 3))) if simplices.size and simplices.size % 3 == 0 else 0
    except Exception:
        simplex_count = 0
    return {
        "status": status,
        "failure_reason": failure_reason or radius_failure,
        "path": output_path,
        "epsilon_ball_radius": epsilon_radius if force_closure and origin_inside else 0.0,
        "force_closure": bool(force_closure and origin_inside),
        "raw_epsilon": result.get("epsilon", 0.0) if result else 0.0,
        "weak_direction": result.get("weakest_direction") if result else None,
        "wrench_point_count": int(len(wrenches)),
        "hull_vertex_count": int(len((result or {}).get("hull_vertices", []) or [])),
        "hull_simplex_count": simplex_count,
        "plot_compute_time_ms": compute_ms,
    }


def save_epsilon_result_json(
    result,
    contacts,
    object_centroid,
    object_radius,
    output_path="output/epsilon_result.json",
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "epsilon": result.get("epsilon", 0.0),
        "force_closure": result.get("force_closure", False),
        "mu": result.get("mu"),
        "num_wrench_points": result.get("num_wrench_points", 0),
        "weakest_direction": result.get("weakest_direction"),
        "closest_face_distance": result.get("closest_face_distance"),
        "closest_face_signed_distance": result.get("closest_face_signed_distance"),
        "closest_face_equation": result.get("closest_face_equation"),
        "closest_hull_point": result.get("closest_hull_point"),
        "wrench_points": result.get("wrench_points", []),
        "hull_vertices": result.get("hull_vertices", []),
        "hull_simplices": result.get("hull_simplices", []),
        "hull_equations": result.get("hull_equations", []),
        "contacts": [
            {
                "point": np.asarray(contact["point"], dtype=float).tolist(),
                "inward_normal": np.asarray(contact["inward_normal"], dtype=float).tolist(),
            }
            for contact in (contacts or [])
        ],
        "object_centroid": np.asarray(object_centroid, dtype=float).tolist(),
        "object_radius": float(object_radius) if object_radius else 0.0,
        "error": result.get("error"),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
