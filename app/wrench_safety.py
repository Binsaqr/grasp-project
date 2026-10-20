import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPSILON_PASS_THRESHOLD = 0.05
FORCE_RESISTANCE_TARGET = 0.25
TORQUE_RESISTANCE_TARGET = 0.25
SCORE_PASS_THRESHOLD = 50.0


def clamp_score(value):
    return float(np.clip(float(value or 0.0), 0.0, 100.0))


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def bidirectional_support(wrenches, direction):
    direction = unit(direction)
    if wrenches.size == 0 or np.linalg.norm(direction) < 1e-9:
        return 0.0
    projections = wrenches @ direction
    positive = float(np.max(projections))
    negative = float(np.max(-projections))
    return max(0.0, min(positive, negative))


def score_support(value, target):
    return clamp_score(100.0 * float(value or 0.0) / max(float(target), 1e-9))


def compute_wrench_safety_summary(epsilon_result):
    force_closure = bool(epsilon_result.get("force_closure", False))
    epsilon_margin = float(epsilon_result.get("epsilon", 0.0) or 0.0)
    weakest = epsilon_result.get("weakest_direction")
    wrenches = np.asarray(epsilon_result.get("wrench_points", []), dtype=float)
    if wrenches.ndim != 2 or wrenches.shape[1] != 3:
        wrenches = np.empty((0, 3), dtype=float)

    fx_support = bidirectional_support(wrenches, np.array([1.0, 0.0, 0.0]))
    fy_support = bidirectional_support(wrenches, np.array([0.0, 1.0, 0.0]))
    tau_support = bidirectional_support(wrenches, np.array([0.0, 0.0, 1.0]))

    push_support = min(fx_support, fy_support)
    push_score = score_support(push_support, FORCE_RESISTANCE_TARGET) if force_closure else 0.0
    slip_score = score_support(fy_support, FORCE_RESISTANCE_TARGET) if force_closure else 0.0
    rotation_score = score_support(tau_support, TORQUE_RESISTANCE_TARGET) if force_closure else 0.0
    epsilon_score = score_support(epsilon_margin, EPSILON_PASS_THRESHOLD) if force_closure else 0.0

    overall_pass = bool(
        force_closure
        and epsilon_margin >= EPSILON_PASS_THRESHOLD
        and push_score >= SCORE_PASS_THRESHOLD
        and slip_score >= SCORE_PASS_THRESHOLD
        and rotation_score >= SCORE_PASS_THRESHOLD
    )

    explanation = (
        "The system computes epsilon in 3D planar wrench space [Fx, Fy, tau]. "
        "This summary converts the existing wrench points into practical resistance scores: "
        "push resistance uses the weaker of horizontal/vertical bidirectional force support, "
        "slip resistance uses vertical Fy support, and rotation resistance uses bidirectional tau support. "
        "The epsilon math itself is unchanged."
    )

    return {
        "force_closure": force_closure,
        "epsilon_margin": epsilon_margin,
        "epsilon_pass_threshold": EPSILON_PASS_THRESHOLD,
        "weakest_disturbance_direction": weakest,
        "weakest_direction_interpretation": interpret_weak_direction(weakest),
        "push_resistance_score": push_score,
        "slip_resistance_score": slip_score,
        "rotation_resistance_score": rotation_score,
        "epsilon_margin_score": epsilon_score,
        "overall_wrench_safety": "PASS" if overall_pass else "FAIL",
        "support_values": {
            "fx_bidirectional_support": fx_support,
            "fy_bidirectional_support": fy_support,
            "tau_bidirectional_support": tau_support,
            "push_support_min_fx_fy": push_support,
        },
        "score_formulas": {
            "push_resistance_score": "clamp(100 * min(Fx_bidirectional_support, Fy_bidirectional_support) / 0.25)",
            "slip_resistance_score": "clamp(100 * Fy_bidirectional_support / 0.25)",
            "rotation_resistance_score": "clamp(100 * Tau_bidirectional_support / 0.25)",
            "epsilon_margin_score": "clamp(100 * epsilon_margin / 0.05)",
            "overall_wrench_safety": "PASS if force_closure true and all scores >= 50 with epsilon >= 0.05",
        },
        "explanation": explanation,
    }


def interpret_weak_direction(weakest):
    if weakest is None:
        return "unknown"
    vector = unit(np.asarray(weakest, dtype=float))
    if vector.size != 3:
        return "unknown"
    labels = ["horizontal push Fx", "vertical slip/push Fy", "rotation tau"]
    idx = int(np.argmax(np.abs(vector)))
    return f"weakest component is mostly {labels[idx]}"


def save_wrench_safety_summary(summary, output_dir="output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "wrench_safety_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    labels = ["Force balance", "Slip resistance", "Rotation resistance", "Epsilon margin"]
    values = [
        float(summary.get("push_resistance_score", 0.0) or 0.0),
        float(summary.get("slip_resistance_score", 0.0) or 0.0),
        float(summary.get("rotation_resistance_score", 0.0) or 0.0),
        float(summary.get("epsilon_margin_score", 0.0) or 0.0),
    ]
    colors = ["#16a34a" if value >= SCORE_PASS_THRESHOLD else "#dc2626" for value in values]
    status = summary.get("overall_wrench_safety", "FAIL")
    status_color = "#166534" if status == "PASS" else "#b91c1c"

    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.88)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.axvline(SCORE_PASS_THRESHOLD, color="#111827", linestyle="--", linewidth=1.4, label="pass threshold")
    for idx, value in enumerate(values):
        ax.text(min(value + 2, 97), idx, f"{value:.0f}/100", va="center", fontsize=11, weight="bold")
    ax.set_xlabel("Practical resistance score")
    ax.set_title(f"Wrench-Space Safety: {status}", fontsize=17, weight="bold", color=status_color)
    subtitle = (
        f"Force closure: {summary.get('force_closure')} | "
        f"Epsilon margin: {float(summary.get('epsilon_margin', 0.0) or 0.0):.4f} | "
        f"Weakest: {summary.get('weakest_direction_interpretation')}"
    )
    ax.text(0.0, 1.05, subtitle, transform=ax.transAxes, fontsize=11, color="#374151")
    ax.text(
        0.0,
        -0.25,
        summary.get("explanation", ""),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
        wrap=True,
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "wrench_safety_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
