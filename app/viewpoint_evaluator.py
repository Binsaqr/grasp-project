import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
SAMPLE_DATA_DIR = ROOT / "sample_data"
VIEW_ROOTS = [SAMPLE_DATA_DIR / "views", OUTPUT_DIR / "views"]

RGB_NAME = "captured_frame.png"
DEPTH_NAME = "depth_frame.png"
DEPTH_NPY_NAME = "depth_frame.npy"
INTRINSICS_NAME = "camera_intrinsics.json"


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def find_view_dirs(root):
    if not root.exists():
        return []
    views = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        rgb = path / RGB_NAME
        depth = path / DEPTH_NAME
        if rgb.exists() and depth.exists():
            views.append({
                "view_id": path.name,
                "rgb_path": rgb,
                "depth_path": depth,
                "depth_npy_path": path / DEPTH_NPY_NAME,
                "intrinsics_path": path / INTRINSICS_NAME,
                "source": str(root.relative_to(ROOT)),
            })
    return views


def discover_views():
    views = []
    for root in VIEW_ROOTS:
        views.extend(find_view_dirs(root))
    if views:
        return views

    fallback_rgb = SAMPLE_DATA_DIR / RGB_NAME
    fallback_depth = SAMPLE_DATA_DIR / DEPTH_NAME
    if fallback_rgb.exists() and fallback_depth.exists():
        return [{
            "view_id": "sample_default",
            "rgb_path": fallback_rgb,
            "depth_path": fallback_depth,
            "depth_npy_path": SAMPLE_DATA_DIR / DEPTH_NPY_NAME,
            "intrinsics_path": SAMPLE_DATA_DIR / INTRINSICS_NAME,
            "source": "sample_data fallback",
        }]
    return []


def copy_view_to_output(view):
    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.copy2(view["rgb_path"], OUTPUT_DIR / RGB_NAME)
    shutil.copy2(view["depth_path"], OUTPUT_DIR / DEPTH_NAME)
    if view["depth_npy_path"].exists():
        shutil.copy2(view["depth_npy_path"], OUTPUT_DIR / DEPTH_NPY_NAME)
    if view["intrinsics_path"].exists():
        shutil.copy2(view["intrinsics_path"], OUTPUT_DIR / INTRINSICS_NAME)
    elif (SAMPLE_DATA_DIR / INTRINSICS_NAME).exists():
        shutil.copy2(SAMPLE_DATA_DIR / INTRINSICS_NAME, OUTPUT_DIR / INTRINSICS_NAME)


def load_report():
    with open(OUTPUT_DIR / "pregrasp_safety_report.json", "r") as f:
        return json.load(f)


def run_single_view(view):
    from scripts.bottle_grasp_planner import run_bottle_pregrasp

    copy_view_to_output(view)
    try:
        run_bottle_pregrasp()
    except SystemExit:
        pass
    report = load_report()
    epsilon = report.get("epsilon_result", {})
    scores = report.get("scores", {})
    risks = report.get("risk_scores", {})
    hard_gates = report.get("hard_gates", {})
    return {
        "view_id": view["view_id"],
        "source": view["source"],
        "image_path": str(view["rgb_path"]),
        "depth_path": str(view["depth_path"]),
        "safety_score": float(report.get("final_score", 0.0) or 0.0),
        "raw_weighted_score": float(report.get("raw_weighted_score", report.get("final_score", 0.0)) or 0.0),
        "decision": report.get("decision"),
        "full_bottle_visible": bool(report.get("full_bottle_visible", False)),
        "visibility_score": float(report.get("visibility_score", 0.0) or 0.0),
        "recommended_view_action": report.get("recommended_view_action"),
        "contact_model_ready": bool(report.get("contact_model_ready", False)),
        "active_recommended_action": report.get("active_positioning", {}).get("recommended_action"),
        "hard_gates_passed": bool(hard_gates.get("passed", False)),
        "epsilon_value": float(epsilon.get("epsilon", 0.0) or 0.0),
        "force_closure": bool(epsilon.get("force_closure", False)),
        "depth_confidence": float(scores.get("depth_confidence", 0.0) or 0.0),
        "alignment_error_px": report.get("alignment_error_px"),
        "miss_risk": float(risks.get("miss_risk", 100.0) or 100.0),
        "slip_risk": float(risks.get("slip_risk", 100.0) or 100.0),
        "rotation_risk": float(risks.get("rotation_risk", 100.0) or 100.0),
        "failure_reasons": list(report.get("failure_reasons", [])),
        "warnings": list(report.get("warnings", [])),
        "report_excerpt": {
            "bottle_center_px": report.get("bottle_geometry", {}).get("center_px"),
            "grip_width_px": report.get("planned_contacts", {}).get("grip_width_px"),
            "grip_width_mm": report.get("planned_contacts", {}).get("grip_width_mm"),
            "workspace_reachability_score": report.get("planned_contacts", {}).get("workspace_reachability_score"),
        },
        "future_robot_offsets": {
            "center": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "note": "Placeholder for centered approach offset."},
            "left": {"dx": -1.0, "dy": 0.0, "dz": 0.0, "note": "Placeholder for left-side approach offset."},
            "right": {"dx": 1.0, "dy": 0.0, "dz": 0.0, "note": "Placeholder for right-side approach offset."},
            "closer": {"dx": 0.0, "dy": 0.0, "dz": -1.0, "note": "Placeholder for moving closer along camera depth."},
            "higher": {"dx": 0.0, "dy": -1.0, "dz": 0.0, "note": "Placeholder for higher image-frame approach."},
            "lower": {"dx": 0.0, "dy": 1.0, "dz": 0.0, "note": "Placeholder for lower image-frame approach."},
        },
    }


def ranking_key(result):
    combined_risk = result["miss_risk"] + result["slip_risk"] + result["rotation_risk"]
    return (
        1 if result["hard_gates_passed"] else 0,
        result["safety_score"],
        result["depth_confidence"],
        result["epsilon_value"],
        -combined_risk,
    )


def explain_selection(best):
    if best is None:
        return "No views were available for evaluation."
    parts = []
    parts.append("hard gates passed" if best["hard_gates_passed"] else "best available view despite failed hard gates")
    parts.append(f"safety score {best['safety_score']:.1f}/100")
    parts.append(f"depth confidence {best['depth_confidence']:.1f}/100")
    parts.append(f"epsilon {best['epsilon_value']:.4f}")
    total_risk = best["miss_risk"] + best["slip_risk"] + best["rotation_risk"]
    parts.append(f"combined risk {total_risk:.1f}")
    return "Selected by ranking: " + ", ".join(parts) + "."


def save_summary_image(results, best_view_id):
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not results:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.04, 0.55, "No multiview RGB-D frames found.", fontsize=18, weight="bold", color="#b91c1c")
        fig.savefig(OUTPUT_DIR / "multiview_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    cols = min(3, max(1, len(results)))
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, result in zip(axes, results):
        image = cv2.imread(result["image_path"], cv2.IMREAD_COLOR)
        if image is not None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        title_color = "#166534" if result["view_id"] == best_view_id else "#1f2937"
        border_color = "#22c55e" if result["view_id"] == best_view_id else "#94a3b8"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(4 if result["view_id"] == best_view_id else 1.5)
        ax.set_title(
            f"{result['view_id']} | {result['decision']}\n"
            f"score {result['safety_score']:.1f}, eps {result['epsilon_value']:.3f}, "
            f"depth {result['depth_confidence']:.0f}",
            fontsize=11,
            color=title_color,
            weight="bold" if result["view_id"] == best_view_id else "normal",
        )
        reason = "BEST VIEW" if result["view_id"] == best_view_id else ""
        if result["failure_reasons"]:
            reason = result["failure_reasons"][0]
        ax.text(
            0.02,
            0.04,
            reason,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none"},
            wrap=True,
        )
    fig.suptitle("Multi-View Pre-Grasp Evaluation", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multiview_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_multiview():
    views = discover_views()
    if not views:
        message = (
            "No saved RGB-D views found. Add view folders under sample_data/views/ or output/views/ "
            "with captured_frame.png and depth_frame.png."
        )
        report = {
            "mode": "bottle_multiview",
            "views_evaluated": 0,
            "selected_best_view": None,
            "selection_reason": message,
            "view_results": [],
            "future_robot_motion": "No robot motion commands are sent; offsets are placeholders only.",
        }
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(OUTPUT_DIR / "multiview_report.json", "w") as f:
            json.dump(report, f, indent=4)
        save_summary_image([], None)
        print(message)
        return report

    results = []
    for view in views:
        print(f"Evaluating multiview candidate: {view['view_id']} ({view['source']})")
        results.append(run_single_view(view))

    ranked = sorted(results, key=ranking_key, reverse=True)
    best = ranked[0] if ranked else None
    report = {
        "mode": "bottle_multiview",
        "views_evaluated": len(results),
        "ranking_order": [item["view_id"] for item in ranked],
        "selected_best_view": best["view_id"] if best else None,
        "selection_reason": explain_selection(best),
        "rank_formula": [
            "hard_gates_passed",
            "safety_score",
            "depth_confidence",
            "epsilon_value",
            "lowest miss+slip+rotation risk",
        ],
        "view_results": ranked,
        "future_robot_motion": "No robot motion commands are sent; offsets are placeholders only.",
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "multiview_report.json", "w") as f:
        json.dump(json_safe(report), f, indent=4)
    save_summary_image(ranked, report["selected_best_view"])
    print(f"Multi-view evaluation complete. Best view: {report['selected_best_view']}")
    print(report["selection_reason"])
    print("Saved output/multiview_report.json")
    print("Saved output/multiview_summary.png")
    return report


if __name__ == "__main__":
    evaluate_multiview()
