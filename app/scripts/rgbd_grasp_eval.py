import os
import json
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RGB_PATH = "output/captured_frame.png"
DEPTH_PATH = "output/depth_frame.png"
INTRINSICS_PATH = "output/camera_intrinsics.json"
CONTACTS_PATH = "output/contact_points.json"
OBJECT_MASK_PATH = "output/object_mask.png"

CONTACT_DEPTH_THRESHOLD_MM = 15.0
CONTACT_PIXEL_THRESHOLD_PX = 20.0
MIN_VALID_DEPTH_RATIO = 0.25


def depth_patch_stats(depth_img, x, y, window=5):
    h, w = depth_img.shape

    x1 = max(0, x - window)
    x2 = min(w, x + window + 1)
    y1 = max(0, y - window)
    y2 = min(h, y + window + 1)

    patch = depth_img[y1:y2, x1:x2]
    valid = patch[patch > 0]
    valid_ratio = float(len(valid) / max(patch.size, 1))

    if len(valid) == 0:
        return {
            "depth_mm": None,
            "valid_ratio": 0.0,
            "valid_pixels": 0,
            "total_pixels": int(patch.size),
        }

    return {
        "depth_mm": float(np.median(valid)),
        "valid_ratio": valid_ratio,
        "valid_pixels": int(len(valid)),
        "total_pixels": int(patch.size),
    }


def get_depth_mm(depth_img, x, y, window=5):
    return depth_patch_stats(depth_img, x, y, window)["depth_mm"]


def pixel_to_3d(x, y, z_mm, fx, fy, cx, cy):
    z = z_mm / 1000.0
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    return np.array([X, Y, Z])


def load_files():
    rgb_bgr = cv2.imread(RGB_PATH)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read {RGB_PATH}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read {DEPTH_PATH}")

    with open(INTRINSICS_PATH, "r") as f:
        intr = json.load(f)

    with open(CONTACTS_PATH, "r") as f:
        contacts = json.load(f)

    object_mask = None
    if os.path.exists(OBJECT_MASK_PATH):
        object_mask = cv2.imread(OBJECT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if object_mask is not None:
            object_mask = (object_mask > 0).astype(np.uint8)

    return rgb, depth, intr, contacts, object_mask


def point_inside_mask(mask, x, y):
    if mask is None:
        return False
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return bool(mask[int(y), int(x)] > 0)


def shift_gripper_sample_outside_object(obj_x, obj_y, grip_x, grip_y, object_mask, direction_xy=None, min_shift=4, max_steps=25):
    if object_mask is None:
        if direction_xy is None:
            return grip_x, grip_y, False
        direction = np.asarray(direction_xy, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return grip_x, grip_y, False
        direction = direction / norm
        return int(round(grip_x + direction[0] * min_shift)), int(round(grip_y + direction[1] * min_shift)), True

    if direction_xy is not None:
        direction = np.asarray(direction_xy, dtype=float)
        norm = np.linalg.norm(direction)
        if norm >= 1e-6:
            direction = direction / norm
            h, w = object_mask.shape[:2]
            for step in range(min_shift, max_steps + 1):
                sx = int(round(grip_x + direction[0] * step))
                sy = int(round(grip_y + direction[1] * step))
                if sx < 0 or sx >= w or sy < 0 or sy >= h:
                    break
                if not point_inside_mask(object_mask, sx, sy):
                    return sx, sy, True

    if not point_inside_mask(object_mask, grip_x, grip_y):
        return grip_x, grip_y, False

    direction = np.array([grip_x - obj_x, grip_y - obj_y], dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    direction = direction / norm

    h, w = object_mask.shape[:2]
    for step in range(1, max_steps + 1):
        sx = int(round(grip_x + direction[0] * step))
        sy = int(round(grip_y + direction[1] * step))
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            break
        if not point_inside_mask(object_mask, sx, sy):
            return sx, sy, True

    return grip_x, grip_y, False


def evaluate_side(side_name, contact_info, depth, intr, object_mask=None):
    obj_x, obj_y = contact_info["object_point_xy"]
    grip_x, grip_y = contact_info.get(
        "gripper_face_point_xy",
        contact_info.get("gripper_point_xy"),
    )

    original_grip_x, original_grip_y = grip_x, grip_y
    grip_inside_object_before = point_inside_mask(object_mask, grip_x, grip_y)
    direction_xy = None if contact_info.get("gripper_depth_sample_shifted") else contact_info.get("gripper_direction_xy")
    grip_x, grip_y, shifted = shift_gripper_sample_outside_object(
        obj_x,
        obj_y,
        grip_x,
        grip_y,
        object_mask,
        direction_xy=direction_xy,
    )
    grip_inside_object_after = point_inside_mask(object_mask, grip_x, grip_y)

    pixel_distance = float(np.linalg.norm(np.array([grip_x - obj_x, grip_y - obj_y], dtype=float)))
    near_same_sample = pixel_distance <= 2.0
    warnings = []
    if near_same_sample:
        warnings.append("RGB-D validation warning: object and gripper depth sampled from same/near-same point.")
        print(warnings[-1])
    if grip_inside_object_before and shifted:
        warnings.append("Gripper sample point shifted outside object mask.")
    elif grip_inside_object_after:
        warnings.append("Gripper sample point lies inside object mask.")

    obj_stats = depth_patch_stats(depth, obj_x, obj_y, window=5)
    grip_stats = depth_patch_stats(depth, grip_x, grip_y, window=5)

    obj_z = obj_stats["depth_mm"]
    grip_z = grip_stats["depth_mm"]

    if obj_z is None or grip_z is None:
        return {
            "valid": False,
            "reason": "Missing valid depth",
            "object_px": (obj_x, obj_y),
            "gripper_px": (grip_x, grip_y),
            "original_gripper_px": (original_grip_x, original_grip_y),
            "pixel_distance_px": pixel_distance,
            "object_depth_mm": obj_z,
            "gripper_depth_mm": grip_z,
            "depth_difference_mm": None,
            "object_depth_valid_ratio": obj_stats["valid_ratio"],
            "gripper_depth_valid_ratio": grip_stats["valid_ratio"],
            "gripper_inside_object_mask": grip_inside_object_after,
            "near_same_sample": near_same_sample,
            "warnings": warnings,
            "object_3d": None,
            "gripper_3d": None
        }

    obj_3d = pixel_to_3d(
        obj_x, obj_y, obj_z,
        intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    )

    grip_3d = pixel_to_3d(
        grip_x, grip_y, grip_z,
        intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    )

    depth_diff = abs(obj_z - grip_z)
    object_depth_ok = obj_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
    gripper_depth_ok = grip_stats["valid_ratio"] >= MIN_VALID_DEPTH_RATIO
    pixel_ok = pixel_distance <= CONTACT_PIXEL_THRESHOLD_PX
    depth_ok = depth_diff <= CONTACT_DEPTH_THRESHOLD_MM
    mask_ok = not grip_inside_object_after
    sample_ok = not near_same_sample
    valid = depth_ok and pixel_ok and object_depth_ok and gripper_depth_ok and mask_ok and sample_ok

    invalid_reasons = []
    if not pixel_ok:
        invalid_reasons.append("Pixel gap too large")
    if not depth_ok:
        invalid_reasons.append("Depth mismatch")
    if not object_depth_ok:
        invalid_reasons.append("Low object depth confidence")
    if not gripper_depth_ok:
        invalid_reasons.append("Low gripper depth confidence")
    if not mask_ok:
        invalid_reasons.append("Gripper sample inside object mask")
    if not sample_ok:
        invalid_reasons.append("Near-same depth sample")

    return {
        "valid": valid,
        "reason": "OK" if valid else ", ".join(invalid_reasons),
        "object_px": (obj_x, obj_y),
        "gripper_px": (grip_x, grip_y),
        "original_gripper_px": (original_grip_x, original_grip_y),
        "pixel_distance_px": pixel_distance,
        "object_depth_mm": obj_z,
        "gripper_depth_mm": grip_z,
        "depth_difference_mm": depth_diff,
        "object_depth_valid_ratio": obj_stats["valid_ratio"],
        "gripper_depth_valid_ratio": grip_stats["valid_ratio"],
        "gripper_inside_object_mask": grip_inside_object_after,
        "near_same_sample": near_same_sample,
        "warnings": warnings,
        "object_3d": obj_3d,
        "gripper_3d": grip_3d
    }


def save_rgbd_contact_check(rgb, results):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(rgb)

    colors = {
        "left": "red",
        "right": "blue"
    }

    for side, res in results.items():
        if res is None:
            continue
        color = colors.get(side, "yellow")

        ox, oy = res["object_px"]
        gx, gy = res["gripper_px"]

        plt.scatter(ox, oy, c=color, s=130, marker="o", label=f"{side} object")
        plt.scatter(gx, gy, c=color, s=90, marker="x", label=f"{side} gripper")
        plt.plot([ox, gx], [oy, gy], color=color, linewidth=2)
        depth_diff = res["depth_difference_mm"]
        depth_text = f"{depth_diff:.1f}mm" if depth_diff is not None else "n/a"
        plt.text(
            gx + 6,
            gy,
            f"{side}\n{res['pixel_distance_px']:.1f}px, {depth_text}",
            color=color,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": color},
        )

    plt.title("RGB-D Contact Validation")
    plt.legend(fontsize=8)
    plt.axis("off")
    plt.savefig("output/rgbd_contact_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rgbd_contact_validation_debug(rgb, results, object_mask=None):
    fig = plt.figure(figsize=(11, 7))
    plt.imshow(rgb)
    if object_mask is not None:
        plt.imshow(object_mask, alpha=0.22)

    colors = {"left": "red", "right": "blue"}
    for side, res in results.items():
        if res is None:
            continue

        color = colors.get(side, "yellow")
        ox, oy = res["object_px"]
        gx, gy = res["gripper_px"]
        ogx, ogy = res.get("original_gripper_px", (gx, gy))

        plt.scatter(ox, oy, c=color, s=150, marker="o", label=f"{side} object contact")
        plt.scatter(gx, gy, c=color, s=120, marker="x", label=f"{side} gripper sample")
        if (ogx, ogy) != (gx, gy):
            plt.scatter(ogx, ogy, c="orange", s=90, marker="+", label=f"{side} original gripper point")
            plt.plot([ogx, gx], [ogy, gy], color="orange", linewidth=1.5, linestyle=":")
        plt.plot([ox, gx], [oy, gy], color=color, linewidth=2)

        depth_diff = res["depth_difference_mm"]
        depth_text = f"{depth_diff:.1f} mm" if depth_diff is not None else "n/a"
        warnings = "; ".join(res.get("warnings", []))
        label = (
            f"{side}: {'valid' if res['valid'] else 'invalid'}\n"
            f"{res['pixel_distance_px']:.1f}px, {depth_text}\n"
            f"obj conf {res['object_depth_valid_ratio']:.2f}, grip conf {res['gripper_depth_valid_ratio']:.2f}"
        )
        if warnings:
            label += f"\n{warnings}"
        plt.text(
            gx + 8,
            gy,
            label,
            color=color,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": color},
        )

    plt.title("RGB-D Contact Validation Debug")
    plt.legend(fontsize=7, loc="lower right")
    plt.axis("off")
    plt.savefig("output/rgbd_contact_validation_debug.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_depth_colormap(depth, results):
    depth_float = depth.astype(np.float32)

    # Keep only useful RealSense working range for this tabletop setup
    valid = (depth_float > 50) & (depth_float < 1000)

    masked = np.where(valid, depth_float, np.nan)

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(masked, cmap="viridis", vmin=50, vmax=1000)
    plt.colorbar(label="Depth (mm)")

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(masked, cmap="viridis")
    plt.colorbar(label="Depth (mm)")

    for side, res in results.items():
        if res is None:
            continue
        ox, oy = res["object_px"]
        gx, gy = res["gripper_px"]

        color = "red" if side == "left" else "blue"

        plt.scatter(ox, oy, c=color, s=130, marker="o")
        plt.scatter(gx, gy, c=color, s=90, marker="x")

    plt.title("Depth Map with Contact Points")
    plt.axis("off")
    plt.savefig("output/depth_colormap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rgbd_quality_matrix(results):
    left = results.get("left")
    right = results.get("right")

    left_valid = left["valid"] if left else False
    right_valid = right["valid"] if right else False

    if left and right and left["object_3d"] is not None and right["object_3d"] is not None:
        grasp_width_mm = np.linalg.norm(right["object_3d"] - left["object_3d"]) * 1000.0
    else:
        grasp_width_mm = 0.0

    final_valid = left_valid and right_valid

    fig = plt.figure(figsize=(8, 7))
    plt.axis("off")

    def contact_text(label, result, valid):
        if result is None:
            return f"""
{label} Contact
Object px: n/a
Gripper sample px: n/a
Pixel distance: n/a
Object depth: n/a
Gripper depth: n/a
Depth difference: n/a
Object depth confidence: n/a
Gripper depth confidence: n/a
Validation: Missing contact
"""

        object_depth = result["object_depth_mm"] if result["object_depth_mm"] is not None else 0
        gripper_depth = result["gripper_depth_mm"] if result["gripper_depth_mm"] is not None else 0
        depth_difference = result["depth_difference_mm"] if result["depth_difference_mm"] is not None else 0
        warnings = "; ".join(result.get("warnings", []))
        return f"""
{label} Contact
Object px: {result['object_px']}
Gripper sample px: {result['gripper_px']}
Pixel distance: {result['pixel_distance_px']:.2f} px
Object depth: {object_depth:.2f} mm
Gripper depth: {gripper_depth:.2f} mm
Depth difference: {depth_difference:.2f} mm
Object depth confidence: {result['object_depth_valid_ratio']:.2f}
Gripper depth confidence: {result['gripper_depth_valid_ratio']:.2f}
Validation: {"Valid" if valid else "Invalid"}
{warnings}
"""

    text = f"""
RGB-D Grasp Evaluation
{contact_text("Left", left, left_valid)}
{contact_text("Right", right, right_valid)}

3D Metrics
3D grasp width: {grasp_width_mm:.2f} mm
Contact threshold: {CONTACT_DEPTH_THRESHOLD_MM:.2f} mm

Final RGB-D Contact Status:
{"Valid physical grasp contact" if final_valid else "Invalid / uncertain physical contact"}
"""

    plt.text(0.04, 0.98, text, fontsize=11.5, va="top")
    plt.savefig("output/rgbd_quality_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("output", exist_ok=True)

    rgb, depth, intr, contacts, object_mask = load_files()

    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    print("Intrinsics:", intr)
    print("Loaded contact points:", contacts)

    results = {}

    for side in ["left", "right"]:
        if side in contacts:
            results[side] = evaluate_side(side, contacts[side], depth, intr, object_mask)

    for side, res in results.items():
        print(f"\n{side.upper()} CONTACT")
        print("Object px:", res["object_px"])
        print("Gripper px:", res["gripper_px"])
        print("Object depth mm:", res["object_depth_mm"])
        print("Gripper depth mm:", res["gripper_depth_mm"])
        print("Depth difference mm:", res["depth_difference_mm"])
        print("Pixel distance px:", res["pixel_distance_px"])
        print("Object depth confidence:", res["object_depth_valid_ratio"])
        print("Gripper depth confidence:", res["gripper_depth_valid_ratio"])
        if res.get("warnings"):
            print("Warnings:", "; ".join(res["warnings"]))
        print("Validation:", res["valid"], "-", res["reason"])

    save_rgbd_contact_check(rgb, results)
    save_rgbd_contact_validation_debug(rgb, results, object_mask)
    save_depth_colormap(depth, results)
    save_rgbd_quality_matrix(results)

    print("\nSaved:")
    print("output/rgbd_quality_matrix.png")
    print("output/rgbd_contact_check.png")
    print("output/rgbd_contact_validation_debug.png")
    print("output/depth_colormap.png")


if __name__ == "__main__":
    main()
