"""Camera-free checks for the live RealSense overlay text policy."""

from pathlib import Path

import numpy as np

from grasp_project.realsense import realtime_runner


SOURCE = Path("src/grasp_project/realsense/realtime_runner.py").read_text()
DRAW_FAST_OVERLAY = SOURCE.split("def draw_fast_overlay", 1)[1].split("def save_latest_dashboard", 1)[0]


def test_fast_overlay_text_is_operator_minimal():
    assert "text_lines = [primary]" in DRAW_FAST_OVERLAY
    assert "Q waiting" not in DRAW_FAST_OVERLAY
    assert "fps_short" not in DRAW_FAST_OVERLAY
    assert "Track {" not in DRAW_FAST_OVERLAY


def test_temporal_locked_band_keeps_one_visual_identity_through_depth_dropout():
    assert "band_invalid = bool(" in DRAW_FAST_OVERLAY
    assert 'lock.get("grasp_band_relock_required")' in DRAW_FAST_OVERLAY
    band_invalid_block = DRAW_FAST_OVERLAY.split("band_invalid = bool(", 1)[1].split(")", 1)[0]
    assert 'projection_mode == "INVALID_3D"' not in band_invalid_block
    assert 'report["grasp_band_display_continuity"]' in DRAW_FAST_OVERLAY
    assert '"BOTTLE_RELATIVE_GUIDE_HELD_SAFETY_BLOCKED"' in DRAW_FAST_OVERLAY
    assert 'if layers["band"] and not band_invalid:' in DRAW_FAST_OVERLAY
    assert "draw_dashed_polygon(patch, band_color, 2)" in DRAW_FAST_OVERLAY


def test_renderer_never_falls_back_to_stale_clamp_after_invalid_projection():
    clamp = {
        "clamp_center_px": [80.0, 60.0],
        "clamp_band_polygon": [[50, 54], [110, 54], [110, 66], [50, 66]],
        "grip_width_px": 60.0,
    }

    def render(mode, *, display_clamp=None, current_guide=False):
        tracking = {
            "locked_projection_mode": mode,
            "projection_mode": mode,
            "projection_valid": mode == "TRUE_3D",
            "display_target_clamp": display_clamp or {},
            "operator_locked_fraction_visual_guide_current": current_guide,
        }
        report = {
            "target_clamp": clamp,
            "target_lock": {
                "target_status": "TRACKING",
                "grasp_band_locked": True,
                "current_tracked_target": tracking,
                "locked_target": {"valid": True},
            },
        }
        realtime_runner.draw_fast_overlay(
            np.zeros((120, 160, 3), dtype=np.uint8),
            None,
            {},
            report,
        )
        return report

    invalid = render("INVALID_3D")
    assert invalid["grasp_band_operator_guide_visible_during_depth_dropout"] is False
    assert invalid["grasp_band_display_continuity"] == "UNAVAILABLE_INVALID_3D"
    assert invalid["overlay_primary_status"] == "WAIT DEPTH"

    current = render(
        "INVALID_3D",
        display_clamp=clamp,
        current_guide=True,
    )
    assert current["grasp_band_operator_guide_visible_during_depth_dropout"] is True
    assert current["grasp_band_display_continuity"] == (
        "CURRENT_BOTTLE_RELATIVE_GUIDE_DISPLAY_ONLY"
    )
    assert current["overlay_primary_status"] == "WAIT DEPTH"

    held = render("HELD_LAST_VALID_3D", display_clamp=clamp)
    assert held["grasp_band_operator_guide_visible_during_depth_dropout"] is True
    assert held["grasp_band_display_continuity"] == (
        "BOTTLE_RELATIVE_GUIDE_HELD_SAFETY_BLOCKED"
    )
    assert held["overlay_primary_status"] == "HELD"

    true_3d = render("TRUE_3D")
    assert true_3d["grasp_band_operator_guide_visible_during_depth_dropout"] is True
    assert true_3d["overlay_primary_status"] == "LOCKED"


def test_no_lock_search_frame_never_claims_actionable_grasp_guidance():
    report = {
        "state": "FIND_FULL_BOTTLE",
        "decision": "SEARCH",
        "target_status": "SEARCHING",
        "target_lock": {
            "target_status": "SEARCHING",
            "grasp_band_locked": False,
        },
    }

    realtime_runner.draw_fast_overlay(
        np.zeros((120, 160, 3), dtype=np.uint8),
        None,
        {},
        report,
    )

    assert report["overlay_primary_status"] == "FIND BOTTLE"
    text_width = realtime_runner.cv2.getTextSize(
        report["overlay_primary_status"],
        realtime_runner.cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        1,
    )[0][0]
    assert 16 + text_width <= 142
    assert report["overlay_primary_status"] not in {
        "ADJUST",
        "APPROACH",
        "CLOSE",
        "LOCKED",
    }


def test_top_grasp_policy_is_labeled_planned_until_contact_geometry_exists():
    assert 'strategy == "TOP_GRASP_ROTATE_UPRIGHT"' in DRAW_FAST_OVERLAY
    assert '"PLANNED"' in DRAW_FAST_OVERLAY
    assert '"top_approach_supported_by_task_policy"' in DRAW_FAST_OVERLAY


def test_presentation_hides_rejected_prelock_fragments_but_engineering_keeps_them():
    assert 'not layers["diagnostics"]' in DRAW_FAST_OVERLAY
    assert 'candidate.get("selected") is not True' in DRAW_FAST_OVERLAY
    assert 'candidate.get("eligible") is not True' in DRAW_FAST_OVERLAY
