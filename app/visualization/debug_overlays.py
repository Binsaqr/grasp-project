"""Debug overlay and result-image generation helpers."""

from legacy_hybrid_loader import get_symbol


draw_box = get_symbol("draw_box")
draw_gripper_contact_annotations = get_symbol("draw_gripper_contact_annotations")
save_contact_debug = get_symbol("save_contact_debug")
save_gripper_detection_debug = get_symbol("save_gripper_detection_debug")
save_best_grasp_overlay = get_symbol("save_best_grasp_overlay")

