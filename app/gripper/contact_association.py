"""Gripper-object contact association and RGB-D contact candidate scoring."""

from legacy_hybrid_loader import get_symbol


evaluate_contact_candidate = get_symbol("evaluate_contact_candidate")
associate_face_to_object = get_symbol("associate_face_to_object")
build_gripper_contact = get_symbol("build_gripper_contact")
shift_gripper_sample_yx = get_symbol("shift_gripper_sample_yx")
point_inside_mask_yx = get_symbol("point_inside_mask_yx")

