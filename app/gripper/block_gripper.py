"""Block-jaw gripper helpers and inner-face contact selection."""

from legacy_hybrid_loader import get_symbol


bbox_face_points = get_symbol("bbox_face_points")
candidate_face_to_object_distance = get_symbol("candidate_face_to_object_distance")
gripper_body_bbox_status = get_symbol("gripper_body_bbox_status")
gripper_inner_face_points = get_symbol("gripper_inner_face_points")
block_face_overlap_contact = get_symbol("block_face_overlap_contact")
refine_candidate_by_depth = get_symbol("refine_candidate_by_depth")
trim_candidate_to_inner_band = get_symbol("trim_candidate_to_inner_band")

