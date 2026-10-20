"""Top-level gripper-body detection entry points."""

from legacy_hybrid_loader import get_symbol


detect_gripper_body = get_symbol("detect_gripper_body")
generate_gripper_candidates = get_symbol("generate_gripper_candidates")
score_gripper_candidate = get_symbol("score_gripper_candidate")
reliable_block_candidate = get_symbol("reliable_block_candidate")
mode_reason_for_block = get_symbol("mode_reason_for_block")

