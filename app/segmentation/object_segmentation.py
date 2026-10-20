"""SAM/OpenCV object prompt and segmentation helpers."""

from legacy_hybrid_loader import get_symbol


find_prompt_in_roi = get_symbol("find_prompt_in_roi")
sam_predict = get_symbol("sam_predict")
object_guided_gripper_rois = get_symbol("object_guided_gripper_rois")

