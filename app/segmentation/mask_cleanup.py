"""Mask cleanup, contour, bounding-box, and ROI helpers."""

from legacy_hybrid_loader import get_symbol


mask_boundary = get_symbol("mask_boundary")
mask_bbox = get_symbol("mask_bbox")
clip_roi = get_symbol("clip_roi")
rect_mask = get_symbol("rect_mask")
ordered_mask_contour_yx = get_symbol("ordered_mask_contour_yx")
smooth_closed_points = get_symbol("smooth_closed_points")
contour_roughness = get_symbol("contour_roughness")

