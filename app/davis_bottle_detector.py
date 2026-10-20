import os
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
USE_DAVIS_YOLO_DETECTOR = os.environ.get("USE_DAVIS_YOLO_DETECTOR", "1").strip().lower() in {"1", "true", "yes", "on"}
DAVIS_YOLO_MODEL_PATH = Path(os.environ.get("DAVIS_YOLO_MODEL_PATH", str(ROOT / "models" / "davis_known_bottle.pt")))
DAVIS_YOLO_CONF_THRESHOLD = float(os.environ.get("DAVIS_YOLO_CONF_THRESHOLD", "0.5"))

STATUS_DETECTED = "DETECTED"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_NOT_DETECTED = "NOT_DETECTED"

_YOLO_MODEL = None
_YOLO_LOAD_ERROR = None


def _base_result(source="failed", reason="No DAVIS APS detector succeeded."):
    return {
        "detected": False,
        "status": STATUS_NOT_DETECTED,
        "source": source,
        "confidence": 0.0,
        "bbox_xyxy": None,
        "contour": None,
        "mask": None,
        "reason": reason,
        "limitations": [
            "DAVIS APS detection is image-space only.",
            "DAVIS-only mode has no metric depth or grip width.",
            "DAVIS-only mode cannot measure force or pressure.",
        ],
        "yolo_available": False,
        "yolo_model_path": str(DAVIS_YOLO_MODEL_PATH),
        "sam2_available": False,
        "opencv_confidence": None,
    }


def _as_bgr(frame):
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr_u8 = cv2.normalize(arr[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr_u8 = arr[:, :, :3]
        if arr_u8.dtype != np.uint8:
            arr_u8 = cv2.normalize(arr_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return arr_u8.copy()
    return None


def yolo_status():
    global _YOLO_MODEL, _YOLO_LOAD_ERROR
    if not USE_DAVIS_YOLO_DETECTOR:
        return False, "disabled"
    if not DAVIS_YOLO_MODEL_PATH.exists():
        return False, f"model missing: {DAVIS_YOLO_MODEL_PATH}"
    if _YOLO_MODEL is not None:
        return True, None
    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(str(DAVIS_YOLO_MODEL_PATH))
        _YOLO_LOAD_ERROR = None
        return True, None
    except Exception as exc:
        _YOLO_LOAD_ERROR = str(exc)
        return False, _YOLO_LOAD_ERROR


def _detect_yolo(frame_bgr):
    available, error = yolo_status()
    if not available:
        result = _base_result("yolo_aps", f"YOLO unavailable: {error}")
        result["yolo_available"] = False
        return result
    try:
        results = _YOLO_MODEL.predict(frame_bgr, conf=DAVIS_YOLO_CONF_THRESHOLD, verbose=False)
    except Exception as exc:
        result = _base_result("yolo_aps", f"YOLO inference failed: {exc}")
        result["yolo_available"] = False
        return result
    best = None
    for item in results:
        boxes = getattr(item, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            if best is None or conf > best[0]:
                best = (conf, xyxy)
    if best is None:
        result = _base_result("yolo_aps", "YOLO model ran but found no bottle.")
        result["yolo_available"] = True
        return result
    conf, xyxy = best
    confidence = float(np.clip(conf * 100.0, 0.0, 100.0))
    status = STATUS_DETECTED if conf >= DAVIS_YOLO_CONF_THRESHOLD else STATUS_LOW_CONFIDENCE
    return {
        **_base_result("yolo_aps"),
        "detected": status == STATUS_DETECTED,
        "status": status,
        "confidence": round(confidence, 1),
        "bbox_xyxy": [int(round(v)) for v in xyxy],
        "reason": f"YOLO DAVIS APS detector selected bbox with confidence={confidence:.1f}/100.",
        "yolo_available": True,
    }


def _detect_sam2(frame_bgr, sam2_segmenter=None, prompt_box=None, prompt_point=None):
    if sam2_segmenter is None:
        result = _base_result("sam2_aps", "SAM2 segmenter was not supplied.")
        result["sam2_available"] = False
        return result
    available = bool(getattr(sam2_segmenter, "available", False))
    if not available and hasattr(sam2_segmenter, "ensure_loaded"):
        available = bool(sam2_segmenter.ensure_loaded())
    if not available:
        result = _base_result("sam2_aps", f"SAM2 unavailable: {getattr(sam2_segmenter, 'reason', 'not loaded')}")
        result["sam2_available"] = False
        return result
    segment, error = sam2_segmenter.segment(frame_bgr, prompt_point=prompt_point, prompt_box=prompt_box)
    if error or segment is None:
        result = _base_result("sam2_aps", f"SAM2 failed: {error}")
        result["sam2_available"] = True
        return result
    mask = segment.get("mask")
    if mask is None or np.count_nonzero(mask) < 20:
        result = _base_result("sam2_aps", "SAM2 returned an empty/too-small mask.")
        result["sam2_available"] = True
        return result
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result = _base_result("sam2_aps", "SAM2 mask had no contour.")
        result["sam2_available"] = True
        return result
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    score = segment.get("score")
    confidence = float(np.clip((score if score is not None else 0.75) * 100.0, 0.0, 100.0))
    status = STATUS_DETECTED if confidence >= 70.0 else STATUS_LOW_CONFIDENCE
    return {
        **_base_result("sam2_aps"),
        "detected": status == STATUS_DETECTED,
        "status": status,
        "confidence": round(confidence, 1),
        "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
        "contour": contour,
        "mask": mask_u8,
        "reason": f"SAM2 APS segmentation selected mask with confidence={confidence:.1f}/100.",
        "sam2_available": True,
    }


def _from_opencv_result(opencv_result):
    if opencv_result is None:
        return _base_result("opencv_aps", "OpenCV fallback was not supplied.")
    confidence = float(opencv_result.get("davis_detection_confidence", 0.0) or 0.0)
    # OpenCV is a fallback cue, not a trained detector. Keep its confidence
    # honest so it cannot masquerade as a perfect detector.
    confidence = min(confidence, 88.0)
    status = opencv_result.get("davis_detection_status", STATUS_NOT_DETECTED)
    if confidence < 70.0:
        status = STATUS_LOW_CONFIDENCE if confidence >= 40.0 else STATUS_NOT_DETECTED
    return {
        **_base_result("opencv_aps"),
        "detected": status == STATUS_DETECTED,
        "status": status,
        "confidence": round(confidence, 1),
        "bbox_xyxy": opencv_result.get("davis_bbox_xyxy"),
        "reason": (
            f"OpenCV APS fallback selected bbox with capped confidence={confidence:.1f}/100. "
            f"Raw heuristic reason: {opencv_result.get('detection_reason', 'OpenCV APS fallback result.')}"
        ),
        "limitations": _base_result()["limitations"] + ["OpenCV APS fallback is heuristic and may be low confidence."],
        "opencv_confidence": round(confidence, 1),
        "opencv_result": opencv_result,
        "yolo_available": yolo_status()[0],
    }


def detect_bottle_aps(frame, *, opencv_fallback=None, sam2_segmenter=None, prompt_box=None, prompt_point=None, allow_sam2=False):
    frame_bgr = _as_bgr(frame)
    if frame_bgr is None:
        return _base_result("failed", "No APS frame available.")

    yolo = _detect_yolo(frame_bgr)
    if yolo["status"] == STATUS_DETECTED:
        return yolo

    if allow_sam2:
        sam2 = _detect_sam2(frame_bgr, sam2_segmenter=sam2_segmenter, prompt_box=prompt_box, prompt_point=prompt_point)
        if sam2["status"] == STATUS_DETECTED:
            sam2["yolo_available"] = yolo.get("yolo_available", False)
            sam2["yolo_model_path"] = str(DAVIS_YOLO_MODEL_PATH)
            return sam2

    opencv_result = opencv_fallback() if callable(opencv_fallback) else None
    opencv = _from_opencv_result(opencv_result)
    opencv["yolo_available"] = yolo.get("yolo_available", False)
    opencv["yolo_model_path"] = str(DAVIS_YOLO_MODEL_PATH)
    opencv["sam2_available"] = bool(sam2_segmenter is not None and getattr(sam2_segmenter, "available", False))
    opencv["detector_fallback_reason"] = yolo.get("reason")
    if allow_sam2 and opencv.get("status") != STATUS_DETECTED:
        opencv["detector_fallback_reason"] = f"{yolo.get('reason')}; SAM2 did not produce a detected mask."
    return opencv
