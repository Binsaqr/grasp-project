from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


DEFAULT_MODEL_PATHS = (
    "models/yolo_bottle.weights",
    "models/yolo_bottle.onnx",
    "models/yolo_bottle.pt",
    "models/bottle_yolo.pt",
    "yolo_bottle.pt",
)


class YoloBottleDetector:
    def __init__(self, model_path=None, conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.model = None
        self.model_backend = None
        self.available = False
        self.unavailable_reason = None
        self._load()

    def _resolve_model_path(self):
        candidates = [self.model_path] if self.model_path else []
        candidates.extend(DEFAULT_MODEL_PATHS)
        for path in candidates:
            if not path:
                continue
            p = Path(path).expanduser()
            if p.exists():
                return str(p)
        return None

    def _load(self):
        resolved = self._resolve_model_path()
        if resolved is None:
            self.unavailable_reason = "YOLO model file not found."
            return
        if Path(resolved).suffix.lower() == ".weights":
            config_path = Path(resolved).with_suffix(".cfg")
            if not config_path.exists():
                self.unavailable_reason = f"YOLO Darknet config file not found: {config_path}"
                return
            try:
                self.model = cv2.dnn.readNetFromDarknet(str(config_path), resolved)
                self.model_path = resolved
                self.model_backend = "opencv_dnn_darknet"
                self.available = True
            except Exception as exc:
                self.unavailable_reason = f"YOLO Darknet model load failed: {exc}"
            return
        if Path(resolved).suffix.lower() == ".onnx":
            try:
                self.model = cv2.dnn.readNetFromONNX(resolved)
                self.model_path = resolved
                self.model_backend = "opencv_dnn_onnx"
                self.available = True
            except Exception as exc:
                self.unavailable_reason = f"YOLO ONNX model load failed: {exc}"
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.unavailable_reason = f"ultralytics unavailable: {exc}"
            return
        try:
            self.model = YOLO(resolved)
            self.model_path = resolved
            self.model_backend = "ultralytics_pytorch"
            self.available = True
        except Exception as exc:
            self.unavailable_reason = f"YOLO model load failed: {exc}"

    def detect(self, rgb):
        result = {
            "yolo_enabled": True,
            "yolo_available": bool(self.available),
            "yolo_detected": False,
            "bottle_detected": False,
            "bbox_xyxy": None,
            "bbox_xywh": None,
            "confidence": None,
            "class_name": None,
            "detections": [],
            "detection_count": 0,
            "detection_source": "yolo",
            "optional_mask": None,
            "yolo_model_path": self.model_path,
            "yolo_model_backend": getattr(self, "model_backend", None),
            "yolo_inference_ms": None,
            "yolo_unavailable_reason": self.unavailable_reason,
        }
        if not self.available or self.model is None:
            result["detection_source"] = "opencv_depth_fallback"
            return result
        if getattr(self, "model_backend", None) == "opencv_dnn_darknet":
            return self._detect_darknet(rgb, result)
        if getattr(self, "model_backend", None) == "opencv_dnn_onnx":
            return self._detect_onnx(rgb, result)
        started = perf_counter()
        try:
            outputs = self.model.predict(rgb, verbose=False, conf=self.conf_threshold)
        except Exception as exc:
            result["yolo_available"] = False
            result["detection_source"] = "opencv_depth_fallback"
            result["yolo_unavailable_reason"] = f"YOLO inference failed: {exc}"
            return result
        result["yolo_inference_ms"] = 1000.0 * (perf_counter() - started)
        best = None
        best_mask = None
        detections = []
        for output in outputs or []:
            names = getattr(output, "names", {}) or {}
            boxes = getattr(output, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            masks = getattr(output, "masks", None)
            mask_data = None
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else np.asarray(masks.data)
            for idx, box in enumerate(xyxy):
                conf = float(confs[idx])
                class_id = int(classes[idx]) if idx < len(classes) else -1
                class_name = str(names.get(class_id, class_id))
                is_bottle = "bottle" in class_name.lower() or class_id == 39
                if conf < self.conf_threshold or not is_bottle:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                detection = {
                    "bbox_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "bbox_xywh": [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))],
                    "confidence": conf,
                    "class_name": class_name,
                }
                detections.append(detection)
                if best is None or conf > best["confidence"]:
                    best = detection
                    best_mask = mask_data[idx] if mask_data is not None and idx < len(mask_data) else None
        if best is None:
            return result
        result.update(best)
        result["detections"] = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        result["detection_count"] = len(detections)
        result["yolo_detected"] = True
        result["bottle_detected"] = True
        if best_mask is not None:
            result["optional_mask"] = (best_mask > 0.5).astype(np.uint8)
        return result

    def _detect_onnx(self, rgb, result, input_size=640, nms_threshold=0.45):
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            result["yolo_available"] = False
            result["detection_source"] = "opencv_depth_fallback"
            result["yolo_unavailable_reason"] = "YOLO ONNX inference requires an RGB HxWx3 image."
            return result

        image_h, image_w = image.shape[:2]
        scale = min(float(input_size) / max(image_w, 1), float(input_size) / max(image_h, 1))
        resized_w = max(1, int(round(image_w * scale)))
        resized_h = max(1, int(round(image_h * scale)))
        pad_x = (input_size - resized_w) // 2
        pad_y = (input_size - resized_h) // 2
        letterboxed = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        letterboxed[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=False,
            crop=False,
        )
        started = perf_counter()
        try:
            self.model.setInput(blob)
            raw = self.model.forward()
        except Exception as exc:
            result["yolo_available"] = False
            result["detection_source"] = "opencv_depth_fallback"
            result["yolo_unavailable_reason"] = f"YOLO ONNX inference failed: {exc}"
            return result
        result["yolo_inference_ms"] = 1000.0 * (perf_counter() - started)

        predictions = np.asarray(raw)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            result["yolo_unavailable_reason"] = f"Unsupported YOLO ONNX output shape: {tuple(np.asarray(raw).shape)}"
            return result
        if predictions.shape[0] in (84, 85):
            predictions = predictions.T
        if predictions.shape[1] < 84:
            result["yolo_unavailable_reason"] = f"Unsupported YOLO ONNX prediction width: {predictions.shape[1]}"
            return result

        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        candidate_indexes = np.where((class_ids == 39) & (confidences >= self.conf_threshold))[0]
        boxes_xywh = []
        scores = []
        for index in candidate_indexes:
            center_x, center_y, width, height = [float(value) for value in predictions[index, :4]]
            x1 = (center_x - 0.5 * width - pad_x) / max(scale, 1e-9)
            y1 = (center_y - 0.5 * height - pad_y) / max(scale, 1e-9)
            x2 = (center_x + 0.5 * width - pad_x) / max(scale, 1e-9)
            y2 = (center_y + 0.5 * height - pad_y) / max(scale, 1e-9)
            x1 = float(np.clip(x1, 0, image_w - 1))
            y1 = float(np.clip(y1, 0, image_h - 1))
            x2 = float(np.clip(x2, x1 + 1, image_w))
            y2 = float(np.clip(y2, y1 + 1, image_h))
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(confidences[index]))

        if not boxes_xywh:
            return result
        kept = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            float(self.conf_threshold),
            float(nms_threshold),
        )
        kept_indexes = np.asarray(kept).reshape(-1).astype(int).tolist() if len(kept) else []
        detections = []
        for index in kept_indexes:
            x, y, width, height = boxes_xywh[index]
            detections.append(
                {
                    "bbox_xyxy": [
                        int(round(x)),
                        int(round(y)),
                        int(round(x + width)),
                        int(round(y + height)),
                    ],
                    "bbox_xywh": [
                        int(round(x)),
                        int(round(y)),
                        int(round(width)),
                        int(round(height)),
                    ],
                    "confidence": scores[index],
                    "class_name": "bottle",
                }
            )
        if not detections:
            return result
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        result.update(detections[0])
        result["detections"] = detections
        result["detection_count"] = len(detections)
        result["yolo_detected"] = True
        result["bottle_detected"] = True
        result["detection_source"] = "yolo_opencv_dnn_onnx"
        return result

    def _detect_darknet(self, rgb, result, input_size=416, nms_threshold=0.45):
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            result["yolo_available"] = False
            result["detection_source"] = "opencv_depth_fallback"
            result["yolo_unavailable_reason"] = "YOLO Darknet inference requires an RGB HxWx3 image."
            return result

        image_h, image_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            swapRB=False,
            crop=False,
        )
        started = perf_counter()
        try:
            self.model.setInput(blob)
            outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())
        except Exception as exc:
            result["yolo_available"] = False
            result["detection_source"] = "opencv_depth_fallback"
            result["yolo_unavailable_reason"] = f"YOLO Darknet inference failed: {exc}"
            return result
        result["yolo_inference_ms"] = 1000.0 * (perf_counter() - started)

        boxes_xywh = []
        scores = []
        for output in outputs or []:
            for prediction in np.asarray(output):
                if prediction.size < 85:
                    continue
                class_scores = prediction[5:]
                class_id = int(np.argmax(class_scores))
                confidence = float(prediction[4]) * float(class_scores[class_id])
                if class_id != 39 or confidence < self.conf_threshold:
                    continue
                center_x = float(prediction[0]) * image_w
                center_y = float(prediction[1]) * image_h
                width = float(prediction[2]) * image_w
                height = float(prediction[3]) * image_h
                boxes_xywh.append(
                    [
                        center_x - 0.5 * width,
                        center_y - 0.5 * height,
                        width,
                        height,
                    ]
                )
                scores.append(confidence)

        if not boxes_xywh:
            return result
        kept = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            float(self.conf_threshold),
            float(nms_threshold),
        )
        kept_indexes = np.asarray(kept).reshape(-1).astype(int).tolist() if len(kept) else []
        detections = []
        for index in kept_indexes:
            x, y, width, height = boxes_xywh[index]
            x1 = int(round(np.clip(x, 0, image_w - 1)))
            y1 = int(round(np.clip(y, 0, image_h - 1)))
            x2 = int(round(np.clip(x + width, x1 + 1, image_w)))
            y2 = int(round(np.clip(y + height, y1 + 1, image_h)))
            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": scores[index],
                    "class_name": "bottle",
                }
            )
        if not detections:
            return result
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        result.update(detections[0])
        result["detections"] = detections
        result["detection_count"] = len(detections)
        result["yolo_detected"] = True
        result["bottle_detected"] = True
        result["detection_source"] = "yolo_opencv_dnn_darknet"
        return result


def expand_bbox_xyxy(bbox_xyxy, image_shape, expand_ratio=0.15):
    if not bbox_xyxy:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = bw * float(expand_ratio)
    pad_y = bh * float(expand_ratio)
    ex1 = int(max(0, round(x1 - pad_x)))
    ey1 = int(max(0, round(y1 - pad_y)))
    ex2 = int(min(w, round(x2 + pad_x)))
    ey2 = int(min(h, round(y2 + pad_y)))
    if ex2 - ex1 < 20 or ey2 - ey1 < 20:
        return None
    return ex1, ey1, ex2, ey2
