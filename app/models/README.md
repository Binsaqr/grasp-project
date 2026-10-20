# Generic bottle proposal model

The final single-bottle release does not download or require a YOLO model.
RGB-D detection is authoritative and `USE_YOLO_BOTTLE_TRIGGER=0` by default.
YOLOv4-tiny remains an optional pre-lock fallback for experiments: place a
compatible configuration and weights in `models/`, set `YOLO_MODEL_PATH`, and
explicitly set `USE_YOLO_BOTTLE_TRIGGER=1`. The runtime then uses the existing
OpenCV DNN dependency; it does not require PyTorch, Ultralytics, ONNX Runtime,
or a replacement OpenCV package.

- Purpose: pre-lock semantic proposals for COCO class `bottle` (class 39).
- Model: YOLOv4-tiny generic MS COCO weights.
- Runtime: existing OpenCV DNN; no new Python runtime dependency.
- Provenance and license: no model weights or Darknet configuration are tracked
  in this checkout. The exact experimental asset URL, version, checksum, license,
  and redistribution obligations are TBD. The upstream
  [Darknet source license](https://github.com/AlexeyAB/darknet/blob/master/LICENSE)
  does not by itself document the provenance or terms of an arbitrary downloaded
  weight file or its training data. Record those separately in
  [THIRD_PARTY.md](../../THIRD_PARTY.md) before enabling or sharing model assets.
- Fusion: YOLO boxes are spatially matched to the existing full-frame RGB-D
  candidates. Unmatched YOLO boxes get one RGB-D ROI recovery attempt, while
  unmatched RGB-D candidates are retained instead of being erased by a
  semantic-model miss.
- Safety: YOLO supplies semantic proposals only. Existing aligned RGB-D pose
  validation, availability, occlusion, target-lock, and grasp-safety gates
  remain authoritative.
- Validation status: no physical grasp validation or reproducible model benchmark is supplied with this repository.
