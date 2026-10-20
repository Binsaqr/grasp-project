import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from davis_grasp_planner import analyze_aps_frame
from event_postgrasp_monitor import EventPostGraspMonitor


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "davis_yolo_dataset" / "raw_images"


def latest_aps_from_frames(frames):
    arr = np.asarray(frames) if frames is not None else np.empty((0,), dtype=np.uint8)
    if not arr.size:
        return None
    return arr[-1] if arr.ndim >= 3 else arr


def save_frame(frame, index, output_dir=RAW_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    _gray, display, info, _debug = analyze_aps_frame(frame)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"davis_aps_{timestamp}_{index:04d}"
    if frame is not None:
        np.save(output_dir / f"{stem}_raw.npy", np.asarray(frame))
    if display is not None:
        cv2.imwrite(str(output_dir / f"{stem}.png"), display)
    return output_dir / f"{stem}.png", info


def capture(interval=0.5, count=300, manual=False):
    monitor = EventPostGraspMonitor(enabled=True)
    report = monitor.start()
    if not monitor.active:
        print(f"DAVIS unavailable: {report.get('davis_unavailable_reason')}")
        return 2
    saved = 0
    next_capture = time.monotonic()
    print("Capturing DAVIS APS frames. Press 'c' to capture in manual mode, 'q' to quit.")
    try:
        while saved < count:
            _events, frames = monitor.poll_event_and_frame_arrays()
            frame = latest_aps_from_frames(frames)
            if frame is None:
                time.sleep(0.02)
                continue
            _gray, display, _info, _debug = analyze_aps_frame(frame)
            if display is not None:
                cv2.imshow("DAVIS APS capture", display)
            key = cv2.waitKey(1) & 0xFF
            do_capture = False
            now = time.monotonic()
            if manual:
                do_capture = key == ord("c")
            elif now >= next_capture:
                do_capture = True
                next_capture = now + max(float(interval), 0.05)
            if key == ord("q"):
                break
            if do_capture:
                path, info = save_frame(frame, saved)
                saved += 1
                print(f"saved {saved}/{count}: {path} aps_mode={info.get('aps_mode')}", flush=True)
            time.sleep(0.005)
    finally:
        monitor.stop()
        cv2.destroyAllWindows()
    print(f"Done. Saved {saved} frames to {RAW_DIR}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture DAVIS APS frames for YOLO bottle dataset labeling.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between timed captures.")
    parser.add_argument("--count", type=int, default=300, help="Number of frames to capture.")
    parser.add_argument("--manual", action="store_true", help="Capture only when pressing 'c'.")
    args = parser.parse_args()
    return capture(interval=args.interval, count=args.count, manual=args.manual)


if __name__ == "__main__":
    raise SystemExit(main())
