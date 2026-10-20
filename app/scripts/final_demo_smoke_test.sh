#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
mkdir -p output

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() { echo "[PASS] $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn() { echo "[WARN] $*"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { echo "[FAIL] $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "Final demo smoke test"
echo "Project: $PROJECT_DIR"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

if ./check_demo_system.sh; then
    pass "system health check completed"
else
    warn "system health check reported failures or warnings; continuing smoke test"
fi

if curl -fsS --max-time 2 http://127.0.0.1:8080/api/status >/tmp/final_demo_status_initial.json 2>/dev/null; then
    pass "dashboard already running"
else
    echo "Starting final demo for smoke test..."
    ROS_DOMAIN_ID=67 ./run_final_grasp_demo.sh --with-davis >output/final_demo_smoke_test.log 2>&1 &
    echo $! >output/final_demo_smoke_test.pid
fi

ready=0
for _ in $(seq 1 45); do
    if curl -fsS --max-time 2 http://127.0.0.1:8080/api/status >/tmp/final_demo_status.json 2>/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -eq 1 ]; then
    pass "dashboard /api/status became reachable"
else
    fail "dashboard /api/status did not become reachable"
fi

fresh=0
for _ in $(seq 1 45); do
    if python3 - <<'PY' >/tmp/final_demo_fresh_check.txt 2>&1
import json, time
from pathlib import Path
base = Path("output")
status = base / "latest_realtime_status.json"
overlay = base / "latest_overlay_live.jpg"
if not status.exists() or not overlay.exists():
    raise SystemExit("missing status or overlay")
now = time.time()
status_age = now - status.stat().st_mtime
overlay_age = now - overlay.stat().st_mtime
data = json.loads(status.read_text())
frame_age = ((data.get("processed_overlay_timing") or {}).get("frame_age_ms")
             or (data.get("realtime_pipeline_audit") or {}).get("pending_frame_age_ms"))
print(json.dumps({
    "status_age_sec": round(status_age, 3),
    "overlay_age_sec": round(overlay_age, 3),
    "frame_age_ms": frame_age,
    "writer_session_id": data.get("writer_session_id"),
    "monotonic_frame_counter": data.get("monotonic_frame_counter"),
}))
if status_age > 2.0 or overlay_age > 2.0:
    raise SystemExit("stale live files")
if frame_age is not None and float(frame_age) > 100.0:
    raise SystemExit("frame age exceeds 100 ms")
PY
    then
        fresh=1
        break
    fi
    sleep 1
done
if [ "$fresh" -eq 1 ]; then
    pass "fresh overlay/status files with frame age under 100 ms"
else
    fail "fresh overlay/status check failed: $(cat /tmp/final_demo_fresh_check.txt 2>/dev/null)"
fi

writer_count="$(pgrep -fc 'real_time_live.py --backend-only' || true)"
if [ "$writer_count" -eq 1 ]; then
    pass "exactly one RealSense backend writer"
else
    fail "expected one RealSense backend writer, found $writer_count"
fi

fast_path=0
for _ in $(seq 1 45); do
    if python3 - <<'PY' >/tmp/final_demo_fast_path_check.txt 2>&1
import json
from pathlib import Path
data = json.loads(Path("output/latest_realtime_status.json").read_text())
target_status = str(data.get("target_status") or (data.get("target_lock") or {}).get("target_status") or "")
overlay_mode = str(data.get("overlay_mode") or (data.get("processed_overlay_timing") or {}).get("mode") or "")
identity_current = data.get("identity_measurement_current") is True
result = {
    "target_status": target_status,
    "overlay_mode": overlay_mode,
    "identity_measurement_current": identity_current,
    "tracking_good": data.get("tracking_good"),
}
print(json.dumps(result))
if target_status not in {"LOCKED", "TRACKING"}:
    raise SystemExit("target is not locked/tracking")
if overlay_mode != "fast_tracking_local_locked_patch":
    raise SystemExit("locked fast-tracking overlay mode is not active")
if not identity_current:
    raise SystemExit("target identity is not current")
PY
    then
        fast_path=1
        break
    fi
    sleep 1
done
if [ "$fast_path" -eq 1 ]; then
    pass "locked fast-tracking phase is active: $(cat /tmp/final_demo_fast_path_check.txt)"
else
    fail "target acquisition did not reach the locked fast-tracking phase: $(cat /tmp/final_demo_fast_path_check.txt 2>/dev/null)"
fi

if [ "$fast_path" -eq 1 ]; then
python3 - <<'PY' >/tmp/final_demo_fps_check.txt 2>&1
import json, time
from pathlib import Path
status_path = Path("output/latest_realtime_status.json")
overlay_path = Path("output/latest_overlay_live.jpg")
samples = []
overlay_times = []
last_overlay = None
deadline = time.time() + 5.0
while time.time() < deadline:
    try:
        data = json.loads(status_path.read_text())
        counter = data.get("monotonic_frame_counter") or data.get("frame_id")
        if counter is not None:
            samples.append((time.time(), int(counter)))
        mtime = overlay_path.stat().st_mtime_ns
        if mtime != last_overlay:
            overlay_times.append(time.time())
            last_overlay = mtime
    except Exception:
        pass
    time.sleep(0.01)
def fps_from_counter(values):
    if len(values) < 2:
        return 0.0
    dt = values[-1][0] - values[0][0]
    df = values[-1][1] - values[0][1]
    return 0.0 if dt <= 0 else df / dt
def fps_from_times(values):
    if len(values) < 2:
        return 0.0
    return (len(values) - 1) / max(values[-1] - values[0], 1e-6)
print(json.dumps({
    "status_fps": round(fps_from_counter(samples), 2),
    "overlay_file_fps": round(fps_from_times(overlay_times), 2),
    "status_samples": len(samples),
    "overlay_updates": len(overlay_times),
}))
PY
fps_json="$(cat /tmp/final_demo_fps_check.txt)"
overlay_fps="$(python3 - <<'PY'
import json
data = json.load(open("/tmp/final_demo_fps_check.txt"))
print(data.get("overlay_file_fps", 0.0))
PY
)"
if python3 - <<PY
fps = float("$overlay_fps")
raise SystemExit(0 if fps >= 20.0 else 1)
PY
then
    pass "overlay FPS target met: $fps_json"
else
    fail "overlay FPS below 20 target: $fps_json"
fi
else
    warn "locked fast-path FPS check was not applicable because target acquisition failed"
fi

if python3 - <<'PY' >/tmp/final_demo_matrix_check.txt 2>&1
import json
from pathlib import Path
path = Path("output/realtime_wrench_matrix_status.json")
if not path.exists():
    print("matrix JSON missing; allowed before lock, but quality panel must show waiting")
    raise SystemExit(0)
data = json.loads(path.read_text())
status = data.get("quality_matrix_status", "idle")
message = data.get("quality_matrix_message") or data.get("quality_matrix_error") or ""
reasons = data.get("quality_matrix_failure_reasons") or data.get("matrix_stale_reasons") or []
print(json.dumps({"status": status, "message": message, "reasons": reasons}))
if status in {"failed", "idle", "STALE_NEEDS_RECOMPUTE"} and not (message or reasons):
    raise SystemExit("matrix has no clear status reason")
PY
then
    pass "matrix JSON is present or has a clear reason"
else
    fail "matrix JSON check failed: $(cat /tmp/final_demo_matrix_check.txt 2>/dev/null)"
fi

echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
