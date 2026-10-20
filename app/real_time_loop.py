import json
import runpy
import time
from pathlib import Path

from scripts.bottle_grasp_planner import run_bottle_pregrasp
from stage_evaluators import (
    evaluate_approach_stage,
    evaluate_contact_modeling_stage,
    evaluate_full_bottle_view_stage,
    evaluate_lift_monitor_stage,
    evaluate_postgrasp_stage,
    evaluate_pregrasp_stage,
)
from state_machine import GraspStateMachine


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "pregrasp_safety_report.json"
STATUS_PATH = OUTPUT_DIR / "realtime_status.json"
HISTORY_PATH = OUTPUT_DIR / "stage_history.json"


def capture_frame():
    runpy.run_path(str(ROOT / "scripts" / "capture_rgb_depth.py"), run_name="__main__")


def load_report():
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


def write_json(path, value):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=4)


def evaluate_for_state(state, report, previous_report=None):
    if state in {"SEARCH", "SEARCH_BOTTLE", "FIND_FULL_BOTTLE", "TARGET_LOST", "FULL_BOTTLE_VIEW_CHECK"}:
        return evaluate_full_bottle_view_stage(report)
    if state in {"CONTACT_MODELING", "SELECT_BEST_GRASP", "TARGET_LOCKED"}:
        return evaluate_contact_modeling_stage(report)
    if state == "PRE_GRASP_EVALUATION":
        return evaluate_pregrasp_stage(report)
    if state in {"APPROACH_EVALUATION", "APPROACH_TRACKING", "CLOSE_UP_VERIFY", "READY_TO_CLOSE", "REGRIP"}:
        return evaluate_approach_stage(report)
    if state in {"POST_GRASP_EVALUATION", "READY_TO_LIFT"}:
        return evaluate_postgrasp_stage(report, pre_close_report=previous_report)
    if state == "LIFT_MONITOR":
        return evaluate_lift_monitor_stage(report)
    return {
        "stage": "FAIL_SAFE",
        "stage_decision": "DO_NOT_ATTEMPT",
        "recommended_action": "STOP",
        "safety_score": 0.0,
        "reasons": ["State machine is in FAIL_SAFE."],
        "warnings": [],
        "metrics": {},
        "source_decision": report.get("decision"),
        "source_recommended_action": "STOP",
    }


def run_realtime_loop(iterations=1, interval_s=0.0, capture_each_frame=False):
    machine = GraspStateMachine(state="FULL_BOTTLE_VIEW_CHECK")
    previous_report = None
    latest_report = None
    latest_result = None

    for index in range(int(iterations)):
        print(f"Realtime staged evaluation frame {index + 1}/{iterations}")
        if capture_each_frame:
            capture_frame()
        try:
            run_bottle_pregrasp()
        except SystemExit:
            pass

        latest_report = load_report()
        latest_result = evaluate_for_state(machine.state, latest_report, previous_report=previous_report)
        machine.record(latest_result)

        status = machine.status(latest_result)
        status["frame_index"] = index
        status["mode"] = "bottle_realtime_staged"
        status["note"] = "No robot commands are sent; recommendations are image/depth-based only."
        write_json(STATUS_PATH, status)
        write_json(HISTORY_PATH, machine.history)

        print(f"Current stage: {status['current_stage']}")
        print(f"Stage decision: {status['stage_decision']}")
        print(f"Recommended action: {status['recommended_action']}")
        print(f"Next state: {status['current_state']}")

        previous_report = latest_report
        if index < int(iterations) - 1 and interval_s > 0:
            time.sleep(float(interval_s))

    return machine.status(latest_result), machine.history, latest_report


if __name__ == "__main__":
    run_realtime_loop()
