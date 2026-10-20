import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
LOG_PATH = OUTPUT_DIR / "grasp_attempt_log.jsonl"
VALID_RESULTS = {"success", "slip", "spin", "dropped", "failed_to_grasp", "unknown"}


def utc_now():
    return datetime.utcnow().isoformat() + "Z"


def append_result(attempt_id, result, notes):
    OUTPUT_DIR.mkdir(exist_ok=True)
    entry = {
        "event_type": "final_result",
        "attempt_id": attempt_id,
        "timestamp": utc_now(),
        "final_result": result,
        "notes": notes or "",
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Append a final result to the grasp attempt log.")
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--result", choices=sorted(VALID_RESULTS), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    entry = append_result(args.attempt_id, args.result, args.notes)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
