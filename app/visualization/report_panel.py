"""Readable report data helpers for the PyQt dashboard."""

import json


def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def load_report_data(output_dir="output"):
    return {
        "contacts": load_json(f"{output_dir}/contact_points.json", {}),
        "epsilon": load_json(f"{output_dir}/epsilon_result.json", {}),
    }

