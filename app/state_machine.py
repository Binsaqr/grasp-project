from dataclasses import dataclass, field
from datetime import datetime, timezone


STATES = [
    "SEARCH",
    "FIND_FULL_BOTTLE",
    "SEARCH_BOTTLE",
    "FULL_BOTTLE_VIEW_CHECK",
    "SELECT_BEST_GRASP",
    "TARGET_LOCKED",
    "APPROACH_TRACKING",
    "CLOSE_UP_VERIFY",
    "CONTACT_MODELING",
    "PRE_GRASP_EVALUATION",
    "APPROACH_EVALUATION",
    "READY_TO_CLOSE",
    "TARGET_LOST",
    "POST_GRASP_EVALUATION",
    "READY_TO_LIFT",
    "REGRIP",
    "DO_NOT_LIFT",
    "LIFT_MONITOR",
    "FAIL_SAFE",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GraspStateMachine:
    state: str = "SEARCH"
    history: list = field(default_factory=list)

    def record(self, stage_result):
        next_state = self.next_state(stage_result)
        event = {
            "timestamp": utc_now_iso(),
            "from_state": self.state,
            "stage": stage_result.get("stage"),
            "stage_decision": stage_result.get("stage_decision"),
            "recommended_action": stage_result.get("recommended_action"),
            "safety_score": stage_result.get("safety_score"),
            "to_state": next_state,
            "reasons": stage_result.get("reasons", []),
            "warnings": stage_result.get("warnings", []),
        }
        self.history.append(event)
        self.state = next_state
        return event

    def next_state(self, stage_result):
        stage = stage_result.get("stage")
        decision = stage_result.get("stage_decision")

        if decision in {"SEARCH"}:
            return "FIND_FULL_BOTTLE"
        if decision == "TARGET_LOST":
            return "TARGET_LOST"
        if decision in {"DO_NOT_ATTEMPT", "DO_NOT_CLOSE"}:
            return "FAIL_SAFE"

        if stage == "FULL_BOTTLE_VIEW_CHECK":
            if decision == "VIEW_OK":
                return "CONTACT_MODELING"
            if decision == "ADJUST_VIEW":
                return "FULL_BOTTLE_VIEW_CHECK"
            return "FIND_FULL_BOTTLE"

        if stage == "CONTACT_MODELING":
            if decision == "SELECT_BEST_GRASP":
                return "SELECT_BEST_GRASP"
            if decision == "TARGET_LOCKED":
                return "TARGET_LOCKED"
            if decision == "CONTACT_MODEL_READY":
                return "PRE_GRASP_EVALUATION"
            return "FULL_BOTTLE_VIEW_CHECK"

        if stage == "PRE_GRASP_EVALUATION":
            if decision == "ADJUST_VIEW":
                return "FULL_BOTTLE_VIEW_CHECK"
            if decision == "CONTACT_MODEL_DISABLED":
                return "CONTACT_MODELING"
            if decision == "READY_TO_APPROACH":
                return "APPROACH_EVALUATION"
            return "PRE_GRASP_EVALUATION"

        if stage == "APPROACH_EVALUATION":
            if decision == "CONTINUE_APPROACH":
                return "APPROACH_TRACKING"
            if decision == "VERIFY_CLOSE_UP":
                return "CLOSE_UP_VERIFY"
            if decision == "READY_TO_CLOSE":
                return "READY_TO_CLOSE"
            if decision == "BACK_UP":
                return "PRE_GRASP_EVALUATION"
            return "APPROACH_EVALUATION"

        if stage == "POST_GRASP_EVALUATION":
            if decision == "READY_TO_LIFT":
                return "READY_TO_LIFT"
            if decision == "REGRIP":
                return "REGRIP"
            if decision == "DO_NOT_LIFT":
                return "DO_NOT_LIFT"
            return "POST_GRASP_EVALUATION"

        if stage == "REGRIP":
            return "APPROACH_EVALUATION"
        if stage == "DO_NOT_LIFT":
            return "DO_NOT_LIFT"

        if stage == "READY_TO_LIFT":
            return "LIFT_MONITOR"
        if stage == "LIFT_MONITOR":
            return "LIFT_MONITOR"

        return self.state if self.state in STATES else "FAIL_SAFE"

    def status(self, latest_result=None):
        latest_result = latest_result or {}
        return {
            "current_state": self.state,
            "current_stage": latest_result.get("stage", self.state),
            "stage_decision": latest_result.get("stage_decision", "UNKNOWN"),
            "recommended_action": latest_result.get("recommended_action", "UNKNOWN"),
            "safety_score": latest_result.get("safety_score", 0.0),
            "reasons": latest_result.get("reasons", []),
            "warnings": latest_result.get("warnings", []),
            "history_length": len(self.history),
            "updated_at": utc_now_iso(),
        }
