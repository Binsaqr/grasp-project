from grasp_project.control.slip_force_controller import SlipForceController, simulate_force_policy


VALID = {"object_motion_valid": True, "event_count": 100, "measurement_status": "ESTIMATED_OBJECT_LOCAL_UNCALIBRATED"}
STANDING = {"orientation": "STANDING", "tilt_angle_deg": 2.0}
SIDE_PLAN = {"strategy": "SIDE_ANTIPODAL", "allow_rotation": False}


def test_incipient_pattern_preempts_and_gross_slip_escalates():
    patterns = [
        {**VALID, "incipient_slip_precursor_score": 55.0, "slip_cue": 5.0},
        {**VALID, "incipient_slip_precursor_score": 70.0, "slip_cue": 80.0},
    ]
    commands = simulate_force_policy(patterns, STANDING, SIDE_PLAN)
    assert commands[0].grip_force_delta > 0
    assert "PREEMPTIVE" in commands[0].reason
    assert commands[1].grip_force_delta > commands[0].grip_force_delta
    assert "ESTIMATED_GROSS_SLIP_CUE" in commands[1].reason
    assert "CONFIRMED_SLIP" not in commands[1].reason
    assert commands[1].physically_validated is False


def test_unwanted_rotation_emits_counter_torque():
    command = SlipForceController().update(
        {**VALID, "rotation_cue": 70.0, "tangential_circulation_px": 2.0},
        STANDING,
        SIDE_PLAN,
    )
    assert command.torque_delta < 0
    assert "SUPPRESS_UNWANTED_ROTATION" in command.reason
    assert not command.physically_validated


def test_lying_policy_regulates_rotation_then_locks_when_upright():
    pattern = {**VALID, "rotation_cue": 60.0, "tangential_circulation_px": 2.0}
    plan = {"strategy": "TOP_GRASP_ROTATE_UPRIGHT", "allow_rotation": True}
    commands = simulate_force_policy(
        [pattern, pattern],
        [{"orientation": "LYING", "tilt_angle_deg": 80.0}, {"orientation": "STANDING", "tilt_angle_deg": 2.0}],
        plan,
    )
    assert commands[0].torque_delta > 0
    assert "REGULATE_ALLOWED_ROTATION" in commands[0].reason
    assert commands[1].torque_delta < 0
    assert "UPRIGHT_ROTATION_LOCK" in commands[1].reason


def test_lighting_change_never_emits_force_action():
    command = SlipForceController().update(
        {**VALID, "incipient_slip_precursor_score": 100.0, "slip_cue": 100.0},
        STANDING,
        SIDE_PLAN,
        {"raw_motion_class": "LIGHTING_CHANGE", "motion_estimation_confidence": 1.0},
    )
    assert command.grip_force_delta == 0
    assert command.torque_delta == 0
    assert command.reason == "NO_ACTION_GLOBAL_DISTURBANCE"


def test_invalid_object_motion_never_emits_force_action():
    command = SlipForceController().update(
        {**VALID, "object_motion_valid": False, "slip_cue": 100.0},
        STANDING,
        SIDE_PLAN,
    )
    assert command.grip_force_delta == command.torque_delta == 0


def test_pre_close_motion_never_emits_force_action():
    command = SlipForceController().update(
        {**VALID, "incipient_slip_precursor_score": 100.0, "slip_cue": 100.0},
        STANDING,
        SIDE_PLAN,
        {
            "current_stage": "PRE_CLOSE",
            "davis_motion_usable_for_grasp": False,
            "motion_estimation_confidence": 1.0,
        },
    )
    assert command.grip_force_delta == command.torque_delta == 0
    assert command.reason == "NO_ACTION_NOT_POST_CLOSE_USABLE"
