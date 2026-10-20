"""Small DAVIS motion classifier smoke tests without a physical DAVIS camera."""

from grasp_project.davis.motion_classifier import simplified_motion_class


def test_simplified_motion_class_labels():
    assert simplified_motion_class("STABLE") == "STABLE"
    assert simplified_motion_class("INCIPIENT_SLIP") == "MICRO SLIP"
    assert simplified_motion_class("GROSS_SLIP") == "GROSS SLIP"
    assert simplified_motion_class("SPIN_OR_ROTATION") == "ROTATION"
    assert simplified_motion_class("SHAKE_OR_VIBRATION") == "VIBRATION"
    assert simplified_motion_class("SENSOR_UNAVAILABLE") == "UNSAFE"

