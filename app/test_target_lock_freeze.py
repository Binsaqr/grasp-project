import unittest

from real_time_live import validate_freeze_trace_sequence


def make_trace(frame_id, fraction=0.5, bottle_y=100.0, clamp_y=120.0):
    return {
        "frame_id": frame_id,
        "target_status": "TRACKING",
        "locked_body_fraction": fraction,
        "bottle_center_raw": [200.0, bottle_y],
        "clamp_patch_left_center": [180.0, clamp_y],
        "clamp_patch_right_center": [220.0, clamp_y],
    }


class TargetLockFreezeTest(unittest.TestCase):
    def test_locked_fraction_stays_constant_for_30_frames(self):
        entries = [make_trace(i, bottle_y=100.0 + i * 0.1, clamp_y=120.0 + i * 0.1) for i in range(30)]
        self.assertEqual(validate_freeze_trace_sequence(entries), [])

    def test_fraction_change_fails_freeze_check(self):
        entries = [make_trace(i) for i in range(30)]
        entries[17]["locked_body_fraction"] = 0.62
        warnings = validate_freeze_trace_sequence(entries)
        self.assertTrue(any("locked_body_fraction changed" in warning for warning in warnings))

    def test_clamp_jump_warns_when_bottle_is_stable(self):
        entries = [make_trace(i) for i in range(30)]
        entries[12]["clamp_patch_left_center"] = [180.0, 150.0]
        warnings = validate_freeze_trace_sequence(entries)
        self.assertTrue(any("clamp patch left moved" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
