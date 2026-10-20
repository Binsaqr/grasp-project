import unittest

from real_time_live import attach_target_lock_state


class TargetLockStateLogicTest(unittest.TestCase):
    def test_pre_lock_requires_full_bottle(self):
        report = {
            "full_bottle_visible": False,
            "target_lock": {"target_status": "SEARCHING", "lock_state": "SEARCHING"},
        }
        attach_target_lock_state(report)
        self.assertFalse(report["target_locked"])
        self.assertTrue(report["full_bottle_required"])
        self.assertEqual(report["state_transition_reason"], "pre_lock_full_bottle_required")

    def test_post_lock_cropped_full_bottle_tracks_patch(self):
        report = {
            "state": "APPROACH_TRACKING",
            "full_bottle_visible": False,
            "target_lock": {
                "target_status": "TRACKING",
                "lock_state": "LOCKED",
                "locked_patch_visible": True,
                "bad_tracking_frames": 0,
            },
        }
        attach_target_lock_state(report)
        self.assertTrue(report["target_locked"])
        self.assertFalse(report["full_bottle_required"])
        self.assertTrue(report["locked_patch_visible"])
        self.assertEqual(report["state_transition_reason"], "post_lock_tracking_locked_patch")

    def test_post_lock_target_lost_only_after_bad_frame_threshold(self):
        report = {
            "state": "TARGET_LOST",
            "full_bottle_visible": False,
            "target_lock": {
                "target_status": "LOST",
                "lock_state": "LOST",
                "locked_patch_visible": False,
                "bad_tracking_frames": 5,
                "max_bad_tracking_frames": 5,
                "target_loss_reason": "Locked patch/bottle pose unavailable for too many frames.",
            },
        }
        attach_target_lock_state(report)
        self.assertFalse(report["target_locked"])
        self.assertTrue(report["full_bottle_required"])
        self.assertEqual(report["target_loss_reason"], "Locked patch/bottle pose unavailable for too many frames.")


if __name__ == "__main__":
    unittest.main()
