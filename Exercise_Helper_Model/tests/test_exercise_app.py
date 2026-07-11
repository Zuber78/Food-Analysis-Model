import sys
import unittest
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
import exercise_app


class ExerciseAppTests(unittest.TestCase):
    def test_parse_args_demo_flag(self):
        args = exercise_app.parse_args(["--demo"])
        self.assertTrue(args.demo)

    def test_open_video_capture_returns_none_for_invalid_device(self):
        cap = exercise_app.open_video_capture(camera_index=999, backends=[cv2.CAP_ANY])
        if cap is not None:
            cap.release()
        self.assertIsNone(cap)

    def test_countdown_state_finishes_at_zero(self):
        state = exercise_app.get_countdown_state(elapsed=5.2, countdown=5)
        self.assertTrue(state["finished"])
        self.assertEqual(state["text"], "GO!")

    def test_build_voice_instruction_is_slow_and_clear(self):
        instruction = exercise_app.build_voice_instruction(
            "Jumping Jacks",
            "warning",
            [("Arms poore upar le jao", (0, 220, 255))],
        )
        self.assertIn("Jumping Jacks", instruction)
        self.assertIn("slowly", instruction.lower())
        self.assertIn("correct", instruction.lower())


if __name__ == "__main__":
    unittest.main()
