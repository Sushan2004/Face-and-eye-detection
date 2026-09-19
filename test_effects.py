"""Small, camera-free tests for gesture classification and spell rendering."""
import unittest

import numpy as np

from effects import SpellEffects
from main import classify_gesture


class SpellEffectsTests(unittest.TestCase):
    def test_gesture_classification(self):
        self.assertEqual(classify_gesture([False, True, False, False, False], False), "POINT")
        self.assertEqual(classify_gesture([True] * 5, False), "OPEN PALM")
        self.assertEqual(classify_gesture([False] * 5, True), "PINCH")

    def test_effects_render_without_camera(self):
        effects = SpellEffects()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for step in range(12):
            effects.update("PINCH", (100 + step * 8, 160), (150, 190), 1 / 30)
        effects.update("NONE", (200, 160), (150, 190), 1 / 30)
        effects.render(frame, (150, 190))
        self.assertGreater(frame.sum(), 0)
        self.assertGreater(len(effects.particles), 0)
        self.assertGreaterEqual(len(effects.strokes), 1)


if __name__ == "__main__":
    unittest.main()
