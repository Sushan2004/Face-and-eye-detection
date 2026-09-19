"""Coarse screen-relative gaze, neutral-calibrated; not a gaze-point regressor."""
import numpy as np

class GazeTracker:
    def __init__(self):
        self.neutral = None
        self.last_features = None

    def calibrate(self):
        if self.last_features is None:
            return False
        self.neutral = self.last_features.copy()
        return True

    def process(self, face, irises):
        if len(irises) != 2:
            self.last_features = None
            return None
        ratios = np.mean([eye["ratio"] for eye in irises], axis=0)
        pitch, yaw, _ = face["angles"]
        self.last_features = np.r_[ratios, pitch, yaw]
        if self.neutral is None:
            return {"direction": "Look center; press K", "calibrated": False}
        diff = self.last_features - self.neutral
        # Heuristic sign convention for the mirrored preview; validate per user.
        horizontal = diff[0] + diff[3] / 90
        vertical = diff[1] + diff[2] / 90
        horizontal_label = "right" if horizontal > .10 else "left" if horizontal < -.10 else ""
        vertical_label = "down" if vertical > .15 else "up" if vertical < -.15 else ""
        return {"direction": " ".join(filter(None, (vertical_label,horizontal_label))) or "center",
                "calibrated": True}
