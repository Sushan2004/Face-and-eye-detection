"""Iris geometry in image pixels. This does not segment or measure the pupil."""
import numpy as np

class IrisTracker:
    @staticmethod
    def process(face, shape):
        points = face["points"]
        if len(points) < 478:
            return []
        result = []
        for center, ring, corners, lids in (
            (468, (469,470,471,472), (33,133), (159,145)),
            (473, (474,475,476,477), (362,263), (386,374))):
            p = points[:, :2] * [shape[1], shape[0]]
            axis = p[corners[1]] - p[corners[0]]
            width = np.linalg.norm(axis)
            if width < 2:
                continue
            vertical = p[lids[1]] - p[lids[0]]
            height = np.linalg.norm(vertical)
            if height / width < 0.08:  # suppress closed-eye estimates
                continue
            horizontal_ratio = np.dot(p[center]-p[corners[0]], axis) / width**2
            vertical_ratio = np.dot(p[center]-p[lids[0]], vertical) / max(height**2, 1)
            radius = float(np.mean(np.linalg.norm(p[list(ring)] - p[center], axis=1)))
            result.append(dict(center=p[center], radius=radius,
                               ratio=(float(horizontal_ratio),float(vertical_ratio))))
        return result
