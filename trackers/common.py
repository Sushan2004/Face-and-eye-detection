"""Shared geometry and MediaPipe Tasks utilities."""
import math
import numpy as np

def xyz(points):
    return np.array([(p.x, p.y, p.z) for p in points], dtype=float)

def pixels(points, shape):
    return (points[:, :2] * [shape[1], shape[0]]).astype(int)

def box(points):
    low, high = np.min(points[:, :2], axis=0), np.max(points[:, :2], axis=0)
    return np.clip(np.r_[low, high], 0, 1)

def crop(frame, bounds, padding=0.15):
    x1, y1, x2, y2 = bounds
    w, h = x2-x1, y2-y1
    a = max(0, int((x1-padding*w)*frame.shape[1]))
    b = max(0, int((y1-padding*h)*frame.shape[0]))
    c = min(frame.shape[1], int((x2+padding*w)*frame.shape[1]))
    d = min(frame.shape[0], int((y2+padding*h)*frame.shape[0]))
    return frame[b:d, a:c]

class TaskTracker:
    def close(self):
        self.task.close()

def image(frame):
    import cv2
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def euler(matrix):
    # Remove scale from the canonical-face transform before Euler decomposition.
    u, _, vt = np.linalg.svd(np.asarray(matrix)[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], sy)
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(math.degrees(v) for v in (pitch, yaw, roll))
