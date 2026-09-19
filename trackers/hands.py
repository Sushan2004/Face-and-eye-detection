"""Tasks hand tracking with image and metric model-estimated world landmarks."""
from .common import TaskTracker, image, xyz

class HandTracker(TaskTracker):
    def __init__(self, model, config):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.task = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=config.get("confidence", 0.65),
            min_tracking_confidence=config.get("confidence", 0.65)))

    @staticmethod
    def decode(result):
        return [dict(points=xyz(p), world=xyz(result.hand_world_landmarks[i]),
                     side=result.handedness[i][0].category_name)
                for i, p in enumerate(result.hand_landmarks)]

    def process(self, frame, timestamp):
        return self.decode(self.task.detect_for_video(image(frame), timestamp))
