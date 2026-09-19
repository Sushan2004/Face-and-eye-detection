"""Full-body pose with normalized and world coordinates."""
from .common import TaskTracker, image, xyz, box

class PoseTracker(TaskTracker):
    def __init__(self, model, config):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.task = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            output_segmentation_masks=False))

    @staticmethod
    def decode(result):
        return [dict(points=xyz(p), world=xyz(result.pose_world_landmarks[i]),
                     visibility=[m.visibility for m in p], box=box(xyz(p)))
                for i, p in enumerate(result.pose_landmarks)]

    def process(self, frame, timestamp):
        return self.decode(self.task.detect_for_video(image(frame), timestamp))
