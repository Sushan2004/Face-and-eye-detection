"""Face mesh, blendshapes, and canonical-face transformation."""
import numpy as np
from .common import TaskTracker, image, xyz, box, euler

class FaceTracker(TaskTracker):
    def __init__(self, model, config):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.task = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=config.get("max_faces", 2),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    @staticmethod
    def decode(result):
        faces = []
        for i, marks in enumerate(result.face_landmarks):
            points = xyz(marks)
            matrix = np.array(result.facial_transformation_matrixes[i])
            faces.append(dict(points=points, box=box(points), matrix=matrix,
                              angles=euler(matrix),
                              blendshapes={s.category_name: s.score for s in result.face_blendshapes[i]}))
        return faces

    def process(self, frame, timestamp):
        return self.decode(self.task.detect_for_video(image(frame), timestamp))
