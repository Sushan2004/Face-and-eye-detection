"""Optional local age-bucket CNN. No gender or emotion model is loaded."""
import cv2
import numpy as np
from .common import crop

class AgeTracker:
    BUCKETS = ("0-2","4-6","8-13","15-20","25-32","38-43","48-53","60+")
    def __init__(self, model_dir):
        self.net = cv2.dnn.readNetFromCaffe(str(model_dir/"age.prototxt"),str(model_dir/"age.caffemodel"))

    @classmethod
    def decode(cls, prediction):
        scores = np.asarray(prediction).ravel()
        if scores.size != 8 or not np.all(np.isfinite(scores)):
            raise ValueError("Age model must return eight finite bucket scores")
        i = int(np.argmax(scores))
        return {"bucket":cls.BUCKETS[i],"score":float(scores[i])}

    def process(self, frame, face):
        roi = crop(frame, face["box"])
        if not roi.size:
            return None
        blob = cv2.dnn.blobFromImage(roi,1.,(227,227),
                                    (78.4263377603,87.7689143744,114.895847746),
                                    swapRB=False,crop=False)
        self.net.setInput(blob)
        return self.decode(self.net.forward())
