"""Opt-in local face embeddings with YuNet alignment and SFace."""
import json
import os
from pathlib import Path
import cv2
import numpy as np

class FaceStore:
    def __init__(self, path, threshold=.45):
        self.path = Path(path).expanduser()
        self.threshold = threshold
        self.records = {}
        if self.path.exists():
            with self.path.open() as f:
                data = json.load(f)
            if data.get("model") != "sface-2021dec":
                raise ValueError("Incompatible face database")
            self.records = {k:np.asarray(v,dtype=np.float32) for k,v in data["faces"].items()}

    @staticmethod
    def normalize(vector):
        vector = np.asarray(vector,dtype=np.float32).ravel()
        if not vector.size or not np.all(np.isfinite(vector)) or np.linalg.norm(vector)<1e-8:
            raise ValueError("Invalid embedding")
        return vector / np.linalg.norm(vector)

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        temp = self.path.with_suffix(".tmp")
        fd = os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,"w") as f:
            json.dump({"model":"sface-2021dec","faces":{k:v.tolist() for k,v in self.records.items()}},f)
        os.replace(temp,self.path)
        os.chmod(self.path,0o600)

    def enroll(self,name,embedding):
        name = name.strip()
        if not name or len(name)>40:
            raise ValueError("Name must contain 1-40 characters")
        self.records[name] = self.normalize(embedding)
        self.save()

    def match(self,embedding):
        vector = self.normalize(embedding)
        candidates = [(name,float(np.dot(value,vector))) for name,value in self.records.items()
                      if value.shape == vector.shape]
        if not candidates:
            return "Unknown",0.
        name,score = max(candidates,key=lambda p:p[1])
        return (name if score>=self.threshold else "Unknown"),score

    def delete(self,name):
        self.records.pop(name,None)
        self.save()

    def delete_all(self):
        self.records.clear()
        if self.path.exists():
            self.path.unlink()

class RecognitionTracker:
    def __init__(self, model_dir, config):
        self.detector = cv2.FaceDetectorYN.create(str(model_dir/"yunet.onnx"),"", (320,320),.8,.3,5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_dir/"sface.onnx"),"")
        self.store = FaceStore(config["database"],config.get("threshold",.45))

    def process(self, frame):
        self.detector.setInputSize((frame.shape[1],frame.shape[0]))
        _, detected = self.detector.detect(frame)
        result = []
        if detected is None:
            return result
        for row in detected:
            aligned = self.recognizer.alignCrop(frame,row)
            embedding = self.store.normalize(self.recognizer.feature(aligned))
            name,score = self.store.match(embedding)
            x,y,w,h = row[:4]
            result.append(dict(box=np.array([x/frame.shape[1],y/frame.shape[0],
                                            (x+w)/frame.shape[1],(y+h)/frame.shape[0]]),
                               embedding=embedding,name=name,score=score))
        return result
