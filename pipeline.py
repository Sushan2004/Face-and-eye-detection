"""Shared inference, dependency resolution, lazy initialization and failure isolation."""
import time
from pathlib import Path
import cv2
import numpy as np
from trackers.hands import HandTracker
from trackers.face import FaceTracker
from trackers.pose import PoseTracker
from trackers.iris import IrisTracker
from trackers.gaze import GazeTracker
from trackers.expressions import ExpressionTracker
from trackers.age import AgeTracker
from trackers.recognition import RecognitionTracker
from trackers.auto_camera import AutoCamera

FACE_USERS=("face","iris","gaze","expressions","age","auto_camera","boxes")
class Pipeline:
    def __init__(self,config,root):
        self.config=config
        self.enabled=config["modules"].copy()
        self.root=Path(root)
        self.models=self.root/config["models_dir"]
        self.instances={}
        self.errors={}
        self.cache={}
        self.last={}
        self.stride=1
        self.inference_width=config["camera"]["inference_width"]
        self.counter=0
        self.last_timestamp=-1
        self.gaze=GazeTracker()
        self.auto_camera=AutoCamera(config["auto_camera"])
        self.face_tracks={}
        self.next_face_id=0
        self.gaze_primary=None
        self.heavy_cache={}
        self.last_heavy=-1.
        self.metrics={}
        self.factories={
            "hands":lambda:HandTracker(self.models/"hands.task",config["tracking"]),
            "face":lambda:FaceTracker(self.models/"face.task",config["tracking"]),
            "pose":lambda:PoseTracker(self.models/"pose.task",config["tracking"]),
            "age":lambda:AgeTracker(self.models),
            "recognition":lambda:RecognitionTracker(self.models,config["recognition"])
        }

    def toggle(self,name):
        self.enabled[name]=not self.enabled[name]
        self.errors.clear()  # allows retry after fixing missing assets
        self.cache.clear()
        self.last.clear()
        self.heavy_cache.clear()
        if name=="gaze": self.gaze=GazeTracker()
        return self.enabled[name]

    def _run(self,name,frame,timestamp):
        if name in self.errors: return []
        try:
            if name not in self.instances:
                self.instances[name]=self.factories[name]()
            start=time.perf_counter()
            result=self.instances[name].process(frame) if name=="recognition" else self.instances[name].process(frame,timestamp)
            self.metrics[name]=(time.perf_counter()-start)*1000
            return result
        except Exception as exc:
            self.errors[name] = ("macOS graphics context unavailable; run python main.py in local Terminal"
                                 if "NSOpenGL" in str(exc) else str(exc).splitlines()[0][:160])
            return []

    def _identify_faces(self,faces,now):
        available={i:t for i,t in self.face_tracks.items() if now-t["time"]<.5}
        for face in faces:
            center=(face["box"][:2]+face["box"][2:])/2
            candidate=min(available,key=lambda i:np.linalg.norm(center-available[i]["center"]),default=None)
            if candidate is None or np.linalg.norm(center-available[candidate]["center"])>.15:
                candidate=self.next_face_id
                self.next_face_id+=1
            else:
                available.pop(candidate)
            face["id"]=candidate
            self.face_tracks[candidate]={"center":center,"time":now}
        self.face_tracks={i:t for i,t in self.face_tracks.items() if now-t["time"]<.5}

    def process(self,frame,now):
        self.counter+=1
        timestamp=max(self.last_timestamp+1,int(now*1000))
        self.last_timestamp=timestamp
        width=min(frame.shape[1],self.inference_width)
        small=cv2.resize(frame,(width,round(frame.shape[0]*width/frame.shape[1])))
        results={}
        required={"hands":self.enabled["hands"],
                  "face":any(self.enabled[k] for k in FACE_USERS),
                  "pose":self.enabled["pose"]}
        for name,needed in required.items():
            if not needed:
                results[name]=[]
                continue
            due=name=="hands" or name not in self.cache or self.counter%self.stride==0
            if due:
                self.cache[name]=self._run(name,small,timestamp)
                self.last[name]=now
                if name=="face": self._identify_faces(self.cache[name],now)
            # Never keep overlays indefinitely after inference stalls.
            results[name]=self.cache.get(name,[]) if now-self.last.get(name,-1)<.25 else []

        for face in results["face"]:
            face["irises"]=IrisTracker.process(face,frame.shape) if self.enabled["iris"] or self.enabled["gaze"] else []
            face["expressions"]=ExpressionTracker.process(face) if self.enabled["expressions"] else {}
            face.pop("age",None)
            face.pop("gaze",None)
        faces=results["face"]
        if self.enabled["gaze"] and faces:
            primary=max(faces,key=lambda f:np.prod(f["box"][2:]-f["box"][:2]))
            if self.gaze_primary!=primary["id"]:
                self.gaze=GazeTracker()
                self.gaze_primary=primary["id"]
            primary["gaze"]=self.gaze.process(primary,primary["irises"])
        elif not faces:
            self.gaze.last_features=None

        # At most one expensive model call per frame; age faces are serviced fairly.
        heavy_ran=False
        if self.enabled["recognition"]:
            interval=self.config["recognition"]["interval_seconds"]
            if now-self.last.get("recognition",-1e9)>=interval:
                self.cache["recognition"]=self._run("recognition",small,timestamp)
                self.last["recognition"]=now
                heavy_ran=True
            # Results are deliberately short-lived: stale identities are not projected indefinitely.
            results["recognition"]=self.cache.get("recognition",[]) if now-self.last.get("recognition",-1) < .25 else []
        else: results["recognition"]=[]

        if self.enabled["age"] and faces:
            interval=self.config["age"]["interval_seconds"]
            due=[f for f in faces if now-self.heavy_cache.get(f["id"],(-1e9,None))[0]>=interval]
            if due and not heavy_ran and "age" not in self.errors:
                face=min(due,key=lambda f:self.heavy_cache.get(f["id"],(-1e9,None))[0])
                try:
                    if "age" not in self.instances: self.instances["age"]=self.factories["age"]()
                    start=time.perf_counter()
                    age=self.instances["age"].process(small,face)
                    self.metrics["age"]=(time.perf_counter()-start)*1000
                    self.heavy_cache[face["id"]]=(now,age)
                except Exception as exc: self.errors["age"]=str(exc)[:160]
            for face in faces:
                cached=self.heavy_cache.get(face["id"])
                if cached and now-cached[0]<interval*2: face["age"]=cached[1]
        self.heavy_cache={i:v for i,v in self.heavy_cache.items() if i in self.face_tracks}
        return results

    def adapt(self,elapsed):
        if not self.config["performance"]["adaptive"] or self.counter%30: return
        target=1/self.config["performance"]["target_fps"]
        if elapsed>target*1.2:
            self.stride=min(self.config["performance"]["max_stride"],self.stride+1)
            if self.stride==self.config["performance"]["max_stride"]:
                self.inference_width=max(256,self.inference_width-64)
        elif elapsed<target*.7:
            self.stride=max(1,self.stride-1)
            self.inference_width=min(self.config["camera"]["inference_width"],self.inference_width+32)

    def close(self):
        for tracker in self.instances.values():
            if hasattr(tracker,"close"): tracker.close()
