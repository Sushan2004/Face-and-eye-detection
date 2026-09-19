"""Smoothed digital framing. Rendering stays in camera coordinates until final crop."""
import math
import cv2
import numpy as np

class AutoCamera:
    def __init__(self, config):
        self.config = config
        self.center = np.array([.5,.5])
        self.zoom = 1.0
        self.last_target = None

    def apply(self, frame, subjects, dt):
        chosen = None
        if subjects:
            # Prefer the existing subject by center distance; initially largest.
            if self.last_target is None:
                chosen = max(subjects, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            else:
                chosen = min(subjects, key=lambda b:np.linalg.norm((b[:2]+b[2:])/2-self.last_target))
        if chosen is None:
            target, zoom = np.array([.5,.5]), 1.0
        else:
            target = (chosen[:2]+chosen[2:])/2
            self.last_target = target
            extent = max(chosen[2]-chosen[0], chosen[3]-chosen[1])
            zoom = min(self.config.get("max_zoom",2.0), max(1., .65/max(extent,.1)))
        alpha = 1 - math.exp(-3*max(dt,0))
        self.center += alpha*(target-self.center)
        self.zoom += alpha*(zoom-self.zoom)
        h,w = frame.shape[:2]
        cw,ch = max(1,int(w/self.zoom)),max(1,int(h/self.zoom))
        x = int(np.clip(self.center[0]*w-cw/2,0,w-cw))
        y = int(np.clip(self.center[1]*h-ch/2,0,h-ch))
        return cv2.resize(frame[y:y+ch,x:x+cw],(w,h))
