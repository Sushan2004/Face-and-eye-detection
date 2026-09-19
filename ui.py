"""OpenCV overlays and keyboard mappings."""
import cv2
import numpy as np
from trackers.common import pixels

TOGGLES={"1":"hands","2":"face","3":"pose","4":"iris","5":"gaze","6":"age",
         "7":"expressions","8":"recognition","9":"auto_camera","0":"gestures","v":"effects"}
HAND_LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
            (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE_LINKS=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
            (23,25),(25,27),(27,29),(29,31),(24,26),(26,28),(28,30),(30,32)]

def label(frame,text,point,color=(230,240,255),scale=.48):
    cv2.putText(frame,text,tuple(map(int,point)),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,text,tuple(map(int,point)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def overlays(frame,result,active,enabled,detail):
    for item in active:
        p=pixels(item["hand"]["points"],frame.shape)
        if enabled["effects"]:
            trail=item["trail"]
            for i in range(1,len(trail)):
                cv2.line(frame,trail[i-1],trail[i],(255,160,40),max(2,i//4))
            color=(0,60,255) if item["gesture"]=="PINCH" else (0,255,120)
            cv2.circle(frame,tuple(p[8]),12,color,-1)
            cv2.line(frame,tuple(p[4]),tuple(p[8]),color,3)
        for a,b in HAND_LINKS: cv2.line(frame,tuple(p[a]),tuple(p[b]),(90,220,255),1,cv2.LINE_AA)
        for i,point in enumerate(p):
            cv2.circle(frame,tuple(point),3,(255,220,120),-1)
            if detail: label(frame,str(i),point,scale=.3)
        if detail:
            z=item["hand"]["world"][8,2]
            label(frame,f'{item["hand"]["side"]} {item["gesture"]} | index world z {z:.3f}m',p[0])
    for body in result["pose"]:
        p=pixels(body["points"],frame.shape)
        for a,b in POSE_LINKS:
            if min(body["visibility"][a],body["visibility"][b])>.5:
                cv2.line(frame,tuple(p[a]),tuple(p[b]),(100,255,100),2,cv2.LINE_AA)
    for face in result["face"]:
        p=pixels(face["points"],frame.shape)
        if enabled["boxes"]:
            cv2.rectangle(frame,tuple(p.min(axis=0)),tuple(p.max(axis=0)),(100,255,100),2)
            for indices in ([33,133,159,145],[362,263,386,374]):
                eye=p[indices]
                cv2.rectangle(frame,tuple(eye.min(axis=0)),tuple(eye.max(axis=0)),(255,120,80),2)
        if enabled["face"]:
            # MediaPipe topology is display-only; inference uses Tasks.
            from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
            for a,b in FACEMESH_TESSELATION:
                cv2.line(frame,tuple(p[a]),tuple(p[b]),(95,125,105),1,cv2.LINE_AA)
            pitch,yaw,roll=face["angles"]
            label(frame,f'pitch {pitch:.0f} yaw {yaw:.0f} roll {roll:.0f}',p[10])
        if enabled["iris"]:
            for eye in face["irises"]:
                center=tuple(np.asarray(eye["center"],dtype=int))
                cv2.circle(frame,center,max(1,int(eye["radius"])),(255,170,0),1,cv2.LINE_AA)
                label(frame,f'iris r {eye["radius"]:.1f}px',center,scale=.35)
        descriptions=[]
        if face.get("age"): descriptions.append(f'age~{face["age"]["bucket"]}')
        descriptions.extend(f'{k} {v:.0%}' for k,v in face["expressions"].items())
        if face.get("gaze"): descriptions.append("gaze: "+face["gaze"]["direction"])
        if descriptions: label(frame," | ".join(descriptions),p[152])
    for face in result["recognition"]:
        x,y,x2,y2=(face["box"]*[frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]]).astype(int)
        cv2.rectangle(frame,(x,y),(x2,y2),(230,140,255),2)
        label(frame,f'{face["name"]} | cosine {face["score"]:.2f}',(x,y-8))

def hud(frame,pipeline,fps,status,debug):
    lines=[f'Air Canvas | actual FPS {fps:.1f} | stride {pipeline.stride} | inference {pipeline.inference_width}px | H help']
    if debug:
        lines += [" ".join(f'{k}:{v}={"on" if pipeline.enabled[v] else "off"}' for k,v in list(TOGGLES.items())[i:i+4])
                  for i in range(0,len(TOGGLES),4)]
        lines += ["E enroll | X,X delete faces | K gaze center | F boxes | C clear | S save | D detail | Q quit"]
        lines += [f'{k}: {v}' for k,v in pipeline.errors.items()]
    if status: lines.append(status)
    height=min(frame.shape[0],len(lines)*24+14)
    shade=frame[:height].copy()
    shade[:]=shade*.25
    frame[:height]=shade
    for i,line in enumerate(lines): label(frame,line,(12,24+i*24),scale=.43)
