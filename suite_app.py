"""Application loop: raw-frame inference, overlays, then digital reframing."""
import argparse
import json
import time
from pathlib import Path
import cv2
from pipeline import Pipeline
from hand_effects import HandEffects
from ui import TOGGLES, overlays, hud, label

ROOT=Path(__file__).resolve().parent

def load_config(path):
    with Path(path).open() as f: config=json.load(f)
    if config["camera"]["inference_width"]<128: raise ValueError("inference_width must be >=128")
    if config["performance"]["target_fps"]<=0: raise ValueError("target_fps must be positive")
    return config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"config.json"))
    args=parser.parse_args()
    config=load_config(args.config)
    pipe=Pipeline(config,ROOT)
    fx=HandEffects()
    camera=None
    status=""
    status_until=0.
    previous=time.perf_counter()
    elapsed_average=1/30
    fps=0.
    debug=config["display"]["debug"]
    detail=True
    enrollment=None
    name=""
    deletion_armed=-10.
    window="Air Canvas Vision Suite"
    try:
        camera=cv2.VideoCapture(config["camera"]["index"])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,config["camera"]["width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,config["camera"]["height"])
        if not camera.isOpened(): raise RuntimeError("Camera unavailable; check macOS camera permission.")
        while True:
            now=time.perf_counter()
            if enrollment is not None:
                preview=enrollment["frame"].copy()
                label(preview,"Enroll this face: "+name+" | Enter save | Esc cancel",(20,50),scale=.65)
                cv2.imshow(window,preview)
                key=cv2.waitKey(20)&0xFF
                if key==27:
                    enrollment=None
                elif key in (10,13):
                    try:
                        pipe.instances["recognition"].store.enroll(name,enrollment["embedding"])
                        status="Face enrolled locally."
                    except Exception as exc: status="Enrollment failed: "+str(exc)[:100]
                    enrollment=None
                    status_until=now+5
                elif key in (8,127): name=name[:-1]
                elif 32<=key<=126 and len(name)<40: name+=chr(key)
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
                continue
            ok,frame=camera.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if config["camera"]["mirror"]: frame=cv2.flip(frame,1)
            dt=min(max(now-previous,0),.1)
            previous=now
            result=pipe.process(frame,now)
            active=fx.update(result["hands"],frame.shape,now,dt,
                             pipe.enabled["gestures"] and pipe.enabled["effects"])
            display=frame.copy()
            if pipe.enabled["effects"]: fx.render(display,active)
            overlays(display,result,active,pipe.enabled,detail)
            if pipe.enabled["auto_camera"]:
                subjects=[p["box"] for p in result["pose"]] or [f["box"] for f in result["face"]]
                display=pipe.auto_camera.apply(display,subjects,dt)
            hud(display,pipe,fps,status if now<status_until else "",debug)
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&0xFF
            if key in (ord("q"),27): break
            if key==ord("d"): detail=not detail
            elif key==ord("h"): debug=not debug
            elif key==ord("f"): pipe.toggle("boxes")
            elif key==ord("c"): fx.clear()
            elif key==ord("k"):
                status="Gaze center calibrated." if pipe.enabled["gaze"] and pipe.gaze.calibrate() else "Enable gaze and show both open eyes first."
                status_until=now+4
            elif key==ord("s"):
                directory=ROOT/"captures"
                directory.mkdir(exist_ok=True)
                path=directory/f"spell_{time.time_ns()}.png"
                status=f"Saved {path.name}" if cv2.imwrite(str(path),display) else "Screenshot save failed."
                status_until=now+4
            elif key==ord("e"):
                if not pipe.enabled["recognition"]:
                    pipe.toggle("recognition")
                # Fresh detection on the frozen raw frame prevents enrolling a stale/bystander result.
                records=pipe._run("recognition",frame,0)
                if len(records)==1:
                    enrollment={"frame":frame.copy(),"embedding":records[0]["embedding"].copy()}
                    name=""
                else:
                    status="Enrollment needs exactly one visible face; inspect model errors with H."
                    status_until=now+5
            elif key==ord("x"):
                if now-deletion_armed<3:
                    try:
                        from trackers.recognition import FaceStore
                        store=pipe.instances["recognition"].store if "recognition" in pipe.instances else FaceStore(config["recognition"]["database"])
                        store.delete_all()
                        pipe.cache.pop("recognition",None)
                        status="Deleted all enrolled face embeddings."
                    except Exception as exc: status="Delete failed: "+str(exc)[:100]
                    deletion_armed=-10
                else:
                    deletion_armed=now
                    status="Press X again within 3 seconds to delete all enrolled faces."
                status_until=now+4
            elif chr(key) in TOGGLES:
                name_toggle=TOGGLES[chr(key)]
                on=pipe.toggle(name_toggle)
                status=f'{name_toggle}: {"on" if on else "off"}'
                status_until=now+3
            elapsed=time.perf_counter()-now
            elapsed_average=.9*elapsed_average+.1*elapsed
            fps=1/max(elapsed_average,.0001)
            pipe.adapt(elapsed_average)
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
    finally:
        pipe.close()
        if camera is not None: camera.release()
        cv2.destroyAllWindows()

if __name__=="__main__": main()
