"""Real-model, camera-free smoke test. Black frames test execution, not detection accuracy."""
import json
import time
from pathlib import Path
import numpy as np
from pipeline import Pipeline
from trackers.age import AgeTracker

def main():
    root=Path(__file__).resolve().parent
    config=json.loads((root/"config.json").read_text())
    config["modules"].update(hands=True,face=True,pose=True,recognition=True)
    pipe=Pipeline(config,root)
    report={}
    blank=np.zeros((360,640,3),np.uint8)
    try:
        result=pipe.process(blank,time.perf_counter())
        for name in ("hands","face","pose","recognition"):
            report[name]={"ok":name not in pipe.errors,"error":pipe.errors.get(name)}
        try:
            age=AgeTracker(root/"models")
            age.process(blank,{"box":np.array([.1,.1,.9,.9])})
            report["age"]={"ok":True}
        except Exception as exc: report["age"]={"ok":False,"error":str(exc)}
        try:
            recognition=pipe.instances["recognition"]
            output=recognition.recognizer.feature(np.zeros((112,112,3),np.uint8))
            report["embedding"]={"ok":output.shape==(1,128),"shape":list(output.shape)}
        except Exception as exc: report["embedding"]={"ok":False,"error":str(exc)}
    finally: pipe.close()
    print(json.dumps(report,indent=2))
    return 0 if all(v["ok"] for v in report.values()) else 1
if __name__=="__main__": raise SystemExit(main())
