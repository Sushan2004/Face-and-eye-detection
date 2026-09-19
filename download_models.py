"""Explicit setup-only downloader. Running the webcam application never uses a network."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = {
 "hands.task": ("tracking","https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
 "face.task": ("tracking","https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
 "pose.task": ("tracking","https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
 "yunet.onnx": ("recognition","https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
 "sface.onnx": ("recognition","https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
 "age.caffemodel": ("age","https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel"),
 "age.prototxt": ("age","https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt")
}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--group",choices=("all","tracking","recognition","age"),default="tracking")
    args=parser.parse_args()
    directory=ROOT/"models"
    directory.mkdir(exist_ok=True)
    manifest={}
    for name,(group,url) in MODELS.items():
        if args.group not in ("all",group): continue
        path=directory/name
        if not path.exists():
            print("Downloading",name,flush=True)
            temporary=path.with_suffix(path.suffix+".part")
            try:
                with urllib.request.urlopen(url,timeout=90) as response, temporary.open("wb") as target:
                    while True:
                        block=response.read(1024*1024)
                        if not block: break
                        target.write(block)
                if temporary.stat().st_size<100:
                    raise ValueError("Model download was unexpectedly small")
                temporary.replace(path)
            finally:
                if temporary.exists(): temporary.unlink()
        manifest[name]={"source":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes":path.stat().st_size}
    prior=directory/"downloads.json"
    existing=json.loads(prior.read_text()) if prior.exists() else {}
    existing.update(manifest)
    prior.write_text(json.dumps(existing,indent=2))
    print("Models ready",flush=True)
if __name__=="__main__": main()
