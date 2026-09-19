"""Mock tracker contracts, multi-hand continuity, module dependencies and privacy tests."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import numpy as np
from unittest.mock import Mock
from trackers.hands import HandTracker
from trackers.face import FaceTracker
from trackers.pose import PoseTracker
from trackers.common import euler, crop
from trackers.iris import IrisTracker
from trackers.gaze import GazeTracker
from trackers.expressions import ExpressionTracker
from trackers.auto_camera import AutoCamera
from trackers.age import AgeTracker
from trackers.recognition import FaceStore, RecognitionTracker
from hand_effects import HandEffects
from pipeline import Pipeline

ROOT=Path(__file__).resolve().parent
def marks(n):
    return [NS(x=.5,y=.5,z=.01,visibility=.99) for _ in range(n)]

def hand(x,side):
    p=np.zeros((21,3))
    p[:,:2]=[x,.5]
    p[5,:2]=[x-.08,.5];p[17,:2]=[x+.08,.5]
    p[4,:2]=[x,.3];p[8,:2]=[x+.01,.3]
    return {"points":p,"world":p.copy(),"side":side}

def face():
    p=np.full((478,3),.5)
    # Construct two open eyes with known centers and pixel radii.
    for c,ring,ends,lids,x in ((468,[469,470,471,472],[33,133],[159,145],.35),
                              (473,[474,475,476,477],[362,263],[386,374],.65)):
        p[c,:2]=[x,.4]
        p[ring,:2]=[[x-.01,.4],[x,.39],[x+.01,.4],[x,.41]]
        p[ends,:2]=[[x-.05,.4],[x+.05,.4]]
        p[lids,:2]=[[x,.38],[x,.42]]
    return {"points":p,"box":np.array([.2,.2,.8,.8]),"angles":(0,0,0),"blendshapes":{}}

class SuiteTests(unittest.TestCase):
    def test_hand_tasks_decode_keeps_world_and_depth(self):
        result=NS(hand_landmarks=[marks(21)],hand_world_landmarks=[marks(21)],
                  handedness=[[NS(category_name="Left")]])
        h=HandTracker.decode(result)[0]
        self.assertEqual(h["world"].shape,(21,3))
        self.assertEqual(h["points"][0,2],.01)
        self.assertEqual(h["side"],"Left")

    def test_face_tasks_decode_and_rotation(self):
        angle=math.radians(30)
        r=np.array([[math.cos(angle),0,math.sin(angle),0],[0,1,0,0],
                    [-math.sin(angle),0,math.cos(angle),0],[0,0,0,1]])
        result=NS(face_landmarks=[marks(478)],facial_transformation_matrixes=[r],
                  face_blendshapes=[[NS(category_name="jawOpen",score=.8)]])
        out=FaceTracker.decode(result)[0]
        self.assertAlmostEqual(out["angles"][1],30)
        self.assertEqual(out["blendshapes"]["jawOpen"],.8)
        self.assertEqual(euler(np.eye(4)),(0,0,0))

    def test_pose_tasks_decode_visibility_and_world(self):
        result=NS(pose_landmarks=[marks(33)],pose_world_landmarks=[marks(33)])
        out=PoseTracker.decode(result)[0]
        self.assertEqual(out["world"].shape,(33,3))
        self.assertEqual(len(out["visibility"]),33)

    def test_iris_geometry_and_closed_eye(self):
        f=face()
        eyes=IrisTracker.process(f,(1000,1000,3))
        self.assertEqual(len(eyes),2)
        self.assertAlmostEqual(eyes[0]["radius"],10)
        self.assertAlmostEqual(eyes[0]["ratio"][0],.5)
        f["points"][145]=f["points"][159]
        self.assertEqual(len(IrisTracker.process(f,(1000,1000,3))),1)

    def test_gaze_calibration_and_direction(self):
        g=GazeTracker(); f=face()
        eyes=IrisTracker.process(f,(1000,1000,3))
        self.assertFalse(g.process(f,eyes)["calibrated"])
        self.assertTrue(g.calibrate())
        self.assertEqual(g.process(f,eyes)["direction"],"center")
        changed=copy.deepcopy(eyes)
        for eye in changed: eye["ratio"]=(.8,.5)
        self.assertEqual(g.process(f,changed)["direction"],"right")
        self.assertIsNone(g.process(f,[]))
        self.assertFalse(g.calibrate())

    def test_expressions_describe_movement(self):
        f=face();f["blendshapes"]={"jawOpen":.9,"mouthSmileLeft":.6,"mouthSmileRight":.6}
        self.assertEqual(ExpressionTracker.process(f),{"smile":.6,"jaw open":.9})

    def test_age_output_and_crop(self):
        out=AgeTracker.decode([0,0,0,0,.9,.05,.05,0])
        self.assertEqual(out["bucket"],"25-32")
        with self.assertRaises(ValueError): AgeTracker.decode([1,2])
        image=np.ones((10,10,3),np.uint8)
        self.assertEqual(crop(image,[-1,-1,2,2]).shape,image.shape)

    def test_local_store_enroll_match_delete(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"faces.json"
            store=FaceStore(path)
            self.assertFalse(path.exists())
            store.enroll("Test Person",[1,0,0])
            self.assertEqual(FaceStore(path).match([1,0,0])[0],"Test Person")
            self.assertEqual(store.match([0,1,0])[0],"Unknown")
            store.delete("Test Person")
            self.assertEqual(FaceStore(path).records,{})
            store.enroll("Other",[0,1,0])
            store.delete_all()
            self.assertFalse(path.exists())
            self.assertEqual(store.match([0,1,0])[0],"Unknown")

    def test_recognition_alignment_and_embedding_contract(self):
        with tempfile.TemporaryDirectory() as d:
            t=RecognitionTracker.__new__(RecognitionTracker)
            t.detector=Mock()
            t.detector.detect.return_value=(1,np.array([[10,10,20,20]+[0]*11],dtype=np.float32))
            t.recognizer=Mock()
            t.recognizer.feature.return_value=np.array([[1,0,0]])
            t.store=FaceStore(Path(d)/"faces.json")
            result=t.process(np.zeros((100,100,3),np.uint8))
            self.assertEqual(result[0]["name"],"Unknown")
            np.testing.assert_allclose(result[0]["box"],[.1,.1,.3,.3])
            t.recognizer.alignCrop.assert_called_once()

    def test_camera_crop_stays_in_bounds_and_returns_full_frame(self):
        camera=AutoCamera({"max_zoom":2})
        image=np.zeros((120,160,3),np.uint8)
        for _ in range(20):
            rendered=camera.apply(image,[np.array([.9,.8,1,1])],.1)
        self.assertEqual(rendered.shape,image.shape)
        self.assertLessEqual(camera.zoom,2)
        zoom=camera.zoom
        camera.apply(image,[],.2)
        self.assertLess(camera.zoom,zoom)

    def test_two_hand_order_reversal_and_release(self):
        fx=HandEffects(); left=hand(.2,"Left");right=hand(.8,"Right")
        first=fx.update([left,right],(720,1280,3),1,.03)
        slot_map={v["hand"]["side"]:v["slot"] for v in first}
        left["points"][:,0]+=.01;right["points"][:,0]-=.01
        second=fx.update([right,left],(720,1280,3),1.03,.03)
        self.assertEqual(slot_map,{v["hand"]["side"]:v["slot"] for v in second})
        self.assertTrue(all(len(s["fx"].active_stroke)==2 for s in fx.slots))
        fx.update([],(720,1280,3),1.1,.03)
        self.assertTrue(all(len(s["fx"].strokes)==1 for s in fx.slots))
        self.assertTrue(all(not s["pinch"] for s in fx.slots))

    def test_module_dependencies_and_failure_isolation(self):
        config=json.loads((ROOT/"config.json").read_text())
        config["modules"]={k:False for k in config["modules"]}
        config["modules"]["iris"]=True
        pipe=Pipeline(config,ROOT)
        fake=Mock()
        fake.process.return_value=[face()]
        pipe.instances["face"]=fake
        result=pipe.process(np.zeros((100,100,3),np.uint8),1)
        self.assertEqual(len(result["face"]),1)
        self.assertEqual(len(result["face"][0]["irises"]),2)
        self.assertEqual(result["hands"],[])
        pipe.toggle("iris")
        self.assertEqual(pipe.process(np.zeros((100,100,3),np.uint8),2)["face"],[])
        pipe.enabled["hands"]=True
        pipe.factories["hands"]=Mock(side_effect=RuntimeError("missing model"))
        pipe.process(np.zeros((100,100,3),np.uint8),3)
        self.assertIn("hands",pipe.errors)
        pipe.close()

    def test_adaptive_scheduler_and_identity_loss(self):
        config=json.loads((ROOT/"config.json").read_text())
        pipe=Pipeline(config,ROOT)
        pipe.counter=30; pipe.adapt(.1)
        self.assertGreater(pipe.stride,1)
        first=face();pipe._identify_faces([first],1)
        second=face();pipe._identify_faces([second],1.1)
        self.assertEqual(first["id"],second["id"])
        third=face();pipe._identify_faces([third],2)
        self.assertNotEqual(first["id"],third["id"])

if __name__=="__main__": unittest.main()
