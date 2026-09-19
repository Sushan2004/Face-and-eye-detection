"""Legacy gesture rules, preserved independently of rendering."""
import math
TIP_IDS = (4,8,12,16,20)

def distance(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])

def finger_states(points, handedness):
    thumb = points[4][0] < points[3][0] if handedness == "Right" else points[4][0] > points[3][0]
    return [thumb] + [points[t][1] < points[t-2][1] for t in TIP_IDS[1:]]

def classify_gesture(fingers,pinch):
    if pinch: return "PINCH"
    if not any(fingers): return "FIST"
    if all(fingers): return "OPEN PALM"
    if fingers == [False,True,False,False,False]: return "POINT"
    if fingers == [False,True,True,False,False]: return "PEACE"
    if fingers == [True,False,False,False,True]: return "CALL ME"
    return f"{sum(fingers)} FINGERS"
