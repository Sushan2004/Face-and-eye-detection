"""Associate hands spatially; retain separate pen state and canvas for both slots."""
import itertools
import time
from collections import deque
import numpy as np
from effects import SpellEffects
from gestures import classify_gesture, finger_states, distance
from trackers.common import pixels

class HandEffects:
    def __init__(self):
        self.slots = [dict(fx=SpellEffects(),point=None,last=-1.,pinch=False,side=None,trail=deque(maxlen=24)) for _ in range(2)]

    def assign(self, hands, now):
        if not hands: return []
        wrists = [h["points"][0,:2] for h in hands[:2]]
        def cost(order):
            total=0
            for i,s in enumerate(order):
                slot=self.slots[s]
                valid=slot["point"] is not None and now-slot["last"]<.5
                total += np.linalg.norm(wrists[i]-slot["point"]) if valid else .5
                if valid and slot["side"]!=hands[i]["side"]: total+=.05
            return total
        order=min(itertools.permutations(range(2),len(wrists)),key=cost)
        return list(zip(order,hands))

    def update(self,hands,shape,now,dt,gestures=True):
        assignments=self.assign(hands,now)
        used=set()
        output=[]
        for slot_id,hand in assignments:
            slot=self.slots[slot_id]
            point=hand["points"][0,:2]
            discontinuity=(now-slot["last"]>.25 or
                           (slot["point"] is not None and np.linalg.norm(point-slot["point"])>.25))
            if discontinuity:
                slot["fx"].update("NONE",(0,0),(0,0),0)
                slot["pinch"]=False
                slot["trail"].clear()
            slot.update(point=point.copy(),last=now,side=hand["side"])
            used.add(slot_id)
            points=pixels(hand["points"],shape)
            slot["trail"].append(tuple(points[8]))
            ratio=distance(points[4],points[8])/max(distance(points[5],points[17]),1)
            slot["pinch"]=bool(gestures and ratio < (.50 if slot["pinch"] else .38))
            gesture=classify_gesture(finger_states(points,hand["side"]),slot["pinch"]) if gestures else "NONE"
            tip=tuple(((points[4]+points[8])//2).tolist())
            palm=tuple(np.mean(points[[0,5,9,13,17]],axis=0).astype(int))
            slot["fx"].update(gesture,tip,palm,dt)
            output.append(dict(slot=slot_id,gesture=gesture,palm=palm,hand=hand,trail=list(slot["trail"])))
        for s,slot in enumerate(self.slots):
            if s not in used:
                slot["pinch"]=False
                slot["trail"].clear()
                slot["fx"].update("NONE",(0,0),(0,0),dt)
        return output

    def render(self,frame,active):
        centers={v["slot"]:v["palm"] for v in active}
        art=np.zeros_like(frame)
        for i,slot in enumerate(self.slots):
            slot["fx"].render(frame,centers.get(i),art=art,composite=False)
        SpellEffects.composite(frame,art)

    def clear(self):
        for slot in self.slots: slot["fx"].clear()
