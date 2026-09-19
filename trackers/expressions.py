"""Describe visible facial movement, not internal emotion or gender."""
class ExpressionTracker:
    @staticmethod
    def process(face):
        values = face["blendshapes"]
        scores = {
            "smile": (values.get("mouthSmileLeft",0)+values.get("mouthSmileRight",0))/2,
            "jaw open": values.get("jawOpen",0),
            "blink": (values.get("eyeBlinkLeft",0)+values.get("eyeBlinkRight",0))/2,
            "brows raised": values.get("browInnerUp",0),
        }
        return {name:float(value) for name,value in scores.items() if value >= .35}
