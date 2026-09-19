# Air Canvas Spell Studio (pre-suite reference)

A real-time computer-vision demo built with Python, OpenCV, and MediaPipe.

See [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) for the complete implementation
history, technical decisions, tests, and known limitations.

## Features

- Tracks up to two hands with all 21 landmarks per hand
- Draws the hand skeleton, landmark IDs, bounding box, and index-finger trail
- Recognizes pinch, fist, open palm, point, peace, and call-me gestures
- Reports handedness, raised-finger count, pinch distance, and FPS
- Optionally detects faces and eyes
- Draws persistent glowing items by touching thumb and index finger together
- Emits procedural spark particles when the fingers pinch
- Charges a rotating energy rune while an open palm is held up

This is currently a visual tracker. It does not control the system mouse. The index-finger trail and pinch state provide a safe base for adding cursor movement and clicks later.

## Setup and run

```bash
cd ~/Desktop/Face-and-eye-detection-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Run the camera-free automated checks with:

```bash
python -m unittest test_effects.py
```

On macOS, allow camera access when prompted. If previously denied, enable it in **System Settings > Privacy & Security > Camera**.

## Controls

| Key | Action |
| --- | --- |
| `F` | Toggle face and eye detection |
| `D` | Toggle landmark number labels |
| `C` | Clear all air-drawn strokes and sparks |
| `S` | Save the current frame as a PNG |
| `Q` or `Esc` | Quit |

## Spell gestures

| Gesture | Effect |
| --- | --- |
| Touch thumb and index finger, then move | Draw a glowing air stroke |
| Begin a pinch | Release a pen-down spark burst |
| Separate thumb and index finger | Lift the pen and finish the item |
| Hold an open palm | Charge and rotate an energy rune |
| Make a fist | Pause drawing |

The visual effects are generated in real time using particles, alpha blending,
Gaussian blur, and simple geometry; no copyrighted movie artwork is included.

Keep your palm facing the camera in reasonable lighting. Pinching is measured relative to palm size, so it works at different camera distances. Gesture recognition uses simple geometric rules and is intended as a demo.
