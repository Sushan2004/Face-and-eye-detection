# Development Log

## 2026-09-19 — GitHub publication preparation

Prepared the modular suite for Sushan2004/Face-and-eye-detection on top of the
existing main branch history. Added exclusions for legacy webcam screenshots
and face-database filenames. Model files, environments, captures, and biometric
data are excluded; the model downloader remains part of the source distribution.
Publication uses a normal commit and fast-forward push, without rewriting history.

## 2026-09-19 — Modular vision suite migration

The dated entries below describe earlier versions; README.md is the current usage guide.

### Scope and implementation

- Preserved the original entry point as legacy_main.py; main.py now delegates to
  suite_app.py and re-exports the old gesture functions for compatibility.
- Added trackers/common.py, hands.py, face.py, pose.py, iris.py, gaze.py, age.py,
  expressions.py, recognition.py and auto_camera.py.
- Default tracking uses MediaPipe Tasks VIDEO mode with monotonic timestamps,
  CPU delegates, image/world hand landmarks, face blendshapes/matrices, and body pose.
- Added pipeline.py for dependency sharing, lazy model setup, per-module failures,
  bounded result caching, age-face association and adaptive scheduling.
- Added gestures.py and hand_effects.py. Each of two spatially associated hand slots
  owns its pinch hysteresis, particles, rune and strokes. Reversed detection order
  is covered by tests. Losing a hand ends its stroke. Existing gestures are preserved.
- effects.py keeps its original rendering API and now supports shared compositing;
  both hands share two quarter-resolution blur layers to reduce rendering cost.
- Added ui.py, config.json, per-module keyboard controls, actual full-loop FPS,
  neutral gaze calibration, screenshot save, frozen-frame face enrollment, and
  two-key deletion of all enrolled identities.
- Added download_models.py and verify_models.py. Downloaded the three MediaPipe
  task assets, YuNet, SFace, and the age-only Caffe model/prototxt (about 97 MB total).
  A manifest records source URLs and locally calculated SHA-256 values.
- Age is an eight-bucket estimate. The combined annotation supports age and visible
  facial-expression movements. Gender and internal emotion inference are excluded.
- Iris geometry is apparent image size/position, not pupil segmentation. Gaze is
  a coarse neutral-calibrated heuristic, not a precise screen coordinate estimate.
- Restored F for face/eye boxes using mesh landmarks; added numeric keys for module
  switches. Runtime toggles do not write configuration automatically.

### Privacy and data

Recognition is disabled by default. E runs a fresh detector, requires exactly one
face, and freezes the frame while a name is typed. Saving writes only the normalized
embedding and name. The database defaults to Application Support outside OneDrive;
user-only file permissions and atomic replacement are used. X twice deletes the
database and in-memory entries. Screenshots remain explicit and may sync because
the project is in OneDrive. There are no runtime network calls or automatic image saves.

### Verification and limitations

- All 15 automated tests passed: original gestures/rendering plus synthetic Tasks
  decoding, iris/gaze geometry, pose, rotation, age output, face-store lifecycle,
  mocked recognition alignment, two-hand continuity, dependencies, failure isolation
  and adaptive cadence. Tests use temporary directories for enrollment data.
- Tests caught a pose bbox conversion error; corrected it to convert MediaPipe
  landmark objects to an array before computing bounds.
- Real age CNN, YuNet detector and SFace embedding smoke calls executed successfully.
- Inspected a generated 1280x720 synthetic preview of both hand effects and face
  overlays. Thirty rendering-only iterations averaged 20.11 ms in this tool session;
  this excludes camera capture and model inference and is not an end-to-end FPS claim.
- MediaPipe Tasks could not initialize in the tool sandbox: NSOpenGLPixelFormat
  creation failed even with explicit CPU delegates. Local Terminal verification is
  still required; this is not recorded as a passing tracking or webcam test.
- Live camera accuracy, enrollment quality and 30 FPS remain unmeasured. Heavy
  models are rate-limited but synchronous, so inference may cause frame-time spikes.
- Shape recognition, undo/redo and object manipulation were not modified.

README.md was updated, the old README was archived under docs/, and model/capture
directories were added to .gitignore. No Git operations, commits or pushes were made.

This file records the development history, technical decisions, tests, and known
limitations of the Air Canvas Spell Studio project. Add a new dated entry whenever
the behavior, dependencies, architecture, or user interface changes.

## Project summary

The project began as a small OpenCV face-detection demo. It has been expanded into
a real-time hand-tracking and gesture-effects application. A webcam tracks 21 hand
landmarks, interprets gestures, and lets the user draw glowing items in the air by
touching the thumb and index finger together.

The project currently processes camera frames locally. It does not upload camera
images or control the operating-system cursor.

## Original project state

### 2026-09-03 — Initial inspection

The repository originally contained:

- `main.py`: an OpenCV webcam loop using a Haar cascade to detect faces.
- `README.md`: described both face and eye detection, although `main.py` only
  implemented face detection.
- `requirements.txt`: requested `opencv-python==5.0.0.93`.

The application drew green rectangles around faces, displayed a face count, and
used `Q` to exit.

## Changes made

### 2026-09-03 — Detailed hand tracking and gesture recognition

Rebuilt `main.py` around MediaPipe Hands while retaining optional OpenCV face and
eye detection.

Added:

- Real-time tracking for up to two hands.
- All 21 MediaPipe landmarks for each detected hand.
- Hand skeleton connections and landmark number labels.
- Left/right handedness reporting.
- Hand bounding boxes.
- Raised-finger counting.
- Index-finger movement trail.
- Gesture classification for:
  - `PINCH`
  - `FIST`
  - `OPEN PALM`
  - `POINT`
  - `PEACE`
  - `CALL ME`
- A live information panel showing gesture, hand, finger count, pinch ratio, and
  frames per second.
- `F` control for optional face and eye detection.
- `D` control for toggling landmark IDs.
- `Q` and `Esc` controls for exiting.

The pinch measurement was normalized using palm width:

```python
pinch_ratio = distance(thumb_tip, index_tip) / palm_size
```

This normalization makes the measurement less sensitive to the hand moving closer
to or farther from the camera.

### 2026-09-03 — Dependency and setup repair

Created a local `.venv` and installed the required packages.

Changed `requirements.txt` to use compatible, reproducible dependencies:

```text
mediapipe==0.10.14
numpy<2
opencv-contrib-python==4.10.0.84
```

The standalone `opencv-python` package was removed because MediaPipe uses the
contrib build, and installing both can cause conflicting `cv2` packages.

Verified versions:

- OpenCV 4.10.0
- MediaPipe 0.10.14

Updated `README.md` with installation instructions, camera-permission guidance,
features, and keyboard controls.

### 2026-09-04 — Procedural spell-effects engine

Created `effects.py` to separate visual rendering from camera tracking and gesture
recognition. This separation makes the code easier to test and extend.

Added the `Particle` data class. Each spark stores:

- Position
- Horizontal and vertical velocity
- Remaining lifetime
- Render size

Particles are updated using frame delta time. A small gravity value changes their
vertical velocity, damping slows them gradually, and expired particles are removed.
The particle list is capped to keep memory use controlled.

Added the `SpellEffects` class with:

- Persistent air-drawn strokes.
- A currently active stroke.
- Spark particle emission.
- A rotating energy rune.
- Open-palm charging animation.
- Large and small Gaussian-blur glow layers.
- Additive visual compositing onto the webcam frame.
- A maximum of 24 stored strokes and 450 live particles.

The energy rune is generated using OpenCV circles, lines, polygons, and
trigonometric rotation. It does not use external or copyrighted movie artwork.

Initial spell controls were:

- `POINT`: draw a glowing stroke.
- `PINCH`: emit sparks.
- `OPEN PALM`: charge an energy ring.
- `FIST`: pause drawing.
- `C`: clear the canvas and particles.
- `S`: save the displayed frame as a PNG.

### 2026-09-04 — Pinch-to-draw interaction

Changed the air canvas to behave like a pen:

- Touching the thumb and index finger puts the pen down.
- Moving while pinched draws a persistent glowing stroke.
- Separating the fingers lifts the pen and stores the completed stroke.
- Beginning a pinch creates a spark burst as pen-down feedback.
- Drawing originates from the midpoint between the thumb and index fingertips.
- `POINT` no longer draws by itself.

Added pinch hysteresis to stabilize drawing:

- Pen-down threshold: pinch ratio below `0.38`.
- Pen-up threshold: pinch ratio above `0.50`.

Using separate start and release thresholds prevents noisy landmark values near one
boundary from rapidly starting and stopping a stroke.

## Current gesture and keyboard controls

| Input | Action |
| --- | --- |
| Pinch thumb and index finger | Put the virtual pen down and release sparks |
| Move while pinched | Draw a glowing air stroke |
| Separate thumb and index finger | Lift the pen and complete the stroke |
| Open palm | Charge and display the rotating energy rune |
| Fist | Pause without clearing the drawing |
| `C` | Clear strokes and particles |
| `S` | Save the current displayed frame as a PNG |
| `D` | Toggle landmark number labels |
| `F` | Toggle face and eye detection |
| `Q` or `Esc` | Quit |

## Automated verification

Created `test_effects.py` using Python's built-in `unittest` framework. The tests do
not require a webcam.

Current tests verify:

- Point, open-palm, and pinch gesture classification.
- Particle creation.
- Completed-stroke storage.
- Rendering visual effects onto a synthetic NumPy frame.

Latest result:

```text
Ran 2 tests in 0.010s
OK
```

The Python source files also pass bytecode compilation.

Run the checks with:

```bash
source .venv/bin/activate
python -m unittest -v test_effects.py
```

## Current file responsibilities

| File | Responsibility |
| --- | --- |
| `main.py` | Camera capture, landmarks, gestures, UI, and keyboard input |
| `effects.py` | Strokes, particles, energy-rune animation, glow, and compositing |
| `test_effects.py` | Camera-free gesture and effects tests |
| `requirements.txt` | Reproducible Python dependencies |
| `README.md` | User-facing setup, usage, and feature overview |
| `DEVELOPMENT_LOG.md` | Complete development history and technical decisions |

## Known limitations

- Live webcam behavior must be evaluated manually because lighting, camera angle,
  background, distance, and hand pose affect landmark quality.
- Gesture classification uses geometric rules rather than a trained custom model.
- Drawings are stored only in memory until a screenshot is saved.
- Screenshots are saved in the directory from which the program is launched.
- Face and eye detection still uses Haar cascades and is not true gaze tracking.
- The program does not yet recognize the shape or meaning of a completed drawing.
- The visual canvas does not yet support undo, colors, brush sizes, or selecting and
  manipulating previously drawn objects.
- The program does not yet control the system cursor.

## Suggested next milestones

1. Add undo and redo for completed strokes.
2. Add gesture-controlled color and brush selection.
3. Recognize closed shapes such as circles, triangles, and rectangles.
4. Convert recognized shapes into selectable screen objects.
5. Use two hands to scale, rotate, and move objects.
6. Record latency, FPS, gesture accuracy, and false pen-down events.
7. Add recorded-video tests for repeatable gesture evaluation.
8. Replace Haar eye boxes with facial and iris landmarks for actual gaze estimation.

## How to record future changes

For every meaningful update, append a section in this format:

```markdown
### YYYY-MM-DD — Short change title

Goal:

Files changed:

Implementation:

Technical decision and reason:

Tests performed:

Known limitations or follow-up work:
```
