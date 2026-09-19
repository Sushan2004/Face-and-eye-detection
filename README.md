# Air Canvas Vision Suite

The webcam application now has a shared, configurable pipeline for hands, face mesh,
body pose, iris geometry, age estimates, local face recognition, and digital framing.
Pinch drawing, sparks, rotating runes, and the original gesture rules remain available.
Each tracked hand has its own pen state and strokes.

## Run

From this project's directory, use its existing environment:

```bash
cd "/Users/sushan_adhikari/Library/CloudStorage/OneDrive-MissouriStateUniversity/codex Project/Face-and-eye-detection-main"
.venv/bin/python main.py
```

For a new checkout (tested dependency versions are in requirements.txt):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py --group all
python main.py
```

The downloader is an explicit setup step. The webcam app does not download models
or make network calls. The downloaded models are already present in this workspace.
Use Python 3.9–3.12 for these pinned MediaPipe wheels. On macOS, launch from local
Terminal and grant camera access when prompted.

## Controls

| Key | Toggle / action |
| --- | --- |
| 1 | Hand tracking (image xyz and world xyz) |
| 2 | Face mesh and pitch/yaw/roll display |
| 3 | Full-body pose |
| 4 | Iris position and apparent radius in pixels |
| 5 | Approximate gaze direction for the largest face |
| 6 | Age bucket estimates |
| 7 | Visible expression scores: smile, jaw opening, blinking, brow raising |
| 8 | Local face recognition |
| 9 | Digital auto-framing |
| 0 | Gesture classification |
| V | Drawing and spell effects |
| F | Face and eye bounding boxes |
| D | Hand landmark numbers and gesture/depth details |
| H | Module/status help overlay |
| K | Calibrate neutral gaze while looking at the preview center |
| E | Freeze a fresh, single-face detection and type an enrollment name |
| X twice within 3 seconds | Delete all enrolled face embeddings |
| C | Clear both air canvases |
| S | Save the preview into captures/ |
| Q or Esc | Quit (Esc cancels enrollment while typing a name) |

In enrollment mode, Enter saves, Backspace edits, and Esc cancels. Names accept
up to 40 printable ASCII characters. Enrolling the same name replaces its stored
embedding. Exactly one face must be visible. Enrollment is explicit; detection
alone never saves biometric data.

A pinch puts the pen down (ratio below 0.38), moving draws, and releasing lifts the
pen (ratio at least 0.50). Open palm charges a rune. Fist, point, peace and call-me
classification retain the old rules. The hand-association layer matches wrists
spatially, with a small handedness preference, and ends a stroke after lost tracking
or a large position jump. Severe occlusion/crossing can still make hand identity
ambiguous; this is not biometric hand identification.

## Capability status and limits

| Requested capability | Implemented behavior |
| --- | --- |
| Automation camera | Smooth digital crop/zoom toward a tracked face, or body when pose is enabled; returns toward full frame after loss |
| Age detection | Local eight-bucket age CNN, evaluated on detected faces periodically |
| 3D face / rotation | MediaPipe Tasks mesh, transformation matrix, pitch/yaw/roll |
| Face description / recognition | Visible movement description and opt-in SFace embeddings with named enrollment |
| Body pose | MediaPipe Tasks Pose Landmarker Lite: 33 landmarks, visibility and world coordinates |
| 3D hands | MediaPipe Tasks Hand Landmarker: 21 image xyz and model-estimated world xyz points per hand |
| Iris analysis | Iris center, apparent radius, and within-eye ratios; not pupil segmentation |
| Age / gender / emotion | Age plus observable expression scores; gender and internal emotional-state inference are not implemented |
| Gaze | Neutral-calibrated, heuristic up/down/left/right/center; not accurate screen-point estimation |
| Gestures | Original rule set and independent hand effects; body/face gesture triggers remain a stretch goal |

World coordinates are model estimates, not sensor-measured depth. Iris radius is
not a physical measurement or a pupil-dilation reading. Head angles are in the
MediaPipe canonical transform convention; the preview is mirrored by default.
Gaze combines iris displacement with head angles using a heuristic; recalibrate
with K after moving the camera or changing your viewing position. It resets when
the primary face identity changes. Recognition cosine scores, expression scores,
and age model scores are not calibrated probabilities.

The age model uses the original labels: 0–2, 4–6, 8–13, 15–20, 25–32, 38–43,
48–53, 60+. These are approximate visual estimates with gaps in their training
labels; do not use them to verify someone's age. The code displays the predicted
bucket instead of inventing a precise numeric age.

## Configuration and architecture

Edit config.json; pass a different file with `python main.py --config path.json`.
Keyboard changes apply to the current session only.

- camera: device index, capture dimensions, mirror, inference_width (default 640).
- modules: independent display/feature switches. Hands, gestures, effects and mesh
  start enabled. Expensive classifiers and other modules start disabled.
- tracking: hand confidence and maximum faces.
- performance: target_fps, adaptive scheduling, maximum face/pose stride.
- age.interval_seconds: per-face CNN cadence (default 1.5 seconds).
- recognition: inference interval, cosine threshold, local database path.
- auto_camera.max_zoom: maximum digital zoom.
- models_dir: relative to the project directory.
- display.debug: initial help/status display.

Dependencies are automatic: disabling mesh display does not remove the face data
needed by enabled iris/gaze/age/expression/framing modules. Those features share
one face inference pass. Recognition uses its own YuNet alignment detector and
does not depend on the mesh toggle. Disabling hands stops hand inference.

```text
main.py -> suite_app.py (capture, keyboard, enrollment, cleanup)
                    -> pipeline.py (dependencies, inference, caches, failures)
                         -> trackers/hands.py
                         -> trackers/face.py
                         -> trackers/pose.py
                         -> trackers/iris.py
                         -> trackers/gaze.py
                         -> trackers/age.py
                         -> trackers/expressions.py
                         -> trackers/recognition.py
                         -> trackers/auto_camera.py
                    -> gestures.py -> hand_effects.py -> effects.py
                    -> ui.py -> final digital crop -> status overlay
```

The original entry point is retained as legacy_main.py for comparison. Its tracking
still uses the original Solutions API; the default main.py uses Tasks. No undo,
shape recognition, or object manipulation was added.

## Performance and failure behavior

Actual full-loop elapsed time (capture, inference, rendering, and keyboard wait)
is averaged for the FPS readout. A 30 FPS target is configuration, not a benchmark.
Tracking inference is downscaled, face/pose cadence adapts, and inference width can
fall to 256 pixels under sustained load. Heavy model calls are staggered: at most
one age or recognition call per frame. These synchronous calls can still cause
occasional frame-time spikes. They are not background workers.

Hand inference runs on each frame to preserve pen responsiveness. Heavy results
are cached with bounded lifetimes; recognition labels expire after 0.25 seconds
rather than following a face indefinitely from an old detection. This means labels
can appear intermittently at the default one-second recognition interval. Age
results attach to short-lived spatial face tracks, not raw detection indices.
Two hands share one glow compositing pass; blur is calculated at quarter resolution.
The app reports individual tracker failures and keeps other modules operational.
Toggle a module off/on to retry after resolving a model problem.

## Privacy and local enrollment

Default face database:
`~/Library/Application Support/AirCanvasSuite/faces.json`

This location is deliberately outside the OneDrive project. Only normalized face
embeddings and typed names are stored, with user-only file permissions; enrollment
does not save photographs. Embeddings remain sensitive biometric data even though
they are not photos. The JSON is not encrypted; the OS account protects access.
Do not change the database path to a synced directory if you want local-only storage.
OS backups are outside the application's control.

X twice deletes the active database and clears the in-memory identities. For
programmatic per-name deletion use `FaceStore.delete(name)`. If the OS denies
writing the database, enrollment shows an error and does not claim success.

S explicitly saves the displayed webcam image to captures/. This project is inside
OneDrive, so saved screenshots may sync via OneDrive. No screenshots are saved
automatically. Existing spell_capture files from the earlier app are preserved.

## Models, sources, and reproducibility

| Model | Source |
| --- | --- |
| Hands | [Google Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) |
| Face | [Google Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) |
| Body | [Google Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) |
| Detection / embeddings | [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| Age-only CNN | [Gil Levi / Tal Hassner model repository](https://github.com/GilLevi/AgeGenderDeepLearning) |

Age uses the authors' Caffe format through OpenCV DNN rather than introducing an
ONNX conversion or another runtime. No gender model is downloaded. Model files
retain their upstream license terms; consult their source licenses before redistribution.
downloads.json records source URLs, byte sizes, and SHA-256 digests of local files.
These are recorded checksums, not independently trusted upstream signatures.

## Verification

```bash
.venv/bin/python -m unittest -v
.venv/bin/python verify_models.py
```

On 2026-09-19, 15 camera-free tests passed, including the two existing tests.
They cover synthetic task results, iris geometry, matrix rotation, pose decoding,
gaze calibration, age output, recognition alignment, enrollment/deletion,
independent hand strokes, module dependencies, scheduling and failure isolation.

Real-model smoke checks loaded and executed the age CNN, YuNet and SFace (128-value
embedding). MediaPipe Tasks initialization failed in the restricted tool session
because it could not create NSOpenGLPixelFormat, including with explicit CPU
delegates. The application surfaces that error with local Terminal guidance.
Therefore live Tasks inference, camera permissions, real-world accuracy, and 30 FPS
with all lightweight modules remain unverified. Run verify_models.py in local
Terminal before a webcam demo; it exits nonzero if any model fails.

Manual checks: pinch with both hands, release one while drawing with the other,
reverse hand order, temporarily hide a hand, toggle each module, calibrate gaze,
enroll one consenting person, test an unenrolled person, delete the database,
test auto-framing at image edges, and compare displayed FPS with age/recognition
off and on. Log observed accuracy and FPS rather than extrapolating from mock tests.

See DEVELOPMENT_LOG.md for the historical changes and this migration.
