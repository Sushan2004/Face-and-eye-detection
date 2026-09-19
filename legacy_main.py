"""Preserved pre-suite entry point for comparison and rollback."""
import math
import time
from collections import deque

import cv2
import mediapipe as mp

from effects import SpellEffects

WINDOW_NAME = "Air Canvas Spell Studio"
TIP_IDS = (4, 8, 12, 16, 20)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(points, handedness):
    thumb = points[4][0] < points[3][0] if handedness == "Right" else points[4][0] > points[3][0]
    return [thumb] + [points[tip][1] < points[tip - 2][1] for tip in TIP_IDS[1:]]


def classify_gesture(fingers, pinch):
    if pinch:
        return "PINCH"
    if not any(fingers):
        return "FIST"
    if all(fingers):
        return "OPEN PALM"
    if fingers == [False, True, False, False, False]:
        return "POINT"
    if fingers == [False, True, True, False, False]:
        return "PEACE"
    if fingers == [True, False, False, False, True]:
        return "CALL ME"
    return f"{sum(fingers)} FINGERS"


def draw_panel(frame, lines):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 20 + 28 * len(lines)), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (22, 38 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Allow Terminal/Python camera access.")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    show_face_and_eyes, show_details = False, True
    trail = deque(maxlen=24)
    spell_effects = SpellEffects()
    primary_pinching = False
    previous_time = time.perf_counter()

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.65, min_tracking_confidence=0.65) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            panel = ["Hands: 0", "Gesture: --"]
            primary_gesture = "NONE"
            primary_tip = None
            primary_palm = None

            if result.multi_hand_landmarks:
                panel[0] = f"Hands: {len(result.multi_hand_landmarks)}"
                for hand_index, landmarks in enumerate(result.multi_hand_landmarks):
                    handedness = result.multi_handedness[hand_index].classification[0].label
                    points = [(int(mark.x * width), int(mark.y * height)) for mark in landmarks.landmark]
                    palm_size = max(distance(points[5], points[17]), 1)
                    pinch_ratio = distance(points[4], points[8]) / palm_size
                    # Hysteresis uses separate pen-down and pen-up thresholds.
                    # It prevents noisy landmarks from rapidly breaking a line.
                    if hand_index == 0:
                        pinch = pinch_ratio < (0.50 if primary_pinching else 0.38)
                        primary_pinching = pinch
                    else:
                        pinch = pinch_ratio < 0.42
                    fingers = finger_states(points, handedness)
                    gesture = classify_gesture(fingers, pinch)
                    if hand_index == 0:
                        panel[1:] = [f"Gesture: {gesture}", f"Hand: {handedness}",
                                     f"Fingers: {sum(fingers)}/5", f"Pinch: {pinch_ratio:.2f}"]
                        trail.append(points[8])
                        primary_gesture = gesture
                        # Draw from the midpoint where thumb and index meet.
                        primary_tip = ((points[4][0] + points[8][0]) // 2,
                                       (points[4][1] + points[8][1]) // 2)
                        primary_palm = (
                            sum(points[i][0] for i in (0, 5, 9, 13, 17)) // 5,
                            sum(points[i][1] for i in (0, 5, 9, 13, 17)) // 5,
                        )

                    drawer.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                          styles.get_default_hand_landmarks_style(),
                                          styles.get_default_hand_connections_style())
                    xs, ys = zip(*points)
                    cv2.rectangle(frame, (min(xs) - 15, min(ys) - 15),
                                  (max(xs) + 15, max(ys) + 15), (60, 220, 255), 2)
                    cv2.putText(frame, f"{handedness}: {gesture}",
                                (min(xs), max(30, min(ys) - 24)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (60, 220, 255), 2, cv2.LINE_AA)
                    if show_details:
                        for landmark_id, point in enumerate(points):
                            cv2.putText(frame, str(landmark_id), point, cv2.FONT_HERSHEY_PLAIN,
                                        0.75, (0, 0, 0), 1)
                    color = (0, 60, 255) if pinch else (0, 255, 120)
                    cv2.circle(frame, points[8], 12, color, -1)
                    cv2.line(frame, points[4], points[8], color, 3)
            else:
                trail.clear()
                primary_pinching = False

            now = time.perf_counter()
            dt = min(now - previous_time, 0.05)
            if primary_tip is not None:
                spell_effects.update(primary_gesture, primary_tip, primary_palm, dt)
            else:
                spell_effects.update("NONE", (0, 0), (0, 0), dt)
            spell_effects.render(frame, primary_palm)

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (255, 160, 40), max(2, i // 4))

            if show_face_and_eyes:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)
                    eyes = eye_detector.detectMultiScale(gray[y:y + h // 2, x:x + w], 1.1, 8, minSize=(20, 20))
                    for ex, ey, ew, eh in eyes[:2]:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 120, 80), 2)
                panel.append(f"Faces: {len(faces)}")

            panel.extend([f"FPS: {1 / max(now - previous_time, 0.001):.0f}",
                          "PINCH + move: draw | PALM: ring",
                          "C clear  S save  D detail  Q quit"])
            previous_time = now
            draw_panel(frame, panel)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                show_face_and_eyes = not show_face_and_eyes
            if key == ord("d"):
                show_details = not show_details
            if key == ord("c"):
                spell_effects.clear()
            if key == ord("s"):
                filename = f"spell_capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
