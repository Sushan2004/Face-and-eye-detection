"""Compatibility entry point. Gesture exports preserve existing tests/imports."""
from gestures import classify_gesture, finger_states, distance

if __name__ == "__main__":
    from suite_app import main
    main()
