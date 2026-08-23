import cv2

# Initialize the default camera
camera = cv2.VideoCapture(0)

# Load OpenCV's pre-trained Haar Cascade face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Continuously capture frames from the camera
while True:
    success, frame = camera.read()

    # Stop the program if a frame cannot be captured
    if not success:
        print("Could not access the camera.")
        break

    # Convert the color frame to grayscale
    # Face detection works with grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Count the number of faces detected in the current frame
    face_count = len(faces)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # Green bounding box
            2             # Thickness of the box
        )

    # Display the number of detected faces on the camera feed
    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the camera feed with detected faces
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera when the program ends
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()