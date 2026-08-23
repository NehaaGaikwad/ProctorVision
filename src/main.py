import cv2
from camera import Camera
from face_tracker import FaceTracker

camera = Camera()
face_tracker = FaceTracker()

while True:
    frame = camera.read()

    if frame is None:
        print("Camera frame could not be read.")
        break

    frame, status, face_count, results = face_tracker.process_frame(frame)

    fps = camera.calculate_fps()
    height, width = frame.shape[:2]

    # FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Camera status
    cv2.putText(
        frame,
        "Camera: ACTIVE",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Resolution
    cv2.putText(
        frame,
        f"Resolution: {width}x{height}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Face Status (Green if detected, Red if not)
    color = (0, 255, 0) if status == "DETECTED" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Face: {status}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Face Count (Red if more than 1 person, Green otherwise)
    count_color = (0, 0, 255) if face_count > 1 else (0, 255, 0)
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        count_color,
        2
    )

    cv2.imshow("ProctorVision AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()