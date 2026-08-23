import cv2
from camera import Camera
from face_tracker import FaceTracker

camera = Camera()
face_tracker = FaceTracker()

window_name = "ProctorVision AI"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

missing_face_frames = 0
multiple_face_frames = 0
gaze_frames = 0

missing_threshold = 15
multiple_threshold = 10
gaze_threshold = 90 

current_warning = "CLEAN"

while True:
    frame = camera.read()

    if frame is None:
        print("Camera frame could not be read.")
        break

    frame, status, face_count, gaze_status, results = face_tracker.process_frame(frame)

    fps = camera.calculate_fps()
    height, width = frame.shape[:2]

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Camera: ACTIVE",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Resolution: {width}x{height}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

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

    violation_msg = "CLEAN"

    if status == "NOT DETECTED":
        missing_face_frames += 1
        multiple_face_frames = 0
        gaze_frames = 0
        if missing_face_frames >= missing_threshold:
            violation_msg = "WARNING: Face Not Detected!"
    elif face_count > 1:
        multiple_face_frames += 1
        missing_face_frames = 0
        gaze_frames = 0
        if multiple_face_frames >= multiple_threshold:
            violation_msg = "WARNING: Multiple Faces Detected!"
    elif gaze_status != "CENTER":
        gaze_frames += 1
        missing_face_frames = 0
        multiple_face_frames = 0
        if gaze_frames >= gaze_threshold:
            violation_msg = f"WARNING: {gaze_status}!"
    else:
        missing_face_frames = 0
        multiple_face_frames = 0
        gaze_frames = 0
        violation_msg = "CLEAN"

    current_warning = violation_msg

    if current_warning != "CLEAN":
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 255), -1)
        cv2.putText(
            frame,
            current_warning,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()