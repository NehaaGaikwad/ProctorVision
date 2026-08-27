import cv2
import threading

from camera import Camera
from face_tracker import FaceTracker
from voice_detector import VoiceDetector
from monitor import WindowMonitor


camera = Camera()
face_tracker = FaceTracker()
voice_detector = VoiceDetector()
window_monitor = WindowMonitor(
    exam_window_name="ProctorVision AI"
)

voice_thread = threading.Thread(
    target=voice_detector.start,
    daemon=True
)

voice_thread.start()

window_name = "ProctorVision AI"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

missing_face_frames = 0
multiple_face_frames = 0
gaze_frames = 0

missing_threshold = 15
multiple_threshold = 10
gaze_threshold = 90

current_warning = "CLEAN"
previous_gaze_status = "CENTER"


while True:

    frame = camera.read()

    if frame is None:
        print("Camera frame could not be read.")
        break

    (
        frame,
        status,
        face_count,
        gaze_status,
        results
    ) = face_tracker.process_frame(frame)

    window_status = window_monitor.check()

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

    face_color = (
        (0, 255, 0)
        if status == "DETECTED"
        else
        (0, 0, 255)
    )

    cv2.putText(
        frame,
        f"Face: {status}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        face_color,
        2
    )

    voice_probability = (
        voice_detector.current_probability
    )

    voice_duration = (
        voice_detector.current_duration
    )

    if voice_detector.voice_violation:

        voice_color = (0, 0, 255)
        voice_display = "Voice: SUSPICIOUS"

    elif voice_detector.speech_active:

        voice_color = (0, 165, 255)
        voice_display = "Voice: DETECTED"

    else:

        voice_color = (0, 255, 0)
        voice_display = "Voice: NO VOICE"

    cv2.putText(
        frame,
        voice_display,
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        voice_color,
        2
    )

    cv2.putText(
        frame,
        f"Voice Probability: {voice_probability:.2f}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if voice_detector.speech_active:

        cv2.putText(
            frame,
            f"Voice Duration: {voice_duration:.1f}s",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    violation_msg = "CLEAN"

    if status == "NOT DETECTED":

        missing_face_frames += 1
        multiple_face_frames = 0
        gaze_frames = 0

        previous_gaze_status = "CENTER"

        if missing_face_frames >= missing_threshold:

            violation_msg = (
                "WARNING: Face Not Detected!"
            )

    elif face_count > 1:

        multiple_face_frames += 1
        missing_face_frames = 0
        gaze_frames = 0

        previous_gaze_status = "CENTER"

        if multiple_face_frames >= multiple_threshold:

            violation_msg = (
                "WARNING: Multiple Faces Detected!"
            )

    elif gaze_status != "CENTER":

        missing_face_frames = 0
        multiple_face_frames = 0

        if gaze_status != previous_gaze_status:

            gaze_frames = 0

        gaze_frames += 1

        previous_gaze_status = gaze_status

        if gaze_frames >= gaze_threshold:

            violation_msg = (
                f"WARNING: {gaze_status}!"
            )

    else:

        missing_face_frames = 0
        multiple_face_frames = 0
        gaze_frames = 0

        previous_gaze_status = "CENTER"

        violation_msg = "CLEAN"

    current_warning = violation_msg

    warnings = []

    if current_warning != "CLEAN":

        warnings.append(
            current_warning
        )

    if voice_detector.voice_violation:

        warnings.append(
            "WARNING: Suspicious Voice Activity!"
        )

    elif voice_detector.speech_active:

        warnings.append(
            "WARNING: Voice Detected!"
        )

    if window_status["active"]:

        warnings.append(
            "WARNING: Window Switch!"
        )

    if warnings:

        warning_text = " | ".join(
            warnings
        )

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            warning_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        bar_width = text_width + 50
        bar_height = 55

        cv2.rectangle(
            frame,
            (0, 0),
            (bar_width, bar_height),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            warning_text,
            (20, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        status_text = "STATUS: CLEAN"

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            status_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        bar_width = text_width + 50
        bar_height = 55

        cv2.rectangle(
            frame,
            (0, 0),
            (bar_width, bar_height),
            (0, 100, 0),
            -1
        )

        cv2.putText(
            frame,
            status_text,
            (20, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        window_name,
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


window_monitor.finalize()

camera.release()
cv2.destroyAllWindows()