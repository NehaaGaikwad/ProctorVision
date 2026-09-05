import cv2
import threading
import os
import time
from datetime import datetime

from camera import Camera
from face_tracker import FaceTracker
from voice_detector import VoiceDetector
from monitor import WindowMonitor
from object_detector import ObjectDetector
from violation_manager import ViolationManager
from report_generator import ReportGenerator


EXAM_DURATION = 120


camera = Camera()
face_tracker = FaceTracker()
violation_manager = ViolationManager()
report_generator = ReportGenerator()
voice_detector = VoiceDetector(
    violation_manager
)

window_monitor = WindowMonitor(
    exam_window_name="ProctorVision AI",
    violation_manager=violation_manager
)

object_detector = ObjectDetector()


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

book_frames = 0
book_missed_frames = 0

missing_threshold = 15
multiple_threshold = 10
gaze_threshold = 90

book_threshold = 1
book_missed_threshold = 20

current_warning = "CLEAN"
previous_gaze_status = "CENTER"

book_warning_active = False

phone_frames = 0
phone_threshold = 5
phone_violation_logged = False
phone_missed_frames = 0
phone_missed_threshold = 20
best_phone_confidence = 0.0
best_phone_frame = None
best_phone_box = None

book_violation_logged = False

no_face_violation_logged = False
multiple_faces_violation_logged = False
gaze_violation_logged = False
logged_gaze_status = "CENTER"


evidence_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "evidence",
    "object_detection"
)

os.makedirs(
    evidence_dir,
    exist_ok=True
)


face_evidence_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "evidence",
    "face_detection"
)

os.makedirs(
    face_evidence_dir,
    exist_ok=True
)


exam_active = False
exam_finished = False
exam_start_time = None


while True:

    frame = camera.read()

    if frame is None:
        print("Camera frame could not be read.")
        break

    height, width = frame.shape[:2]

    if not exam_active and not exam_finished:

        cv2.putText(
            frame,
            "EXAM: NOT STARTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press S to Start Exam",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            violation_manager.start_session()

            voice_detector.set_monitoring_active(
                True
            )

            exam_active = True
            exam_finished = False
            exam_start_time = time.time()

            missing_face_frames = 0
            multiple_face_frames = 0
            gaze_frames = 0

            book_frames = 0
            book_missed_frames = 0

            current_warning = "CLEAN"
            previous_gaze_status = "CENTER"

            book_warning_active = False

            phone_frames = 0
            phone_violation_logged = False
            phone_missed_frames = 0
            best_phone_confidence = 0.0
            best_phone_frame = None
            best_phone_box = None

            book_violation_logged = False

            no_face_violation_logged = False
            multiple_faces_violation_logged = False
            gaze_violation_logged = False
            logged_gaze_status = "CENTER"

            print()
            print("Exam monitoring started.")
            print(
                f"Exam duration: "
                f"{EXAM_DURATION} seconds"
            )

        elif key == ord("q"):

            break

        continue


    if exam_finished:

        cv2.putText(
            frame,
            "EXAM COMPLETED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Monitoring stopped",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        continue


    elapsed_time = (
        time.time()
        - exam_start_time
    )

    remaining_time = max(
        0,
        EXAM_DURATION - elapsed_time
    )


    if elapsed_time >= EXAM_DURATION:

        print()
        print("=" * 60)
        print("EXAM TIME COMPLETED")
        print("=" * 60)

        voice_detector.set_monitoring_active(
            False
        )

        window_monitor.finalize()

        session_id = violation_manager.end_session()

        report_generator.generate(
            session_id
        )

        exam_active = False
        exam_finished = True

        print("=" * 60)
        print()

        continue


    object_detections = object_detector.detect(
        frame
    )


    phone_detected = any(
        detection["class"] == "cell phone"
        for detection in object_detections
    )

    book_detected = any(
        detection["class"] == "book"
        for detection in object_detections
    )


    phone_confidence = max(
        (
            detection["confidence"]
            for detection in object_detections
            if detection["class"] == "cell phone"
        ),
        default=0.0
    )


    book_confidence = max(
        (
            detection["confidence"]
            for detection in object_detections
            if detection["class"] == "book"
        ),
        default=0.0
    )


    if book_detected:

        book_frames += 1
        book_missed_frames = 0

        if book_frames >= book_threshold:
            book_warning_active = True

    else:

        book_frames = 0

        if book_warning_active:

            book_missed_frames += 1

            if (
                book_missed_frames
                >= book_missed_threshold
            ):

                book_warning_active = False
                book_missed_frames = 0
                book_violation_logged = False


    if phone_detected:

        phone_frames += 1
        phone_missed_frames = 0

        current_phone = max(
            (
                detection
                for detection in object_detections
                if detection["class"] == "cell phone"
            ),
            key=lambda detection: detection["confidence"]
        )

        if (
            current_phone["confidence"]
            > best_phone_confidence
        ):

            best_phone_confidence = (
                current_phone["confidence"]
            )

            best_phone_box = current_phone["box"]
            best_phone_frame = frame.copy()

    else:

        phone_frames = 0

        if phone_violation_logged:

            phone_missed_frames += 1

            if (
                phone_missed_frames
                >= phone_missed_threshold
            ):

                phone_violation_logged = False
                phone_missed_frames = 0
                best_phone_confidence = 0.0
                best_phone_frame = None
                best_phone_box = None


    phone_confirmed = (
        phone_frames >= phone_threshold
    )


    (
        frame,
        status,
        face_count,
        gaze_status,
        results
    ) = face_tracker.process_frame(
        frame
    )


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

    cv2.putText(
        frame,
        f"Time Remaining: {int(remaining_time)}s",
        (20, 150),
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
        (20, 185),
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
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        voice_color,
        2
    )


    cv2.putText(
        frame,
        f"Voice Probability: {voice_probability:.2f}",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    if voice_detector.speech_active:

        cv2.putText(
            frame,
            f"Voice Duration: {voice_duration:.1f}s",
            (20, 290),
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


    if status == "NOT DETECTED":

        multiple_faces_violation_logged = False
        gaze_violation_logged = False
        logged_gaze_status = "CENTER"

        if (
            missing_face_frames >= missing_threshold
            and not no_face_violation_logged
        ):

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            evidence_path = os.path.join(
                face_evidence_dir,
                f"no_face_{timestamp}.jpg"
            )

            cv2.imwrite(
                evidence_path,
                frame
            )

            violation_manager.report_violation(
                violation_type="NO_FACE",
                duration=0.0,
                confidence=1.0,
                severity="HIGH",
                evidence_path=evidence_path
            )

            no_face_violation_logged = True

    else:

        no_face_violation_logged = False


    if face_count > 1:

        no_face_violation_logged = False
        gaze_violation_logged = False
        logged_gaze_status = "CENTER"

        if (
            multiple_face_frames >= multiple_threshold
            and not multiple_faces_violation_logged
        ):

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            evidence_path = os.path.join(
                face_evidence_dir,
                f"multiple_faces_{timestamp}.jpg"
            )

            cv2.imwrite(
                evidence_path,
                frame
            )

            violation_manager.report_violation(
                violation_type="MULTIPLE_FACES",
                duration=0.0,
                confidence=1.0,
                severity="HIGH",
                evidence_path=evidence_path
            )

            multiple_faces_violation_logged = True

    else:

        multiple_faces_violation_logged = False


    if (
        status == "DETECTED"
        and face_count == 1
        and gaze_status != "CENTER"
    ):

        if (
            gaze_frames >= gaze_threshold
            and (
                not gaze_violation_logged
                or logged_gaze_status != gaze_status
            )
        ):

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            evidence_path = os.path.join(
                face_evidence_dir,
                f"gaze_{timestamp}.jpg"
            )

            cv2.imwrite(
                evidence_path,
                frame
            )

            violation_manager.report_violation(
                violation_type="GAZE_VIOLATION",
                duration=0.0,
                confidence=1.0,
                severity="MEDIUM",
                evidence_path=evidence_path
            )

            gaze_violation_logged = True
            logged_gaze_status = gaze_status

    else:

        gaze_violation_logged = False
        logged_gaze_status = "CENTER"


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


    if phone_confirmed:

        warnings.append(
            "WARNING: Phone Detected!"
        )


    if book_warning_active:

        warnings.append(
            "WARNING: Book Detected!"
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


    if phone_confirmed and not phone_violation_logged:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        evidence_path = os.path.join(
            evidence_dir,
            f"phone_{timestamp}.jpg"
        )

        evidence_frame = (
            best_phone_frame.copy()
            if best_phone_frame is not None
            else frame.copy()
        )


        if best_phone_box is not None:

            x1, y1, x2, y2 = best_phone_box

            cv2.rectangle(
                evidence_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                evidence_frame,
                f"PHONE {best_phone_confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        cv2.imwrite(
            evidence_path,
            evidence_frame
        )


        phone_confidence = best_phone_confidence


        if phone_confidence >= 0.70:

            severity = "HIGH"

        else:

            severity = "MEDIUM"


        violation_manager.report_violation(
            violation_type="PHONE",
            duration=0.0,
            confidence=phone_confidence,
            severity=severity,
            evidence_path=evidence_path
        )

        phone_violation_logged = True


    if book_warning_active and not book_violation_logged:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        evidence_path = os.path.join(
            evidence_dir,
            f"book_{timestamp}.jpg"
        )

        cv2.imwrite(
            evidence_path,
            frame
        )


        if book_confidence >= 0.80:

            severity = "HIGH"

        elif book_confidence >= 0.50:

            severity = "MEDIUM"

        else:

            severity = "LOW"


        violation_manager.report_violation(
            violation_type="BOOK",
            duration=0.0,
            confidence=book_confidence,
            severity=severity,
            evidence_path=evidence_path
        )

        book_violation_logged = True


    cv2.imshow(
        window_name,
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("e"):

        voice_detector.set_monitoring_active(
            False
        )

        window_monitor.finalize()

        session_id = violation_manager.end_session()

        report_generator.generate(
            session_id
        )

        exam_active = False
        exam_finished = True

        print()
        print("EXAM ENDED BY USER")
        print()


    elif key == ord("q"):

        if exam_active:

            voice_detector.set_monitoring_active(
                False
            )

            window_monitor.finalize()

            session_id = violation_manager.end_session()

            report_generator.generate(
                session_id
            )

            exam_active = False
            exam_finished = True

        break


camera.release()
cv2.destroyAllWindows()