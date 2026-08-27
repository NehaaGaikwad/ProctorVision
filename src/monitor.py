import os
import time
import ctypes
from datetime import datetime
from PIL import ImageGrab

from violation_manager import ViolationManager


class WindowMonitor:

    def __init__(self, exam_window_name="ProctorVision AI"):

        self.exam_window_name = exam_window_name

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.evidence_dir = os.path.join(
            project_root,
            "evidence",
            "window_switch"
        )

        os.makedirs(
            self.evidence_dir,
            exist_ok=True
        )

        self.violation_manager = ViolationManager()

        self.switch_active = False
        self.switch_start_time = None
        self.current_window = None
        self.evidence_path = None

        self.last_event_time = 0
        self.cooldown = 2.0

    def get_active_window_title(self):

        user32 = ctypes.windll.user32

        hwnd = user32.GetForegroundWindow()

        if not hwnd:
            return ""

        length = user32.GetWindowTextLengthW(hwnd)

        if length == 0:
            return ""

        buffer = ctypes.create_unicode_buffer(
            length + 1
        )

        user32.GetWindowTextW(
            hwnd,
            buffer,
            length + 1
        )

        return buffer.value.strip()

    def is_exam_window(self, title):

        if not title:
            return False

        return (
            self.exam_window_name.lower()
            in title.lower()
        )

    def save_screen_evidence(self, window_title):

        timestamp = datetime.now()

        safe_title = "".join(
            character
            if character.isalnum()
            else "_"
            for character in window_title
        )

        safe_title = safe_title[:50]

        filename = (
            "window_switch_"
            f"{safe_title}_"
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
            ".png"
        )

        filepath = os.path.join(
            self.evidence_dir,
            filename
        )

        try:

            screenshot = ImageGrab.grab(
                all_screens=True
            )

            screenshot.save(
                filepath
            )

            return filepath

        except Exception as e:

            print(
                "Screen evidence error:",
                e
            )

            return None

    def calculate_severity(self, duration):

        if duration >= 10:
            return "HIGH"

        if duration >= 5:
            return "MEDIUM"

        return "LOW"

    def check(self):

        active_window = (
            self.get_active_window_title()
        )

        if self.is_exam_window(
            active_window
        ):

            if self.switch_active:

                duration = (
                    time.time()
                    - self.switch_start_time
                )

                severity = (
                    self.calculate_severity(
                        duration
                    )
                )

                self.violation_manager.report_violation(
                    violation_type="WINDOW_SWITCH",
                    duration=duration,
                    confidence=1.0,
                    severity=severity,
                    evidence_path=self.evidence_path
                )

                print()
                print(
                    "WINDOW SWITCH ENDED"
                )

                print(
                    f"Previous Window : "
                    f"{self.current_window}"
                )

                print(
                    f"Duration        : "
                    f"{duration:.2f}s"
                )

                self.switch_active = False
                self.switch_start_time = None
                self.current_window = None
                self.evidence_path = None

            return {
                "active": False,
                "window": active_window,
                "duration": 0.0
            }

        if not active_window:

            return {
                "active": False,
                "window": "",
                "duration": 0.0
            }

        if not self.switch_active:

            current_time = time.time()

            if (
                current_time
                - self.last_event_time
                < self.cooldown
            ):

                return {
                    "active": False,
                    "window": active_window,
                    "duration": 0.0
                }

            self.switch_active = True

            self.switch_start_time = (
                current_time
            )

            self.current_window = (
                active_window
            )

            self.evidence_path = (
                self.save_screen_evidence(
                    active_window
                )
            )

            self.last_event_time = (
                current_time
            )

            print()
            print("=" * 60)
            print(
                "WINDOW / APPLICATION SWITCH DETECTED"
            )
            print("=" * 60)
            print(
                f"Application : {active_window}"
            )

            if self.evidence_path:

                print(
                    f"Evidence    : "
                    f"{self.evidence_path}"
                )

            print("=" * 60)
            print()

        duration = (
            time.time()
            - self.switch_start_time
        )

        return {
            "active": True,
            "window": self.current_window,
            "duration": duration
        }

    def finalize(self):

        if not self.switch_active:
            return

        duration = (
            time.time()
            - self.switch_start_time
        )

        severity = (
            self.calculate_severity(
                duration
            )
        )

        self.violation_manager.report_violation(
            violation_type="WINDOW_SWITCH",
            duration=duration,
            confidence=1.0,
            severity=severity,
            evidence_path=self.evidence_path
        )

        self.switch_active = False
        self.switch_start_time = None
        self.current_window = None
        self.evidence_path = None