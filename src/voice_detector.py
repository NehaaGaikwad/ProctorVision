import sounddevice as sd
import numpy as np
import torch
import time
import os
import wave
from collections import deque
from datetime import datetime

from silero_vad import load_silero_vad
from violation_manager import ViolationManager


class VoiceDetector:

    def __init__(self):

        self.sample_rate = 16000
        self.device = 1
        self.chunk_size = 512

        self.model = load_silero_vad()

        self.speech_threshold = 0.25

        self.violation_duration = 3.0
        self.silence_duration = 0.5

        self.pre_buffer_seconds = 2.0
        self.post_buffer_seconds = 1.0

        self.pre_buffer_chunks = int(
            self.pre_buffer_seconds *
            self.sample_rate /
            self.chunk_size
        )

        self.post_buffer_chunks = int(
            self.post_buffer_seconds *
            self.sample_rate /
            self.chunk_size
        )

        self.rolling_buffer = deque(
            maxlen=self.pre_buffer_chunks
        )

        self.speech_buffer = []
        self.post_speech_buffer = []

        self.speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None

        self.violation_triggered = False

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.evidence_dir = os.path.join(
            project_root,
            "evidence"
        )

        os.makedirs(
            self.evidence_dir,
            exist_ok=True
        )

        self.violation_manager = (
            ViolationManager()
        )

        print("Voice Detector initialized.")
        print(
            "Microphone: "
            "Microphone Array "
            "(Intel Smart Sound Technology)"
        )
        print(
            f"Evidence storage: "
            f"{self.evidence_dir}"
        )
        print(
            f"Database: "
            f"{self.violation_manager.database}"
        )


    def get_speech_probability(self, audio):

        audio_tensor = torch.from_numpy(
            audio
        )

        try:

            probability = self.model(
                audio_tensor,
                self.sample_rate
            ).item()

            return probability

        except Exception as e:

            print(
                "VAD error:",
                e
            )

            return 0.0


    def start_speech(self):

        self.speech_active = True

        self.speech_start_time = (
            time.time()
        )

        self.last_speech_time = (
            time.time()
        )

        self.speech_buffer = []

        self.post_speech_buffer = []

        self.violation_triggered = False

        print()
        print("SPEECH STARTED")


    def calculate_confidence(self):

        if not self.speech_buffer:
            return 0.0

        probabilities = []

        for audio in self.speech_buffer:

            probability = (
                self.get_speech_probability(
                    audio
                )
            )

            probabilities.append(
                probability
            )

        if not probabilities:
            return 0.0

        return float(
            np.mean(probabilities)
        )


    def calculate_severity(
        self,
        duration,
        confidence
    ):

        if (
            duration >= 8.0
            and confidence >= 0.80
        ):

            return "HIGH"

        if (
            duration >= 5.0
            and confidence >= 0.60
        ):

            return "MEDIUM"

        return "LOW"


    def save_evidence(self, duration):

        if not self.speech_buffer:
            return None

        audio_parts = list(
            self.rolling_buffer
        )

        audio_parts.extend(
            self.speech_buffer
        )

        audio_parts.extend(
            self.post_speech_buffer
        )

        if not audio_parts:
            return None

        audio = np.concatenate(
            audio_parts
        )

        audio_int16 = np.int16(
            np.clip(
                audio,
                -1.0,
                1.0
            ) * 32767
        )

        timestamp = datetime.now()

        filename = (
            "voice_"
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
            ".wav"
        )

        filepath = os.path.join(
            self.evidence_dir,
            filename
        )

        with wave.open(
            filepath,
            "wb"
        ) as wav_file:

            wav_file.setnchannels(1)

            wav_file.setsampwidth(2)

            wav_file.setframerate(
                self.sample_rate
            )

            wav_file.writeframes(
                audio_int16.tobytes()
            )

        confidence = (
            self.calculate_confidence()
        )

        severity = (
            self.calculate_severity(
                duration,
                confidence
            )
        )

        self.violation_manager.report_violation(
            violation_type="VOICE",
            duration=duration,
            confidence=confidence,
            severity=severity,
            evidence_path=filepath
        )

        return filepath


    def finish_speech(self):

        if not self.speech_active:
            return

        duration = (
            time.time()
            - self.speech_start_time
        )

        if self.violation_triggered:

            filepath = (
                self.save_evidence(
                    duration
                )
            )

            if filepath:

                print()
                print(
                    f"Evidence saved: "
                    f"{filepath}"
                )

        else:

            print(
                f"Speech ended normally "
                f"({duration:.2f}s) - discarded."
            )

        self.speech_active = False

        self.speech_start_time = None

        self.last_speech_time = None

        self.speech_buffer = []

        self.post_speech_buffer = []

        self.violation_triggered = False


    def process_audio(self, audio):

        probability = (
            self.get_speech_probability(
                audio
            )
        )

        current_time = time.time()

        self.rolling_buffer.append(
            audio.copy()
        )

        if (
            probability
            >= self.speech_threshold
        ):

            if not self.speech_active:

                self.start_speech()

            self.speech_buffer.append(
                audio.copy()
            )

            self.last_speech_time = (
                current_time
            )

            duration = (
                current_time
                - self.speech_start_time
            )

            if (
                duration
                >= self.violation_duration
            ):

                if not self.violation_triggered:

                    self.violation_triggered = True

                    print()
                    print(
                        "SUSPICIOUS VOICE ACTIVITY"
                    )

                    print(
                        "Duration threshold "
                        f"reached: {duration:.2f}s"
                    )

            return probability, duration

        else:

            if self.speech_active:

                self.speech_buffer.append(
                    audio.copy()
                )

                silence_time = (
                    current_time
                    - self.last_speech_time
                )

                duration = (
                    current_time
                    - self.speech_start_time
                )

                if (
                    silence_time
                    < self.silence_duration
                ):

                    return (
                        probability,
                        duration
                    )

                if self.violation_triggered:

                    self.post_speech_buffer.append(
                        audio.copy()
                    )

                    if (
                        len(
                            self.post_speech_buffer
                        )
                        >= self.post_buffer_chunks
                    ):

                        self.finish_speech()

                else:

                    self.finish_speech()

            return probability, 0.0


    def start(self):

        print()
        print(
            "Voice monitoring started."
        )

        print(
            f"Speech longer than "
            f"{self.violation_duration:.1f}s "
            "will be treated as suspicious."
        )

        print(
            "Normal speech is kept "
            "temporarily in RAM."
        )

        print(
            "Only suspicious speech "
            "is saved."
        )

        print(
            "Press CTRL+C to stop."
        )

        print()

        try:

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=self.chunk_size,
                callback=self._callback
            ):

                while True:

                    sd.sleep(1000)

        except KeyboardInterrupt:

            print(
                "\nVoice monitoring stopped."
            )

            if self.speech_active:

                if self.violation_triggered:

                    duration = (
                        time.time()
                        - self.speech_start_time
                    )

                    self.save_evidence(
                        duration
                    )

                else:

                    print(
                        "Remaining speech "
                        "discarded."
                    )

        except Exception as e:

            print(
                "Microphone error:",
                e
            )


    def _callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status:

            print(
                "Audio status:",
                status
            )

        audio = (
            indata[:, 0].copy()
        )

        probability, duration = (
            self.process_audio(
                audio
            )
        )

        amplitude = np.max(
            np.abs(audio)
        )

        if self.violation_triggered:

            print(
                f"VIOLATION | "
                f"Amplitude: {amplitude:.4f} | "
                f"Probability: {probability:.4f} | "
                f"Duration: {duration:.2f}s"
            )

        elif self.speech_active:

            print(
                f"SPEECH | "
                f"Amplitude: {amplitude:.4f} | "
                f"Probability: {probability:.4f} | "
                f"Duration: {duration:.2f}s"
            )

        else:

            print(
                f"CLEAN | "
                f"Amplitude: {amplitude:.4f} | "
                f"Probability: {probability:.4f}"
            )


if __name__ == "__main__":

    detector = VoiceDetector()
    detector.start()