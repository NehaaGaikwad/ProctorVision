import cv2
import time


class Camera:
    def __init__(self, camera_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.prev_time = time.time()

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def calculate_fps(self):
        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        return fps

    def release(self):
        self.cap.release()