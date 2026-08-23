import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceTracker:
    def __init__(self, max_faces=2):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'face_landmarker.task')
        model_path = model_path.replace('\\', '/')
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        status = "NOT DETECTED"
        face_count = 0
        gaze_status = "CENTER"
        
        if results.face_landmarks:
            face_count = len(results.face_landmarks)
            status = "DETECTED"
            
            for face_landmarks in results.face_landmarks:
                left_eye = face_landmarks[33]
                right_eye = face_landmarks[263]
                nose = face_landmarks[1]
                chin = face_landmarks[152]
                
                eye_mid_x = (left_eye.x + right_eye.x) / 2
                eye_mid_y = (left_eye.y + right_eye.y) / 2
                
                eye_distance = abs(right_eye.x - left_eye.x)
                
                x_offset = (nose.x - eye_mid_x) / eye_distance
                
                face_height = abs(chin.y - eye_mid_y)
                y_offset = (nose.y - eye_mid_y) / face_height
                
                if x_offset < -0.12:
                    gaze_status = "LOOKING RIGHT"
                elif x_offset > 0.12:
                    gaze_status = "LOOKING LEFT"
                elif y_offset < 0.28:
                    gaze_status = "LOOKING UP"
                elif y_offset > 0.42:
                    gaze_status = "LOOKING DOWN"
                else:
                    gaze_status = "CENTER"
                    
                text_x = w - 320
                cv2.putText(frame, f"Gaze: {gaze_status}", (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Faces: {face_count}", (text_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for landmark in face_landmarks:
                    lx = int(landmark.x * w)
                    ly = int(landmark.y * h)
                    cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)
                
        return frame, status, face_count, results