import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceTracker:
    def __init__(self, max_faces=2):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'face_landmarker.task')
        model_path = model_path.replace('\\', '/')
        
        # Check karte hain ki file sach mein exist karti hai ya nahi
        if os.path.exists(model_path):
            print(f"SUCCESS: Model file found at -> {model_path}")
        else:
            print(f"ERROR: Model file NOT found at -> {model_path}")
            print("Check karo ki file ka naam kahin 'face_landmarker.task.txt' toh nahi ho gaya hai!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        status = "NOT DETECTED"
        face_count = 0
        
        if results.face_landmarks:
            face_count = len(results.face_landmarks)
            status = "DETECTED"
            
            for face_landmarks in results.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
        return frame, status, face_count, results