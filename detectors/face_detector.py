import cv2
import numpy as np

class FaceDetector:
    def __init__(self, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['face_detection_scale'],
                minNeighbors=self.config['face_detection_neighbors'],
                minSize=self.config['min_face_size']
            )
            return faces
        except Exception as e:
            raise Exception(f"Face detection error: {str(e)}")

    def preprocess_face(self, frame, x, y, w, h):
        """Preprocess detected face region"""
        try:
            face_region = frame[y:y+h, x:x+w]
            if w < 60 or h < 60:
                return None

            # Enhanced preprocessing
            face_region = cv2.resize(face_region, (224, 224))
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            face_region = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return face_region
        except Exception as e:
            raise Exception(f"Face preprocessing error: {str(e)}") 