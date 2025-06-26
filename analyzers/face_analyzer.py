from deepface import DeepFace
import mediapipe as mp
import cv2

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        
    def analyze_face(self, face_region):
        """Analyze face using DeepFace"""
        try:
            results = DeepFace.analyze(
                face_region,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=False,
                detector_backend='opencv',
                align=True,
                silent=True
            )
            return results
        except Exception as e:
            raise Exception(f"Face analysis error: {str(e)}")
            
    def get_face_mesh(self, face_region):
        """Get facial landmarks using MediaPipe"""
        try:
            return self.mp_face_mesh.process(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
        except Exception as e:
            raise Exception(f"Face mesh error: {str(e)}") 