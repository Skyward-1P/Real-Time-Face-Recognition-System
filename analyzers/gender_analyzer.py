import cv2
import numpy as np

class GenderAnalyzer:
    def analyze(self, face_mesh, landmarks, analysis, w, h):
        """Enhanced multi-model gender detection"""
        try:
            # Get initial DeepFace prediction
            gender_str = str(analysis.get('gender', '')).strip().lower()
            
            # Calculate comprehensive facial metrics
            cheekbone_width = abs(landmarks.landmark[123].x - landmarks.landmark[352].x)
            jaw_width = abs(landmarks.landmark[172].x - landmarks.landmark[397].x)
            face_height = abs(landmarks.landmark[10].y - landmarks.landmark[152].y)
            brow_height = abs(landmarks.landmark[282].y - landmarks.landmark[295].y)
            chin_height = abs(landmarks.landmark[152].y - landmarks.landmark[10].y)
            
            # Calculate advanced ratios
            jaw_cheek_ratio = jaw_width / cheekbone_width if cheekbone_width > 0 else 1
            face_width_height = w / h
            brow_face_ratio = brow_height / face_height if face_height > 0 else 0
            chin_face_ratio = chin_height / face_height if face_height > 0 else 0
            
            # Weighted feature scoring
            male_score = 0
            female_score = 0
            
            # Facial structure scoring
            if jaw_cheek_ratio > 0.88:  # Square jaw
                male_score += 1
            else:
                female_score += 1
                
            if face_width_height > 0.80:  # Wider face
                male_score += 1
            else:
                female_score += 1
                
            if brow_face_ratio > 0.04:  # Thicker brows
                male_score += 1
            else:
                female_score += 1
                
            if chin_face_ratio > 0.33:  # Longer chin
                male_score += 1
            else:
                female_score += 1
            
            # Give more weight to DeepFace prediction
            if 'man' in gender_str or 'male' in gender_str:
                male_score += 3  # Increased weight for DeepFace
            elif 'woman' in gender_str or 'female' in gender_str:
                female_score += 3  # Increased weight for DeepFace
            
            # Make final determination
            if abs(male_score - female_score) <= 1:  # Close call
                # Use DeepFace's prediction as tiebreaker
                return 'Male' if 'man' in gender_str or 'male' in gender_str else 'Female'
            else:
                return 'Male' if male_score > female_score else 'Female'
                
        except Exception as e:
            raise Exception(f"Gender analysis error: {str(e)}") 