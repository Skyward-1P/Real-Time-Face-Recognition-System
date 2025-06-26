import numpy as np
import logging

class AttentionAnalyzer:
    def __init__(self):
        self.attention_window = 60  # seconds
        self.gaze_history = []
        self.logger = logging.getLogger('AttentionAnalyzer')
        
    def analyze_attention(self, face_orientation, gaze_direction):
        """Analyze user attention patterns"""
        try:
            attention_score = self.calculate_attention_score(face_orientation, gaze_direction)
            return attention_score
        except Exception as e:
            self.logger.error(f"[attention_analyzer.py] Attention analysis error: {str(e)}")
            return None
            
    def calculate_attention_score(self, face_orientation, gaze_direction):
        """Calculate attention score based on face and gaze"""
        try:
            face_score = 1.0 - min(abs(face_orientation) / 45.0, 1.0)
            gaze_score = 1.0 - min(abs(gaze_direction) / 30.0, 1.0)
            return (face_score + gaze_score) / 2.0
        except Exception as e:
            self.logger.error(f"Score calculation error: {str(e)}")
            return 0.0 