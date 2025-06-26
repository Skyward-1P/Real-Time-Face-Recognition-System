import numpy as np
import logging
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.gesture_history = []
        self.logger = logging.getLogger('GestureRecognizer')
        self.gesture_patterns = {
            'nod': self.handle_nod,
            'shake': self.handle_shake,
            'wave': self.handle_wave,
            'tilt': self.handle_tilt,
            'blink': self.handle_blink
        }
        self.landmark_history = deque(maxlen=30)
        
    def detect_gesture(self, landmarks):
        """Detect gestures from facial landmarks"""
        try:
            if gesture := self.identify_gesture(landmarks):
                return self.gesture_patterns[gesture]()
            return None
        except Exception as e:
            self.logger.error(f"Gesture detection error: {str(e)}")
            return None
            
    def identify_gesture(self, landmarks):
        """Identify specific gesture from landmarks"""
        try:
            if not landmarks:
                return None
                
            # Add current landmarks to history
            self.landmark_history.append(landmarks)
            
            if len(self.landmark_history) < 10:
                return None
                
            # Calculate movement patterns
            head_movement = self.calculate_head_movement()
            eye_movement = self.calculate_eye_movement()
            
            # Identify gestures
            if self.is_nodding(head_movement):
                return 'nod'
            elif self.is_shaking(head_movement):
                return 'shake'
            elif self.is_waving(landmarks):
                return 'wave'
            elif self.is_tilting(head_movement):
                return 'tilt'
            elif self.is_blinking(eye_movement):
                return 'blink'
                
            return None
            
        except Exception as e:
            self.logger.error(f"Gesture identification error: {str(e)}")
            return None
            
    def calculate_head_movement(self):
        """Calculate head movement pattern"""
        movements = []
        for i in range(1, len(self.landmark_history)):
            prev = self.landmark_history[i-1]
            curr = self.landmark_history[i]
            dx = curr[33].x - prev[33].x
            dy = curr[33].y - prev[33].y
            movements.append((dx, dy))
        return movements
        
    def calculate_eye_movement(self):
        """Calculate eye movement pattern"""
        movements = []
        for landmarks in self.landmark_history:
            left_eye = landmarks[159]  # Left eye center
            right_eye = landmarks[386]  # Right eye center
            movements.append((left_eye, right_eye))
        return movements
        
    def is_nodding(self, movements):
        """Detect nodding gesture"""
        if not movements:
            return False
        vertical_movement = [m[1] for m in movements]
        return self.is_oscillating(vertical_movement, threshold=0.02)
        
    def is_shaking(self, movements):
        """Detect head shake gesture"""
        if not movements:
            return False
        horizontal_movement = [m[0] for m in movements]
        return self.is_oscillating(horizontal_movement, threshold=0.02)
        
    def is_oscillating(self, values, threshold):
        """Check if movement pattern is oscillating"""
        if len(values) < 6:
            return False
        crossings = 0
        for i in range(1, len(values)):
            if abs(values[i]) > threshold and values[i] * values[i-1] < 0:
                crossings += 1
        return crossings >= 3
        
    def handle_nod(self):
        return "Nod detected"
        
    def handle_shake(self):
        return "Head shake detected"
        
    def handle_wave(self):
        return "Wave detected"
        
    def handle_tilt(self):
        return "Tilt detected"
        
    def handle_blink(self):
        return "Blink detected" 