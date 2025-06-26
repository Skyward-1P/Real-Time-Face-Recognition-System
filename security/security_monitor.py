import cv2
import numpy as np
import datetime
import logging

class SecurityMonitor:
    def __init__(self):
        self.known_faces_db = {}
        self.unauthorized_attempts = []
        self.alert_threshold = 3
        self.logger = logging.getLogger('SecurityMonitor')
        
    def monitor_access(self, face_encoding, timestamp):
        """Monitor and log access attempts"""
        try:
            if not self.is_authorized(face_encoding):
                self.unauthorized_attempts.append(timestamp)
                if len(self.unauthorized_attempts) >= self.alert_threshold:
                    return "Security Alert: Unauthorized Access Attempt"
            return None
        except Exception as e:
            self.logger.error(f"Security monitoring error: {str(e)}")
            return None
            
    def is_authorized(self, face_encoding):
        """Check if face is in authorized database"""
        try:
            for known_encoding in self.known_faces_db.values():
                if np.linalg.norm(face_encoding - known_encoding) < 0.6:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Authorization check error: {str(e)}")
            return False 