from collections import deque
import numpy as np
import logging

class HealthMonitor:
    def __init__(self):
        self.blink_rate_history = deque(maxlen=50)
        self.fatigue_threshold = 0.3
        self.logger = logging.getLogger('HealthMonitor')
        
    def monitor_fatigue(self, eye_aspect_ratio, blink_count):
        """Monitor user fatigue levels"""
        try:
            # Store blink rate instead of raw count
            blink_rate = blink_count / 60.0  # Blinks per second
            self.blink_rate_history.append(blink_rate)
            
            if eye_aspect_ratio < self.fatigue_threshold:
                if self.check_sustained_fatigue():
                    return "Take a break alert"
            return None
        except Exception as e:
            self.logger.error(f"[health_monitor.py] Fatigue monitoring error: {str(e)}")
            return None
            
    def check_sustained_fatigue(self):
        """Check if fatigue is sustained"""
        if len(self.blink_rate_history) < 10:
            return False
        return np.mean(list(self.blink_rate_history)) > 0.5 