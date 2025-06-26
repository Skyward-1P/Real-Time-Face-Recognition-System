from collections import deque
import logging

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_history = deque(maxlen=100)
        self.logger = logging.getLogger('EmotionAnalyzer')
        
    def analyze_mood_patterns(self, emotion_data):
        """Track mood patterns over time"""
        try:
            self.emotion_history.append(emotion_data)
            if len(self.emotion_history) >= 10:
                return self.get_dominant_mood()
            return None
        except Exception as e:
            self.logger.error(f"[emotion_analyzer.py] Mood analysis error: {str(e)}")
            return None
            
    def get_dominant_mood(self):
        """Calculate dominant mood from history"""
        try:
            mood_counts = {}
            for mood in self.emotion_history:
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
            return max(mood_counts.items(), key=lambda x: x[1])[0]
        except Exception as e:
            self.logger.error(f"Dominant mood calculation error: {str(e)}")
            return None 