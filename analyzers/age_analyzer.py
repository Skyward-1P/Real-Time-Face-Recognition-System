class AgeAnalyzer:
    def analyze(self, age):
        """Process age prediction with improved accuracy"""
        try:
            age = float(age)
            # More accurate age adjustment based on DeepFace's tendencies
            if age > 30:
                age = age * 0.6  # DeepFace tends to overestimate older ages significantly
            elif age > 25:
                age = age * 0.7  # Moderate overestimation for mid-range
            elif age > 20:
                age = age * 0.8  # Slight overestimation for young adults
            elif 15 <= age <= 20:
                age = age * 0.95  # Generally accurate for teens
            
            return int(round(age))
        except Exception as e:
            raise Exception(f"Age analysis error: {str(e)}") 