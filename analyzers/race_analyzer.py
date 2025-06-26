class RaceAnalyzer:
    def analyze(self, race_predictions):
        """Process race predictions"""
        try:
            if not race_predictions:
                return {'Unknown': 100.0}
                
            # Filter out low confidence predictions
            confidence_threshold = 0.15
            race_predictions = {
                k: v for k, v in race_predictions.items() 
                if float(v) > confidence_threshold
            }
            
            # Normalize remaining probabilities
            total = sum(race_predictions.values())
            if total > 0:
                race_predictions = {
                    k: (v/total) * 100 
                    for k, v in race_predictions.items()
                }
            
            return race_predictions
        except Exception as e:
            raise Exception(f"Race analysis error: {str(e)}") 