import logging

class VoiceController:
    def __init__(self):
        self.logger = logging.getLogger('VoiceController')
        self.commands = {
            'start': self.start_analysis,
            'stop': self.stop_analysis,
            'report': self.generate_report,
            'toggle_stats': self.toggle_stats,
            'capture': self.capture_frame,
            'reset': self.reset_analysis
        }
        
        # Try to initialize speech recognition
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.sr_module = sr
            self.is_available = True
        except ImportError:
            self.logger.warning("Speech recognition module not available. Voice commands disabled.")
            self.is_available = False
        
    def process_command(self, audio_input):
        """Process voice commands"""
        if not self.is_available:
            return None
            
        try:
            command = self.recognize_command(audio_input)
            if command in self.commands:
                return self.execute_command(command)
            return None
        except Exception as e:
            self.logger.error(f"Voice command error: {str(e)}")
            return None
            
    def recognize_command(self, audio_input):
        """Convert audio to text command"""
        if not self.is_available:
            return None
            
        try:
            with self.sr_module.Microphone() as source:
                audio = self.recognizer.listen(source)
                return self.recognizer.recognize_google(audio)
        except Exception as e:
            self.logger.error(f"Speech recognition error: {str(e)}")
            return None
            
    def execute_command(self, command):
        """Execute recognized command"""
        try:
            return self.commands[command]()
        except Exception as e:
            self.logger.error(f"Command execution error: {str(e)}")
            return None
            
    def start_analysis(self):
        """Start continuous analysis"""
        self.analyzer.processing = True
        return "Analysis started"
        
    def stop_analysis(self):
        """Stop continuous analysis"""
        self.analyzer.processing = False
        return "Analysis stopped"
        
    def generate_report(self):
        """Generate analysis report"""
        if self.analyzer.averaged_analysis:
            return "Report generated"
        return "No analysis available"
        
    def toggle_stats(self):
        """Toggle statistics display"""
        self.analyzer.toggle_stats()
        return "Statistics toggled"
        
    def capture_frame(self):
        """Capture current frame"""
        self.analyzer.save_detection(self.analyzer.last_frame, 
                                   self.analyzer.current_analysis)
        return "Frame captured"
        
    def reset_analysis(self):
        """Reset current analysis"""
        self.analyzer.detection_complete = False
        self.analyzer.face_detections = []
        return "Analysis reset" 