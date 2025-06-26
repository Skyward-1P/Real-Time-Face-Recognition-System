import cv2
import numpy as np
from deepface import DeepFace
import datetime
import os
import threading
import time
from collections import deque
import mediapipe as mp
import pygame
import json
from PIL import Image, ImageTk
import logging
import csv
from pathlib import Path
import face_recognition
import warnings
import platform
import psutil
import subprocess
import tkinter as tk
from analyzers.gender_analyzer import GenderAnalyzer
from analyzers.age_analyzer import AgeAnalyzer
from analyzers.race_analyzer import RaceAnalyzer
from detectors.face_detector import FaceDetector
from analyzers.face_analyzer import FaceAnalyzer as DeepFaceAnalyzer
from analyzers.emotion_analyzer import EmotionAnalyzer
from analyzers.health_monitor import HealthMonitor
from analyzers.attention_analyzer import AttentionAnalyzer
from security.security_monitor import SecurityMonitor
from controls.gesture_recognizer import GestureRecognizer
from controls.voice_controller import VoiceController
from analyzers.environment_analyzer import EnvironmentAnalyzer
from ui.face_analyzer_ui import FaceAnalyzerUI
from models.model_manager import ModelManager

class FaceAnalyzer:
    def __init__(self):
        # Initialize logging first
        self.output_dir = 'face_detection_output'
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_logging()
        
        # Initialize basic configuration
        self.config = {
            'save_detections': True,
            'detection_interval': 0.03,
            'blur_background': False,
            'show_landmarks': False,
            'enable_audio_alerts': False,
            'track_attention': True,
            'enable_statistics': True,
            'face_recognition_enabled': False,
            'enable_sound_effects': False,
            'frame_width': 640,
            'frame_height': 480,
            'target_fps': 15,    # Your CPU can handle this
            'use_threading': True,  # Enable for 4 cores
            'skip_frames': 2,    # Process every 2nd frame
            'batch_process': True,  # Enable for 19GB RAM
            'detection_confidence': 0.7,
            'buffer_size': 2,
            'process_every_n_frames': 2,
            'face_detection_scale': 1.1,
            'face_detection_neighbors': 4,
            'min_face_size': (60, 60),
            'temporal_smoothing': True,
            'smoothing_factor': 0.7,
            'enable_face_recognition': False,
            'enable_eye_tracking': True,
            'enable_head_pose': True,
            'enable_recording': False,
            'enable_voice_alerts': True,
            'enable_drowsiness_detection': True,
            'enable_mask_detection': True,
            'low_light_enhancement': False,
            'enable_age_verification': False,
            'minimum_age': 18,
            'enable_security_alerts': True,
            'sounds': {
                'drowsiness_alert': True,
                'security_alert': True,
                'age_verification_alert': True,
                'face_detection_alert': False,
                'recording_alert': False
            },
            'mood_tracking': True,
            'health_monitoring': True,
            'attention_tracking': True,
            'security_monitoring': True,
            'gesture_control': True,
            'voice_commands': True,
            'environment_analysis': True
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Now detect system capabilities and update config
        self.detect_system_capabilities()
        
        # Initialize components with adjusted parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pygame for audio alerts
        if self.config['enable_sound_effects']:
            pygame.mixer.init()
        
        # Initialize statistics with better defaults
        self.stats = {
            'total_detections': 0,
            'average_age': 0,
            'emotions_history': deque(maxlen=100),
            'attention_score': 100,
            'session_start': time.time(),
            'blink_count': 0,
            'drowsiness_alerts': 0,
            'mask_detections': 0,
            'security_alerts': 0,
            'recognized_faces': set(),
            'eye_aspect_ratios': deque(maxlen=50),
            'head_poses': deque(maxlen=50),
            'mood_patterns': [],
            'fatigue_alerts': 0,
            'attention_scores': deque(maxlen=100),
            'total_frames': 0,  # Add frame counter
            'blink_timestamps': deque(maxlen=100),  # For blink rate calculation
            'last_blink_check': time.time()  # For blink rate calculation
        }
        
        # Load known faces if recognition is enabled
        self.known_faces = {}
        if self.config['face_recognition_enabled']:
            self.load_known_faces()
        
        # Initialize video writer
        self.video_writer = None
        self.recording = False

        # Load alert sounds
        self.load_alert_sounds()

        # Adjust resolution based on system capabilities
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
        
        # Initialize threading components
        self.frame_queue = deque(maxlen=2)
        self.result_queue = deque(maxlen=2)
        self.processing = False
        self.last_frame = None
        self.frame_count = 0

        # Add these to existing initialization
        self.face_detections = []  # Store last N detections
        self.detection_complete = False  # Flag to indicate if detection is complete
        self.last_face_position = None  # Store last face position
        self.averaged_analysis = None  # Store averaged results
        self.face_change_threshold = 50  # Pixel threshold for new face detection
        self.required_detections = 5  # Number of detections to average

        # Initialize detectors and analyzers
        self.face_detector = FaceDetector(self.config)
        self.deep_face_analyzer = DeepFaceAnalyzer()
        self.gender_analyzer = GenderAnalyzer()
        self.age_analyzer = AgeAnalyzer()
        self.race_analyzer = RaceAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()
        self.health_monitor = HealthMonitor()
        self.attention_analyzer = AttentionAnalyzer()

        # Extend configuration
        self.config.update({
            'enable_mood_tracking': True,
            'enable_health_monitoring': True,
            'enable_attention_tracking': True,
            'fatigue_alert_threshold': 0.3,
            'attention_alert_threshold': 0.5
        })

        # Add new analyzers and controllers
        self.security_monitor = SecurityMonitor()
        self.gesture_recognizer = GestureRecognizer()
        self.voice_controller = VoiceController()
        self.environment_analyzer = EnvironmentAnalyzer()
        
        # Update configuration
        self.config.update({
            'enable_security_monitoring': True,
            'enable_gesture_control': True,
            'enable_voice_commands': True,
            'enable_environment_analysis': True
        })

        # Instead, add:
        self.ui = FaceAnalyzerUI(self)

        # Add after other initializations
        self.model_manager = ModelManager()

    def setup_logging(self):
        """Setup logging configuration"""
        try:
            logging.basicConfig(
                filename=f'{self.output_dir}/face_analyzer.log',
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            self.logger = logging.getLogger('FaceAnalyzer')
        except Exception as e:
            print(f"Failed to initialize logger: {str(e)}")
            self.logger = logging.getLogger('FaceAnalyzer')
            self.logger.addHandler(logging.NullHandler())

    def load_alert_sounds(self):
        """Load alert sound effects if enabled"""
        try:
            # Check if directory exists first
            if not os.path.exists('assets'):
                os.makedirs('assets')
                self.logger.warning("Created assets directory - sound files needed")
                self.config['enable_sound_effects'] = False
                self.sounds = {}
                return

            if self.config.get('enable_sound_effects', False):  # Use get() with default
                sound_files = {
                    'drowsiness': 'assets/drowsiness_alert.wav',
                    'security': 'assets/security_alert.wav',
                    'age_verification': 'assets/age_alert.wav',
                    'face_detection': 'assets/detection.wav',
                    'recording': 'assets/record.wav'
                }
                
                # Only load sounds that exist
                self.sounds = {}
                for sound_name, sound_path in sound_files.items():
                    if os.path.exists(sound_path):
                        try:
                            self.sounds[sound_name] = pygame.mixer.Sound(sound_path)
                        except:
                            self.logger.warning(f"Failed to load {sound_path}")
            else:
                self.sounds = {}
        except Exception as e:
            self.logger.error(f"Sound initialization error: {str(e)}")
            self.config['enable_sound_effects'] = False
            self.sounds = {}

    def load_known_faces(self):
        """Load known faces from the database"""
        if os.path.exists('known_faces.json'):
            with open('known_faces.json', 'r') as f:
                self.known_faces = json.load(f)

    def draw_text(self, img, text, x, y, color_bg=(0, 128, 0)):
        """Enhanced text drawing with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        cv2.rectangle(img, (x, y), (x + text_w, y - text_h - 5), color_bg, -1)
        cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, font_thickness)

    def apply_artistic_filters(self, frame):
        """Apply artistic filters to the frame"""
        filters = {
            '1': cv2.COLORMAP_VIRIDIS,
            '2': cv2.COLORMAP_PLASMA,
            '3': cv2.COLORMAP_HOT
        }
        key = cv2.waitKey(1) & 0xFF
        if chr(key) in filters:
            frame = cv2.applyColorMap(frame, filters[chr(key)])
        return frame

    def detect_facial_landmarks(self, frame):
        """Detect and draw facial landmarks using MediaPipe"""
        if self.config['show_landmarks']:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
        return frame

    def calculate_attention_score(self, face_angle):
        """Calculate attention score based on face angle"""
        if abs(face_angle) > 30:
            self.stats['attention_score'] = max(0, self.stats['attention_score'] - 5)
        else:
            self.stats['attention_score'] = min(100, self.stats['attention_score'] + 1)
        return self.stats['attention_score']

    def save_detection(self, frame, face_data):
        """Save detection data and image"""
        if self.config['save_detections']:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{self.output_dir}/detection_{timestamp}.jpg", frame)
            with open(f"{self.output_dir}/detection_{timestamp}.json", 'w') as f:
                json.dump(face_data, f)

    def apply_background_effects(self, frame, face_coords):
        """Apply background effects"""
        if self.config['blur_background']:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for (x, y, w, h) in face_coords:
                mask[y:y+h, x:x+w] = 255
            blurred = cv2.GaussianBlur(frame, (99, 99), 30)
            frame = np.where(mask[:,:,None] == 255, frame, blurred)
        return frame

    def draw_face_info(self, frame, x, y, analysis):
        """Enhanced information display with averaging status"""
        if analysis:
            y_offset = y - 10
            info_lines = [
                f"Age: {analysis.get('age', '?')}",
                f"Gender: {analysis.get('gender', '?')}",
                f"Mood: {analysis.get('dominant_emotion', '?')}",
                f"Race: {max(analysis.get('race', {}).items(), key=lambda x: x[1])[0]}"
            ]
            
            # Add detection status
            if self.detection_complete:
                info_lines.append("Detection Complete")
            else:
                info_lines.append(f"Analyzing... {len(self.face_detections)}/{self.required_detections}")
            
            for text in info_lines:
                y_offset -= 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                size = cv2.getTextSize(text, font, 0.7, 2)[0]
                
                # Enhanced text visibility
                cv2.rectangle(frame, 
                             (x - 2, y_offset - 20),
                             (x + size[0] + 5, y_offset + 5),
                             (0, 0, 0), -1)
                
                # Use green for complete detection, yellow for in-progress
                color = (0, 255, 0) if self.detection_complete else (0, 255, 255)
                cv2.putText(frame, text, (x, y_offset),
                           font, 0.7, color, 2)

    def analyze_face_details(self, frame, x, y, w, h, frame_count):
        """Enhanced face analysis with new features"""
        try:
            if self.detection_complete:
                if self.last_face_position:
                    old_x, old_y = self.last_face_position
                    if abs(x - old_x) < self.face_change_threshold and abs(y - old_y) < self.face_change_threshold:
                        return self.averaged_analysis
                    else:
                        self.detection_complete = False
                        self.face_detections = []
            
            # Get preprocessed face region
            face_region = self.face_detector.preprocess_face(frame, x, y, w, h)
            if face_region is None:
                return None

            # Get facial landmarks
            face_mesh = self.deep_face_analyzer.get_face_mesh(face_region)
            
            # Get main analysis results
            results = self.deep_face_analyzer.analyze_face(face_region)
            
            if results and isinstance(results, list) and len(results) > 0:
                analysis = results[0]
                self.current_analysis = analysis

                # Update total detections (already fixed)
                self.stats['total_detections'] += 1
                self.stats['total_frames'] += 1

                # Fix age calculation - simple running average
                if 'age' in analysis:
                    try:
                        age = float(analysis['age'])
                        if self.stats['total_detections'] == 1:
                            self.stats['average_age'] = age
                        else:
                            # Simple running average
                            self.stats['average_age'] = (
                                self.stats['average_age'] + 
                                (age - self.stats['average_age']) / self.stats['total_detections']
                            )
                    except Exception as e:
                        self.logger.error(f"Age calculation error: {str(e)}")

                # Process individual attributes
                analysis['age'] = self.age_analyzer.analyze(analysis.get('age', 0))
                
                if face_mesh.multi_face_landmarks:
                    analysis['gender'] = self.gender_analyzer.analyze(
                        face_mesh, 
                        face_mesh.multi_face_landmarks[0],
                        analysis,
                        w, h
                    )
                
                if 'race' in analysis:
                    analysis['race'] = self.race_analyzer.analyze(analysis['race'])
                
                # Remove probability from display
                if 'gender_probability' in analysis:
                    del analysis['gender_probability']
                
                # Create simplified copy with improved race handling
                simplified_analysis = {
                    'age': analysis['age'],
                    'gender': analysis['gender'],
                    'dominant_emotion': analysis.get('dominant_emotion', ''),
                    'race': {str(k): float(v) for k, v in analysis.get('race', {}).items()},
                    'emotion': {str(k): float(v) for k, v in analysis.get('emotion', {}).items()}
                }
                
                # Add to detections list
                self.face_detections.append(simplified_analysis)
                
                if len(self.face_detections) >= self.required_detections:
                    self.averaged_analysis = self.calculate_average_analysis(self.face_detections)
                    self.detection_complete = True
                    self.last_face_position = (x, y)
                    return self.averaged_analysis
                
                # Add new analysis features
                if self.config['enable_mood_tracking']:
                    mood_pattern = self.emotion_analyzer.analyze_mood_patterns(
                        analysis.get('dominant_emotion', '')
                    )
                    if mood_pattern:
                        analysis['mood_pattern'] = mood_pattern
                
                if self.config['enable_health_monitoring']:
                    # Calculate fatigue level based on blink rate and eye aspect ratio
                    blink_rate = self.stats.get('blink_count', 0) / 60.0  # blinks per second
                    ear = self.stats.get('eye_aspect_ratio', 0)
                    fatigue_level = (1 - ear/100) * 50 + (min(blink_rate, 1.0) * 50)
                    self.stats['fatigue_level'] = fatigue_level
                    
                    # Update drowsiness alerts if needed
                    if fatigue_level > 70:  # High fatigue threshold
                        self.stats['drowsiness_alerts'] = self.stats.get('drowsiness_alerts', 0) + 1
                
                if self.config['enable_attention_tracking']:
                    attention_score = self.attention_analyzer.analyze_attention(
                        analysis.get('face_angle', 0),
                        analysis.get('gaze_direction', 0)
                    )
                    if attention_score is not None:
                        self.stats['attention_scores'].append(attention_score)
                        analysis['attention_score'] = sum(self.stats['attention_scores']) / len(self.stats['attention_scores']) * 100
                
                # Update security status
                if self.config['security_monitoring']:
                    security_status = self.security_monitor.monitor_access(
                        analysis.get('face_encoding', None),
                        time.time()
                    )
                    if security_status:
                        self.stats['security_alerts'] = self.stats.get('security_alerts', 0) + 1
                
                # Update environment analysis
                if self.config['environment_analysis']:
                    # Calculate average brightness
                    brightness = np.mean(frame)
                    self.stats['light_level'] = int((brightness / 255) * 100)  # Convert to percentage
                    
                    # Calculate frame quality
                    frame_quality = cv2.Laplacian(frame, cv2.CV_64F).var()
                    self.stats['frame_quality'] = min(100, int((frame_quality / 500) * 100))  # Normalize to percentage

                # Use ModelManager for additional features
                face_landmarks = self.model_manager.get_face_landmarks(face_region)
                if face_landmarks and face_landmarks.multi_face_landmarks:
                    # Enhanced facial analysis
                    landmarks = face_landmarks.multi_face_landmarks[0]
                    
                    # Update existing analysis with landmark data
                    if analysis:
                        analysis['landmarks'] = {
                            'eye_aspect_ratio': self.calculate_ear(landmarks),
                            'mouth_aspect_ratio': self.calculate_mar(landmarks),
                            'face_orientation': self.calculate_face_orientation(landmarks)
                        }

                # Get face region for additional analysis
                face_region = frame[y:y+h, x:x+w]

                # Add gaze estimation
                if self.config['enable_eye_tracking']:
                    gaze_result = self.model_manager.estimate_gaze(face_region)
                    if gaze_result is not None:
                        analysis['gaze_direction'] = gaze_result

                # Add head pose estimation
                if self.config['enable_head_pose']:
                    head_pose = self.model_manager.estimate_head_pose(face_region)
                    if head_pose is not None:
                        analysis['head_pose'] = head_pose

                # Add face quality assessment
                face_quality = self.model_manager.assess_face_quality(face_region)
                if face_quality is not None:
                    analysis['face_quality'] = face_quality
                    self.stats['frame_quality'] = face_quality

                # Get face landmarks for blink detection
                face_landmarks = self.model_manager.get_face_landmarks(frame)
                if face_landmarks and face_landmarks.multi_face_landmarks:
                    try:
                        landmarks = face_landmarks.multi_face_landmarks[0]
                        ear = self.calculate_ear(landmarks)
                        
                        # Store EAR history
                        self.stats['eye_aspect_ratios'].append(ear)
                        
                        # Improved blink detection
                        current_time = time.time()
                        if len(self.stats['eye_aspect_ratios']) >= 3:
                            # Check for blink pattern: open -> closed -> open
                            if (self.stats['eye_aspect_ratios'][-3] > 0.3 and  # Eye open
                                self.stats['eye_aspect_ratios'][-2] < 0.21 and  # Eye closed
                                self.stats['eye_aspect_ratios'][-1] > 0.3):     # Eye open
                                
                                # Ensure minimum time between blinks
                                if not self.stats['blink_timestamps'] or \
                                   current_time - self.stats['blink_timestamps'][-1] > 0.2:
                                    self.stats['blink_count'] += 1
                                    self.stats['blink_timestamps'].append(current_time)
                        
                        # Update drowsiness if eyes stay closed
                        if ear < 0.25:  # Drowsiness threshold
                            self.stats['drowsiness_alerts'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Blink detection error: {str(e)}")

                return analysis
                
        except Exception as e:
            self.logger.error(f"[main.py] Analysis error: {str(e)}")
            return None

    def calculate_average_analysis(self, detections):
        """Calculate average of multiple detections"""
        try:
            avg_analysis = {}
            
            # Average age
            ages = [d.get('age', 0) for d in detections]
            avg_analysis['age'] = int(round(sum(ages) / len(ages)))
            
            # Most common gender with string handling
            genders = [str(d.get('gender', '')).strip() for d in detections]
            avg_analysis['gender'] = max(set(genders), key=genders.count)
            
            # Most common emotion with string handling
            emotions = [str(d.get('dominant_emotion', '')).strip() for d in detections]
            avg_analysis['dominant_emotion'] = max(set(emotions), key=emotions.count)
            
            # Average race probabilities
            race_probs = {}
            for d in detections:
                if isinstance(d.get('race'), dict):  # Check if race is a dict
                    for race, prob in d['race'].items():
                        race_probs[str(race)] = race_probs.get(str(race), 0) + float(prob)
            
            if race_probs:  # Only if we have race data
                avg_analysis['race'] = {
                    race: prob/len(detections) 
                    for race, prob in race_probs.items()
                }
            else:
                avg_analysis['race'] = {'Unknown': 100.0}
            
            # Add emotion probabilities if available
            emotion_probs = {}
            for d in detections:
                if isinstance(d.get('emotion'), dict):  # Check if emotion is a dict
                    for emotion, prob in d['emotion'].items():
                        emotion_probs[str(emotion)] = emotion_probs.get(str(emotion), 0) + float(prob)
            
            if emotion_probs:
                avg_analysis['emotion'] = {
                    emotion: prob/len(detections)
                    for emotion, prob in emotion_probs.items()
                }
            
            # Store confidence level
            avg_analysis['confidence'] = len(detections) / self.required_detections * 100
            
            return avg_analysis
            
        except Exception as e:
            self.logger.error(f"[main.py] Average calculation error: {str(e)}")
            return None

    def estimate_head_pose(self, face_region):
        """Estimate head pose using facial landmarks"""
        try:
            face_mesh = self.mp_face_mesh.process(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            if face_mesh.multi_face_landmarks:
                landmarks = face_mesh.multi_face_landmarks[0]
                # Calculate head pose angles
                # This is a simplified calculation
                nose_tip = landmarks.landmark[4]
                left_eye = landmarks.landmark[33]
                right_eye = landmarks.landmark[263]
                
                # Calculate angles
                yaw = (right_eye.x - left_eye.x) * 100  # Rough estimation
                pitch = nose_tip.y * 100
                
                return {
                    'yaw': yaw,
                    'pitch': pitch
                }
        except Exception as e:
            self.logger.error(f"[main.py] Head pose estimation error: {str(e)}")
            return None

    def smooth_predictions(self, current, previous):
        """Apply temporal smoothing to predictions"""
        smoothing_factor = 0.7  # Higher value = more smoothing
        
        # Smooth numerical values
        current['age'] = int(previous['age'] * smoothing_factor + 
                           current['age'] * (1 - smoothing_factor))
        
        # Keep high confidence predictions for categorical values
        if float(previous.get('gender_probability', 0)) > float(current.get('gender_probability', 0)):
            current['gender'] = previous['gender']
            current['gender_probability'] = previous['gender_probability']
            
        return current

    def update_statistics(self, analysis):
        """Update session statistics"""
        self.stats['total_detections'] += 1
        self.stats['emotions_history'].append(analysis['dominant_emotion'])
        self.stats['average_age'] = (
            (self.stats['average_age'] * (self.stats['total_detections'] - 1) + 
             analysis['age']) / self.stats['total_detections']
        )

    def display_session_stats(self, frame):
        """Display session statistics overlay"""
        if self.config['enable_statistics']:
            session_time = int(time.time() - self.stats['session_start'])
            stats_text = [
                f"Session Time: {session_time//60}m {session_time%60}s",
                f"Detections: {self.stats['total_detections']}",
                f"Avg Age: {self.stats['average_age']:.1f}",
                f"Common Emotion: {max(set(self.stats['emotions_history']), key=list(self.stats['emotions_history']).count)}"
            ]
            
            for i, text in enumerate(stats_text):
                self.draw_text(frame, text, 10, 30 + i*25, color_bg=(50, 50, 50))

    def enhance_low_light(self, frame):
        """Enhance frame in low light conditions"""
        if self.config['low_light_enhancement']:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return frame

    def detect_eyes(self, frame, face_region):
        """Detect eye blinks using facial landmarks"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Detect facial landmarks
            face_mesh = self.mp_face_mesh.process(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            
            if face_mesh.multi_face_landmarks:
                landmarks = face_mesh.multi_face_landmarks[0]
                
                # Get eye landmarks
                left_eye = [(int(landmark.x * face_region.shape[1]), 
                            int(landmark.y * face_region.shape[0])) 
                           for landmark in landmarks.landmark[133:145]]  # Left eye landmarks
                right_eye = [(int(landmark.x * face_region.shape[1]), 
                             int(landmark.y * face_region.shape[0])) 
                            for landmark in landmarks.landmark[362:374]]  # Right eye landmarks
                
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                # Detect blink
                if not hasattr(self, 'ear_history'):
                    self.ear_history = []
                
                self.ear_history.append(ear)
                if len(self.ear_history) > 3:
                    self.ear_history.pop(0)
                    if self.detect_blink(self.ear_history):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"[main.py] Eye detection error: {str(e)}")
            return False

    def detect_blink(self, ear_history):
        """Detect if a blink occurred"""
        EAR_THRESHOLD = 0.2
        return (ear_history[-2] < EAR_THRESHOLD and 
                ear_history[-3] >= EAR_THRESHOLD and 
                ear_history[-1] >= EAR_THRESHOLD)

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_ear(self, landmarks):
        """Calculate eye aspect ratio"""
        try:
            if isinstance(landmarks, mp.framework.formats.landmark_pb2.NormalizedLandmarkList):
                # Get eye landmarks
                left_eye = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]  # Left eye landmarks
                right_eye = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]  # Right eye landmarks
                
                # Calculate EAR for each eye
                left_ear = self._calculate_single_ear(left_eye)
                right_ear = self._calculate_single_ear(right_eye)
                
                # Return average EAR
                return (left_ear + right_ear) / 2.0
            else:
                # For direct point calculations
                v1 = self.calculate_distance(landmarks[1], landmarks[5])
                v2 = self.calculate_distance(landmarks[2], landmarks[4])
                h = self.calculate_distance(landmarks[0], landmarks[3])
                return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"[main.py] EAR calculation error: {str(e)}")
            return 0.0

    def _calculate_single_ear(self, eye_points):
        """Helper method to calculate EAR for a single eye"""
        try:
            # Calculate vertical distances
            v1 = abs(eye_points[1].y - eye_points[5].y)
            v2 = abs(eye_points[2].y - eye_points[4].y)
            # Calculate horizontal distance
            h = abs(eye_points[0].x - eye_points[3].x)
            
            # Return EAR
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except Exception as e:
            self.logger.error(f"[main.py] Single EAR calculation error: {str(e)}")
            return 0.0

    def calculate_mar(self, landmarks):
        """Calculate mouth aspect ratio"""
        try:
            if isinstance(landmarks, mp.framework.formats.landmark_pb2.NormalizedLandmarkList):
                # Get mouth landmarks
                mouth_points = [landmarks.landmark[i] for i in [61, 291, 39, 181, 0, 17]]
                
                # Calculate vertical distance
                v = abs(mouth_points[1].y - mouth_points[4].y)
                # Calculate horizontal distance
                h = abs(mouth_points[0].x - mouth_points[3].x)
                
                # Return MAR
                return v / h if h > 0 else 0.0
            return 0.0
        except Exception as e:
            self.logger.error(f"[main.py] MAR calculation error: {str(e)}")
            return 0.0

    def detect_drowsiness(self, ear):
        """Detect drowsiness based on eye aspect ratio"""
        if self.config['enable_drowsiness_detection']:
            DROWSINESS_THRESHOLD = 0.25
            FRAMES_THRESHOLD = 20
            
            if ear < DROWSINESS_THRESHOLD:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= FRAMES_THRESHOLD:
                    self.alert_drowsiness()
            else:
                self.drowsiness_counter = 0

    def detect_mask(self, face_region):
        """Detect if person is wearing a mask"""
        if self.config['enable_mask_detection']:
            try:
                # Implement mask detection logic here
                # This is a placeholder - you would need to implement or use a pre-trained model
                return False
            except Exception as e:
                self.logger.error(f"[main.py] Mask detection error: {str(e)}")
                return None

    def verify_age(self, age):
        """Verify if person meets minimum age requirement"""
        if self.config['enable_age_verification']:
            if age < self.config['minimum_age']:
                self.alert_age_verification()
                return False
            return True

    def play_sound(self, sound_type):
        """Play sound effect if enabled"""
        if (self.config['enable_sound_effects'] and 
            sound_type in self.sounds and 
            self.config['sounds'].get(f'{sound_type}_alert', False)):
            try:
                self.sounds[sound_type].play()
            except Exception as e:
                self.logger.error(f"[main.py] Failed to play {sound_type} sound: {str(e)}")

    def alert_drowsiness(self):
        """Alert when drowsiness is detected"""
        self.play_sound('drowsiness')
        self.stats['drowsiness_alerts'] += 1
        self.logger.warning("Drowsiness detected!")

    def alert_security(self, reason):
        """Alert when security concern is detected"""
        self.play_sound('security')
        self.stats['security_alerts'] += 1
        self.logger.warning(f"Security alert: {reason}")

    def start_recording(self):
        """Start video recording"""
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_writer = cv2.VideoWriter(
                f"{self.output_dir}/recording_{timestamp}.avi", 
                fourcc, 20.0, (640,480)
            )
            self.recording = True

    def stop_recording(self):
        """Stop video recording"""
        if self.recording:
            self.video_writer.release()
            self.recording = False

    def toggle_sound_effects(self):
        """Toggle all sound effects"""
        self.config['enable_sound_effects'] = not self.config['enable_sound_effects']
        if self.config['enable_sound_effects']:
            pygame.mixer.init()
            self.load_alert_sounds()
        else:
            pygame.mixer.quit()
            self.sounds = {}

    def toggle_specific_sound(self, sound_type):
        """Toggle specific sound effect"""
        if sound_type in self.config['sounds']:
            self.config['sounds'][sound_type] = not self.config['sounds'][sound_type]

    def process_frame_thread(self):
        """Thread for processing frames"""
        while self.processing:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft()
                results = self.process_single_frame(frame)
                self.result_queue.append((frame, results))

    def process_single_frame(self, frame):
        """Process a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect_system_capabilities(self):
        """Detect system capabilities and adjust parameters"""
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 2200  # Your CPU frequency
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Optimized parameters for your system
        self.config.update({
            'frame_width': 960,  # Half of 1920 for better performance
            'frame_height': 540,  # Half of 1080
            'target_fps': 24,    # Increased FPS target
            'detection_interval': 0.04,  # ~25fps
            'skip_frames': 2,    # Skip fewer frames
            'use_threading': True,  # Enable threading (4 cores available)
            'batch_process': True,  # Enable batch processing (plenty of RAM)
            'detection_confidence': 0.8,  # Higher confidence threshold
            'enable_background_processing': True,
            'buffer_size': 2,
            'process_every_n_frames': 2,
            'face_detection_scale': 1.1,
            'face_detection_neighbors': 4,
            'min_face_size': (80, 80),
            'max_faces': 4,  # Limit number of faces to process
            'temporal_smoothing': True,
            'smoothing_factor': 0.7,
            'enable_gpu_processing': False  # Intel UHD 620 might not help much
        })
        
        # Log system info
        self.system_info = {
            'cpu_cores': cpu_count,
            'cpu_freq': cpu_freq,
            'total_memory': memory.total,
            'os': platform.system(),
            'os_version': platform.version()
        }

    def create_stats_window(self):
        """Create enhanced statistics window with all analysis data"""
        stats_height = 1500  # Increased height to ensure all content fits
        stats_width = 900   # Increased width to fit environment text
        stats_image = np.ones((stats_height, stats_width, 3), np.uint8) * 245

        def add_text(text, y_pos, color=(50, 50, 50), scale=0.7, x=30):
            cv2.putText(stats_image, str(text), (x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            return y_pos

        def add_section_header(text, y_pos):
            cv2.rectangle(stats_image, (20, y_pos-30), (stats_width-20, y_pos+10), 
                         (230, 240, 255), -1)
            cv2.putText(stats_image, text, (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)
            return y_pos + 40

        def add_text_with_progress(label, value, y_pos, color=(50, 50, 50)):
            # Increased width for label
            add_text(f"{label}: ", y_pos, color, x=30)
            
            if isinstance(value, (int, float)):
                try:
                    norm_value = float(value)
                    if norm_value > 1 and norm_value <= 100:
                        norm_value = value
                    elif norm_value <= 1:
                        norm_value = value * 100
                    else:
                        norm_value = min(value, 100)
                    
                    # Adjusted bar position and width
                    bar_width = int((norm_value / 100) * 400)
                    cv2.rectangle(stats_image, 
                                (300, y_pos-15),  # Moved right
                                (300 + bar_width, y_pos),
                                (0, 120, 180), -1)
                    cv2.rectangle(stats_image, 
                                (300, y_pos-15),  # Moved right
                                (700, y_pos),  # Increased width
                                (200, 200, 200), 1)
                    add_text(f"{norm_value:.1f}%", y_pos, color, x=720)  # Moved percentage
                except:
                    add_text(str(value), y_pos, color, x=300)  # Moved text
            else:
                add_text(str(value), y_pos, color, x=300)  # Moved text
            
            return y_pos + 30

        # Get current analysis data
        if self.detection_complete and self.averaged_analysis:
            analysis = self.averaged_analysis
        elif hasattr(self, 'current_analysis') and self.current_analysis:
            analysis = self.current_analysis
        else:
            analysis = None

        # Start with title at top
        y_pos = 40
        # Title with larger font and more prominent background
        cv2.rectangle(stats_image, (0, 0), (stats_width, 70), (200, 230, 255), -1)
        y_pos = add_text("Face Analysis Dashboard", y_pos, (0, 100, 200), 1.2, x=300)
        cv2.line(stats_image, (20, y_pos), (stats_width-20, y_pos), (0, 100, 200), 2)
        y_pos += 30

        if analysis:
            # Basic Analysis
            y_pos = add_section_header("Basic Analysis", y_pos)
            y_pos = add_text_with_progress("Gender", analysis.get('gender', 'Unknown'), y_pos)
            y_pos = add_text_with_progress("Age", f"{analysis.get('age', 'Unknown')} years", y_pos)
            y_pos = add_text_with_progress("Dominant Emotion", analysis.get('dominant_emotion', 'Unknown'), y_pos)
            y_pos += 20

            # Emotion Analysis
            if 'emotion' in analysis:
                y_pos = add_section_header("Emotion Analysis", y_pos)
                for emotion, value in analysis['emotion'].items():
                    y_pos = add_text_with_progress(emotion, float(value), y_pos)
                y_pos += 20

            # Race Analysis
            if 'race' in analysis:
                y_pos = add_section_header("Race Analysis", y_pos)
                for race, value in analysis['race'].items():
                    y_pos = add_text_with_progress(race, float(value), y_pos)
                y_pos += 20

            # Attention Metrics
            y_pos = add_section_header("Attention Metrics", y_pos)
            
            # Calculate blink rate properly
            current_time = time.time()
            one_minute_ago = current_time - 60
            recent_blinks = len([t for t in self.stats['blink_timestamps'] 
                               if t > one_minute_ago])
            
            # Display metrics
            y_pos = add_text_with_progress("Blink Rate (per min)", str(recent_blinks), y_pos)
            y_pos = add_text_with_progress("Total Blinks", str(self.stats['blink_count']), y_pos)
            
            # Health Monitoring
            y_pos = add_section_header("Health Monitoring", y_pos)
            drowsiness_percentage = min((self.stats['drowsiness_alerts'] / 
                                       max(self.stats['total_frames'], 1)) * 100, 100)
            y_pos = add_text_with_progress("Drowsiness Level", drowsiness_percentage, y_pos)
            y_pos = add_text_with_progress("Drowsiness Alerts", 
                                         str(self.stats['drowsiness_alerts']), y_pos)
            
            # Security Status
            y_pos = add_section_header("Security Status", y_pos)
            y_pos = add_text_with_progress("Security Alerts", self.stats.get('security_alerts', 0), y_pos)
            y_pos = add_text_with_progress("Unauthorized Attempts", len(self.security_monitor.unauthorized_attempts), y_pos)
            y_pos += 20

            # Environment Analysis
            y_pos = add_section_header("Environment Analysis", y_pos)
            light_level = self.stats.get('light_level', 0)
            if isinstance(light_level, str):
                light_level = 50  # Default value if string
            y_pos = add_text_with_progress("Light Level", light_level, y_pos)
            frame_quality = self.stats.get('frame_quality', 0)
            if isinstance(frame_quality, str):
                frame_quality = 50  # Default value if string
            y_pos = add_text_with_progress("Frame Quality", frame_quality, y_pos)
            y_pos += 20

            # Session Statistics
            y_pos = add_section_header("Session Statistics", y_pos)
            session_time = int(time.time() - self.stats['session_start'])
            y_pos = add_text_with_progress("Session Duration", 
                                         f"{session_time//3600:02d}:{(session_time%3600)//60:02d}:{session_time%60:02d}", 
                                         y_pos)
            y_pos = add_text_with_progress("Total Detections", 
                                         str(self.stats['total_detections']), y_pos)
            y_pos = add_text_with_progress("Average Age", 
                                         f"{self.stats['average_age']:.1f}", y_pos)

        else:
            y_pos = add_text("No face detected", 40, (200, 0, 0))

        # Add footer
        cv2.rectangle(stats_image, (0, stats_height-40), (stats_width, stats_height), (200, 230, 255), -1)
        cv2.putText(stats_image, "Press 'L' to toggle statistics", 
                    (250, stats_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        return stats_image

    def get_cpu_temp(self):
        """Get CPU temperature (placeholder)"""
        try:
            temp = psutil.sensors_temperatures().get('coretemp', [{}])[0].current
            return temp
        except:
            return 0.0

    def get_storage_info(self):
        """Get storage information"""
        usage = psutil.disk_usage('/')
        gb_free = usage.free / (1024**3)
        return f"{gb_free:.1f}GB free"

    def run(self):
        """Main processing loop"""
        self.averaged_analysis = None
        frame_count = 0
        
        def update():
            nonlocal frame_count
            ret, frame = self.cap.read()
            if ret:
                # Store the current frame
                self.last_frame = frame.copy()
                
                faces = self.face_detector.detect_faces(frame)

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        if frame_count % self.config['process_every_n_frames'] == 0:
                            analysis = self.analyze_face_details(frame, x, y, w, h, frame_count)
                            if analysis:
                                self.draw_face_info(frame, x, y, analysis)
                else:
                    self.averaged_analysis = None

                frame_count += 1
                
                # Update UI instead of direct frame updates
                self.ui.update_camera_frame(frame)
                
                if self.ui.show_stats:
                    try:
                        stats_image = self.create_stats_window()
                        self.ui.update_stats_frame(stats_image)
                    except Exception as e:
                        self.logger.error(f"[main.py] Stats window error: {str(e)}")

            # Schedule next update through UI
            self.ui.root.after(int(1000/self.config['target_fps']), update)

        # Start update loop through UI
        update()
        self.ui.run()

    def cleanup(self):
        """Cleanup resources"""
        if self.recording:
            self.stop_recording()
        self.logger.info("Session ended")
        self.cap.release()
        # Let UI handle window cleanup
        self.ui.root.quit()
        self.ui.root.destroy()

        # Add to cleanup
        self.model_manager.cleanup()

    def alert_fatigue(self, alert_message):
        """Handle fatigue alerts"""
        self.stats['fatigue_alerts'] += 1
        self.logger.warning(f"Fatigue alert: {alert_message}")
        if self.config['enable_sound_effects']:
            self.play_sound('fatigue_alert')

def main():
    analyzer = FaceAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()

