import cv2
import numpy as np
import mediapipe as mp
import logging
from pathlib import Path
import tensorflow as tf
import torch
import onnxruntime as ort

class ModelManager:
    def __init__(self):
        self.logger = logging.getLogger('ModelManager')
        self.models_path = Path('models')
        self.models_path.mkdir(exist_ok=True)
        
        # Check for existing models
        self.available_models = self._scan_models_directory()
        
        # Initialize model paths and check existence
        self.model_paths = {
            'face_landmark': str(self.models_path / 'face_landmarks.dat'),
            'age_net': str(self.models_path / 'age_net.caffemodel'),
            'gender_net': str(self.models_path / 'gender_net.caffemodel'),
            'emotion_net': str(self.models_path / 'emotion_net.h5'),
            'face_recognition': str(self.models_path / 'face_recognition.pth'),
            'gaze_estimation': str(self.models_path / 'gaze_estimation.onnx'),
            'head_pose': str(self.models_path / 'head_pose.onnx'),
            'face_quality': str(self.models_path / 'face_quality.pb')
        }
        
        # Initialize MediaPipe models (always available)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize additional models if available
        self.initialize_additional_models()
        
    def _scan_models_directory(self):
        """Scan models directory for available models"""
        available_models = {}
        if self.models_path.exists():
            for model_file in self.models_path.glob('*'):
                if model_file.is_file():
                    model_type = model_file.suffix.lower()
                    model_name = model_file.stem.lower()
                    available_models[model_name] = {
                        'path': str(model_file),
                        'type': model_type
                    }
        return available_models
        
    def initialize_additional_models(self):
        """Initialize additional models based on availability"""
        try:
            # Initialize ONNX models
            if 'gaze_estimation.onnx' in self.available_models:
                self.gaze_session = ort.InferenceSession(
                    self.model_paths['gaze_estimation']
                )
            
            if 'head_pose.onnx' in self.available_models:
                self.head_pose_session = ort.InferenceSession(
                    self.model_paths['head_pose']
                )
            
            # Initialize TensorFlow models
            if 'face_quality.pb' in self.available_models:
                self.face_quality_model = tf.saved_model.load(
                    self.model_paths['face_quality']
                )
            
            # Initialize PyTorch models
            if 'face_recognition.pth' in self.available_models and torch.cuda.is_available():
                self.face_recognition_model = torch.load(
                    self.model_paths['face_recognition']
                )
                self.face_recognition_model.eval()
                
        except Exception as e:
            self.logger.error(f"Model initialization error: {str(e)}")
    
    def get_face_landmarks(self, frame):
        """Get facial landmarks using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.mp_face_mesh.process(rgb_frame)
        except Exception as e:
            self.logger.error(f"Face landmark error: {str(e)}")
            return None
    
    def estimate_gaze(self, face_region):
        """Estimate gaze direction if model available"""
        try:
            if hasattr(self, 'gaze_session'):
                # Preprocess input
                input_name = self.gaze_session.get_inputs()[0].name
                preprocessed = cv2.resize(face_region, (224, 224))
                preprocessed = preprocessed.transpose(2, 0, 1)
                preprocessed = preprocessed[np.newaxis, ...]
                
                # Run inference
                result = self.gaze_session.run(None, {input_name: preprocessed})
                return result[0]
            return None
        except Exception as e:
            self.logger.error(f"Gaze estimation error: {str(e)}")
            return None
    
    def estimate_head_pose(self, face_region):
        """Estimate head pose if model available"""
        try:
            if hasattr(self, 'head_pose_session'):
                # Preprocess input
                input_name = self.head_pose_session.get_inputs()[0].name
                preprocessed = cv2.resize(face_region, (224, 224))
                preprocessed = preprocessed.transpose(2, 0, 1)
                preprocessed = preprocessed[np.newaxis, ...]
                
                # Run inference
                result = self.head_pose_session.run(None, {input_name: preprocessed})
                return {
                    'yaw': result[0][0],
                    'pitch': result[0][1],
                    'roll': result[0][2]
                }
            return None
        except Exception as e:
            self.logger.error(f"Head pose estimation error: {str(e)}")
            return None
    
    def assess_face_quality(self, face_region):
        """Assess face image quality if model available"""
        try:
            if hasattr(self, 'face_quality_model'):
                preprocessed = cv2.resize(face_region, (224, 224))
                preprocessed = preprocessed / 255.0
                preprocessed = np.expand_dims(preprocessed, axis=0)
                
                result = self.face_quality_model(preprocessed)
                return float(result[0])
            return None
        except Exception as e:
            self.logger.error(f"Face quality assessment error: {str(e)}")
            return None
    
    def cleanup(self):
        """Cleanup model resources"""
        try:
            self.mp_face_mesh.close()
            self.mp_face_detection.close()
            self.mp_hands.close()
            
            # Cleanup additional models
            if hasattr(self, 'gaze_session'):
                del self.gaze_session
            if hasattr(self, 'head_pose_session'):
                del self.head_pose_session
            if hasattr(self, 'face_recognition_model'):
                del self.face_recognition_model
                
        except Exception as e:
            self.logger.error(f"Model cleanup error: {str(e)}") 