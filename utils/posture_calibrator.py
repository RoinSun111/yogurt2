import logging
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import mediapipe as mp

class PostureCalibrator:
    """
    Custom posture classifier that can be trained on user-specific data
    to improve accuracy over MediaPipe's generic pose detection
    """
    
    def __init__(self, model_path="models/posture_classifier.pkl"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        self.metadata_path = model_path.replace('.pkl', '_metadata.json')
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_metadata = {}
        
        # MediaPipe pose for feature extraction
        self.mp_pose = mp.solutions.pose
        
        # Load existing model if available
        self._load_model()
        
        self.logger.info(f"Posture calibrator initialized. Trained: {self.is_trained}")
    
    def extract_posture_features(self, pose_landmarks):
        """
        Extract comprehensive features from MediaPipe pose landmarks
        for posture classification
        """
        if not pose_landmarks:
            return None
        
        features = []
        landmark_names = []
        
        # Key landmarks for posture analysis
        key_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        # Extract raw coordinates and visibility
        for landmark_idx in key_landmarks:
            landmark = pose_landmarks[landmark_idx]
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            landmark_names.extend([
                f"{landmark_idx.name}_x",
                f"{landmark_idx.name}_y", 
                f"{landmark_idx.name}_z",
                f"{landmark_idx.name}_visibility"
            ])
        
        # Calculate derived features (angles and distances)
        derived_features, derived_names = self._calculate_derived_features(pose_landmarks)
        features.extend(derived_features)
        landmark_names.extend(derived_names)
        
        # Store feature names for reference
        if not self.feature_names:
            self.feature_names = landmark_names
        
        return np.array(features)
    
    def _calculate_derived_features(self, landmarks):
        """
        Calculate derived features like angles and distances
        that are important for posture classification
        """
        features = []
        names = []
        
        # Shoulder-hip angles (left and right)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Left side angle
        left_angle = self._calculate_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_hip.x, left_hip.y),
            (left_hip.x, 0)
        )
        features.append(left_angle)
        names.append("left_shoulder_hip_angle")
        
        # Right side angle
        right_angle = self._calculate_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_hip.x, right_hip.y),
            (right_hip.x, 0)
        )
        features.append(right_angle)
        names.append("right_shoulder_hip_angle")
        
        # Average angle
        features.append((left_angle + right_angle) / 2)
        names.append("avg_shoulder_hip_angle")
        
        # Shoulder alignment (height difference)
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        features.append(shoulder_height_diff)
        names.append("shoulder_height_diff")
        
        # Head forward position
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        ear_center_x = (left_ear.x + right_ear.x) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        head_forward = abs(ear_center_x - shoulder_center_x)
        features.append(head_forward)
        names.append("head_forward_distance")
        
        # Neck angle (ear-nose-shoulder)
        neck_angle = self._calculate_angle(
            (left_ear.x, left_ear.y),
            (nose.x, nose.y),
            (left_shoulder.x, left_shoulder.y)
        )
        features.append(neck_angle)
        names.append("neck_angle")
        
        # Spine curvature estimate
        shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        
        # Use knees for spine reference
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        knee_center = ((left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2)
        
        spine_angle = self._calculate_angle(shoulder_center, hip_center, knee_center)
        features.append(spine_angle)
        names.append("spine_curvature_angle")
        
        # Torso width (shoulder distance)
        torso_width = abs(left_shoulder.x - right_shoulder.x)
        features.append(torso_width)
        names.append("torso_width")
        
        # Hip alignment
        hip_height_diff = abs(left_hip.y - right_hip.y)
        features.append(hip_height_diff)
        names.append("hip_height_diff")
        
        return features, names
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        import math
        
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return 0
        
        cos_angle = min(1.0, max(-1.0, dot_product / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def collect_training_sample(self, pose_landmarks, label, user_id="default"):
        """
        Collect a training sample with pose landmarks and ground truth label
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            label: "upright" or "slouched"
            user_id: Identifier for user-specific training
        """
        if label not in ["upright", "slouched"]:
            raise ValueError("Label must be 'upright' or 'slouched'")
        
        features = self.extract_posture_features(pose_landmarks)
        if features is None:
            self.logger.warning("Could not extract features from pose landmarks")
            return False
        
        # Initialize training data storage
        if not hasattr(self, 'training_data'):
            self.training_data = {
                'features': [],
                'labels': [],
                'timestamps': [],
                'user_ids': []
            }
        
        # Store training sample
        self.training_data['features'].append(features)
        self.training_data['labels'].append(label)
        self.training_data['timestamps'].append(datetime.now().isoformat())
        self.training_data['user_ids'].append(user_id)
        
        self.logger.info(f"Collected training sample: {label} (Total samples: {len(self.training_data['labels'])})")
        return True
    
    def train_classifier(self, model_type="random_forest", test_size=0.2):
        """
        Train the posture classifier on collected samples
        
        Args:
            model_type: "logistic_regression" or "random_forest"
            test_size: Fraction of data to use for testing
        """
        if not hasattr(self, 'training_data') or len(self.training_data['labels']) < 10:
            raise ValueError("Need at least 10 training samples to train classifier")
        
        # Prepare data
        X = np.array(self.training_data['features'])
        y = np.array(self.training_data['labels'])
        
        # Encode labels
        label_map = {'upright': 1, 'slouched': 0}
        y_encoded = np.array([label_map[label] for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        else:
            raise ValueError("model_type must be 'logistic_regression' or 'random_forest'")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Store training metadata
        self.training_metadata = {
            'model_type': model_type,
            'training_date': datetime.now().isoformat(),
            'num_samples': len(X),
            'train_score': float(train_score),
            'test_score': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_count': len(self.feature_names),
            'class_distribution': {
                'upright': int(np.sum(y_encoded)),
                'slouched': int(len(y_encoded) - np.sum(y_encoded))
            }
        }
        
        # Get detailed classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=['slouched', 'upright'],
            output_dict=True
        )
        self.training_metadata['classification_report'] = class_report
        
        # Feature importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            # Get top 10 most important features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            self.training_metadata['top_features'] = top_features
        
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        self.logger.info(f"Trained {model_type} classifier:")
        self.logger.info(f"  Training accuracy: {train_score:.3f}")
        self.logger.info(f"  Test accuracy: {test_score:.3f}")
        self.logger.info(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.training_metadata
    
    def predict_posture(self, pose_landmarks):
        """
        Predict posture using the trained classifier
        
        Returns:
            dict with prediction, confidence, and method used
        """
        if not self.is_trained or self.model is None:
            return {
                'posture': 'unknown',
                'confidence': 0.0,
                'method': 'untrained',
                'fallback_needed': True
            }
        
        features = self.extract_posture_features(pose_landmarks)
        if features is None:
            return {
                'posture': 'unknown',
                'confidence': 0.0,
                'method': 'feature_extraction_failed',
                'fallback_needed': True
            }
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()
            
            posture = 'upright' if prediction == 1 else 'slouched'
            
            return {
                'posture': posture,
                'confidence': float(confidence),
                'method': 'custom_classifier',
                'fallback_needed': False
            }
            
        except Exception as e:
            self.logger.error(f"Error in posture prediction: {str(e)}")
            return {
                'posture': 'unknown',
                'confidence': 0.0,
                'method': 'prediction_error',
                'fallback_needed': True
            }
    
    def _save_model(self):
        """Save trained model and metadata"""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata_to_save = {
                'training_metadata': self.training_metadata,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def _load_model(self):
        """Load existing trained model"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Load model
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                # Load scaler
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.training_metadata = metadata.get('training_metadata', {})
                        self.feature_names = metadata.get('feature_names', [])
                        self.is_trained = metadata.get('is_trained', False)
                
                self.logger.info("Loaded existing trained model")
                
        except Exception as e:
            self.logger.warning(f"Could not load existing model: {str(e)}")
            self.model = None
            self.is_trained = False
    
    def get_training_status(self):
        """Get current training status and statistics"""
        status = {
            'is_trained': self.is_trained,
            'model_exists': self.model is not None,
            'samples_collected': 0,
            'training_metadata': self.training_metadata
        }
        
        if hasattr(self, 'training_data'):
            status['samples_collected'] = len(self.training_data['labels'])
            status['label_distribution'] = {
                label: self.training_data['labels'].count(label)
                for label in set(self.training_data['labels'])
            }
        
        return status
    
    def reset_training_data(self):
        """Clear collected training data"""
        if hasattr(self, 'training_data'):
            del self.training_data
        self.logger.info("Training data cleared")
    
    def export_training_data(self, filepath):
        """Export collected training data for analysis"""
        if not hasattr(self, 'training_data'):
            raise ValueError("No training data to export")
        
        export_data = {
            'features': [f.tolist() for f in self.training_data['features']],
            'labels': self.training_data['labels'],
            'timestamps': self.training_data['timestamps'],
            'user_ids': self.training_data['user_ids'],
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Training data exported to {filepath}")