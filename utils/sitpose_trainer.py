
import numpy as np
import pandas as pd
import os
import logging
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mediapipe as mp

logger = logging.getLogger(__name__)

class SitPoseTrainer:
    """
    SitPose-inspired posture classifier trainer
    Based on research: "SitPose: Real-Time Detection of Sitting Posture and Sedentary Behavior"
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "sitpose_classifier.pkl")
        self.scaler_path = os.path.join(model_dir, "sitpose_scaler.pkl")
        self.metadata_path = os.path.join(model_dir, "sitpose_metadata.json")
        
        # SitPose 7-class classification
        self.posture_classes = {
            0: 'sitting_straight',
            1: 'hunching_over', 
            2: 'left_sitting',
            3: 'right_sitting',
            4: 'leaning_forward',
            5: 'lying',
            6: 'standing'
        }
        
        # Initialize MediaPipe for feature extraction
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Create model directory
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def extract_sitpose_features(self, landmarks):
        """
        Extract SitPose-style geometric features from pose landmarks
        
        Features based on the SitPose paper:
        1. Joint angles (shoulder-hip-knee)
        2. Body ratios (torso length, shoulder width)
        3. Symmetry measures
        4. Head orientation
        5. Spine curvature indicators
        """
        if not landmarks or len(landmarks) < 33:
            return np.zeros(42)  # Return zero vector if no landmarks
        
        features = []
        
        # Convert landmarks to numpy array for easier processing
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # 1. Joint angles (6 features)
        left_shoulder_angle = self._calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        right_shoulder_angle = self._calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        left_hip_angle = self._calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        right_hip_angle = self._calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        spine_angle = self._calculate_spine_angle(points)
        neck_angle = self._calculate_neck_angle(points)
        
        features.extend([left_shoulder_angle, right_shoulder_angle, left_hip_angle, 
                        right_hip_angle, spine_angle, neck_angle])
        
        # 2. Body ratios (8 features)
        torso_length = self._calculate_distance(
            points[self.mp_pose.PoseLandmark.NOSE.value],
            (points[self.mp_pose.PoseLandmark.LEFT_HIP.value] + 
             points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]) / 2
        )
        shoulder_width = self._calculate_distance(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
        hip_width = self._calculate_distance(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        shoulder_hip_ratio = shoulder_width / (hip_width + 1e-6)
        
        # Vertical positions (normalized)
        nose_y = points[self.mp_pose.PoseLandmark.NOSE.value][1]
        shoulder_y = (points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] + 
                     points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2
        hip_y = (points[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] + 
                points[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]) / 2
        knee_y = (points[self.mp_pose.PoseLandmark.LEFT_KNEE.value][1] + 
                 points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value][1]) / 2
        
        features.extend([torso_length, shoulder_width, hip_width, shoulder_hip_ratio,
                        nose_y, shoulder_y, hip_y, knee_y])
        
        # 3. Symmetry measures (10 features)
        # Shoulder symmetry
        shoulder_height_diff = abs(points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - 
                                  points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
        # Hip symmetry  
        hip_height_diff = abs(points[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] - 
                             points[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1])
        # Knee symmetry
        knee_height_diff = abs(points[self.mp_pose.PoseLandmark.LEFT_KNEE.value][1] - 
                              points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value][1])
        
        # Lateral lean indicators
        left_shoulder_hip_distance = self._calculate_distance(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        right_shoulder_hip_distance = self._calculate_distance(
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        # Center of mass indicators
        upper_body_center_x = (points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] + 
                              points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2
        lower_body_center_x = (points[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] + 
                              points[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]) / 2
        lateral_displacement = abs(upper_body_center_x - lower_body_center_x)
        
        # Forward lean indicators
        shoulder_x = (points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] + 
                     points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2
        hip_x = (points[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] + 
                points[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]) / 2
        forward_lean = shoulder_x - hip_x
        
        # Head position relative to shoulders
        nose_x = points[self.mp_pose.PoseLandmark.NOSE.value][0]
        head_forward = nose_x - shoulder_x
        
        features.extend([shoulder_height_diff, hip_height_diff, knee_height_diff,
                        left_shoulder_hip_distance, right_shoulder_hip_distance,
                        lateral_displacement, forward_lean, head_forward,
                        upper_body_center_x, lower_body_center_x])
        
        # 4. Head orientation (6 features)
        # Use ears and nose for head orientation
        left_ear = points[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = points[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        nose = points[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Head tilt (roll)
        ear_height_diff = abs(left_ear[1] - right_ear[1])
        ear_distance = self._calculate_distance(left_ear, right_ear)
        head_roll = ear_height_diff / (ear_distance + 1e-6)
        
        # Head turn (yaw) - simplified using ear visibility and nose position
        ear_center_x = (left_ear[0] + right_ear[0]) / 2
        nose_ear_offset = nose[0] - ear_center_x
        
        # Head nod (pitch) - nose position relative to ears
        ear_center_y = (left_ear[1] + right_ear[1]) / 2
        nose_ear_y_offset = nose[1] - ear_center_y
        
        features.extend([ear_height_diff, ear_distance, head_roll, 
                        nose_ear_offset, nose_ear_y_offset, ear_center_y])
        
        # 5. Additional postural indicators (12 features)
        # Wrist positions (for hunching detection)
        left_wrist_y = points[self.mp_pose.PoseLandmark.LEFT_WRIST.value][1]
        right_wrist_y = points[self.mp_pose.PoseLandmark.RIGHT_WRIST.value][1]
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        wrist_shoulder_diff = avg_wrist_y - shoulder_y
        
        # Elbow positions
        left_elbow_y = points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value][1]
        right_elbow_y = points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]
        avg_elbow_y = (left_elbow_y + right_elbow_y) / 2
        elbow_shoulder_diff = avg_elbow_y - shoulder_y
        
        # Ankle positions (for sitting vs standing)
        left_ankle_y = points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
        right_ankle_y = points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
        knee_ankle_diff = knee_y - avg_ankle_y
        
        # Overall body compactness
        body_height = nose_y - avg_ankle_y
        body_width = max(shoulder_width, hip_width)
        aspect_ratio = body_height / (body_width + 1e-6)
        
        # Standing detection features
        hip_knee_distance = abs(hip_y - knee_y)
        knee_ankle_distance = abs(knee_y - avg_ankle_y)
        
        features.extend([avg_wrist_y, wrist_shoulder_diff, avg_elbow_y, elbow_shoulder_diff,
                        avg_ankle_y, knee_ankle_diff, body_height, body_width, 
                        aspect_ratio, hip_knee_distance, knee_ankle_distance, torso_length])
        
        return np.array(features)
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        except:
            return 90.0  # Default angle
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)
    
    def _calculate_spine_angle(self, points):
        """Calculate spine curvature angle"""
        try:
            # Use shoulder midpoint, hip midpoint, and knee midpoint
            shoulder_mid = (points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value] + 
                           points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) / 2
            hip_mid = (points[self.mp_pose.PoseLandmark.LEFT_HIP.value] + 
                      points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]) / 2
            knee_mid = (points[self.mp_pose.PoseLandmark.LEFT_KNEE.value] + 
                       points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]) / 2
            
            return self._calculate_angle(shoulder_mid, hip_mid, knee_mid)
        except:
            return 180.0  # Straight spine default
    
    def _calculate_neck_angle(self, points):
        """Calculate neck angle"""
        try:
            nose = points[self.mp_pose.PoseLandmark.NOSE.value]
            shoulder_mid = (points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value] + 
                           points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) / 2
            hip_mid = (points[self.mp_pose.PoseLandmark.LEFT_HIP.value] + 
                      points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]) / 2
            
            return self._calculate_angle(nose, shoulder_mid, hip_mid)
        except:
            return 160.0  # Default neck angle
    
    def generate_synthetic_dataset(self, num_samples=5000):
        """
        Generate synthetic SitPose dataset with realistic feature distributions
        Based on typical posture characteristics from research
        """
        np.random.seed(42)
        
        data = []
        labels = []
        
        samples_per_class = num_samples // len(self.posture_classes)
        
        for class_id, class_name in self.posture_classes.items():
            for _ in range(samples_per_class):
                features = self._generate_class_features(class_name)
                data.append(features)
                labels.append(class_id)
        
        # Add some additional random samples
        remaining = num_samples - len(data)
        for _ in range(remaining):
            class_id = np.random.choice(list(self.posture_classes.keys()))
            class_name = self.posture_classes[class_id]
            features = self._generate_class_features(class_name)
            data.append(features)
            labels.append(class_id)
        
        return np.array(data), np.array(labels)
    
    def _generate_class_features(self, class_name):
        """Generate realistic features for a specific posture class"""
        # Base feature vector (42 features)
        features = np.random.normal(0, 0.1, 42)
        
        if class_name == 'sitting_straight':
            # Good posture characteristics
            features[0:2] = np.random.normal(90, 5, 2)  # Shoulder angles
            features[2:4] = np.random.normal(90, 5, 2)  # Hip angles  
            features[4] = np.random.normal(175, 5)      # Spine angle (close to straight)
            features[5] = np.random.normal(160, 10)     # Neck angle
            features[10:13] = np.random.normal(0, 0.02, 3)  # Low asymmetry
            features[16] = np.random.normal(0, 0.05)    # Minimal lateral displacement
            features[17] = np.random.normal(0, 0.03)    # Minimal forward lean
            
        elif class_name == 'hunching_over':
            # Hunched posture characteristics
            features[0:2] = np.random.normal(70, 10, 2)  # Reduced shoulder angles
            features[2:4] = np.random.normal(85, 8, 2)   # Slightly reduced hip angles
            features[4] = np.random.normal(145, 15)      # Curved spine
            features[5] = np.random.normal(140, 15)      # Forward head
            features[17] = np.random.normal(0.15, 0.05)  # Significant forward lean
            features[18] = np.random.normal(0.1, 0.03)   # Head forward
            features[26] = np.random.normal(-0.1, 0.03)  # Wrist-shoulder diff (arms forward)
            
        elif class_name == 'left_sitting':
            # Left lean characteristics
            features[0] = np.random.normal(75, 8)        # Left shoulder angle different
            features[1] = np.random.normal(95, 8)        # Right shoulder angle different
            features[2] = np.random.normal(85, 8)        # Left hip angle
            features[3] = np.random.normal(95, 8)        # Right hip angle
            features[4] = np.random.normal(165, 10)      # Slightly curved spine
            features[10] = np.random.normal(0.08, 0.02)  # Shoulder height difference
            features[16] = np.random.normal(-0.1, 0.03)  # Lateral displacement (left)
            
        elif class_name == 'right_sitting':
            # Right lean characteristics
            features[0] = np.random.normal(95, 8)        # Left shoulder angle
            features[1] = np.random.normal(75, 8)        # Right shoulder angle different
            features[2] = np.random.normal(95, 8)        # Left hip angle
            features[3] = np.random.normal(85, 8)        # Right hip angle different
            features[4] = np.random.normal(165, 10)      # Slightly curved spine
            features[10] = np.random.normal(0.08, 0.02)  # Shoulder height difference
            features[16] = np.random.normal(0.1, 0.03)   # Lateral displacement (right)
            
        elif class_name == 'leaning_forward':
            # Forward lean characteristics
            features[0:2] = np.random.normal(80, 8, 2)   # Slightly reduced shoulder angles
            features[2:4] = np.random.normal(95, 5, 2)   # Hip angles
            features[4] = np.random.normal(160, 10)      # Moderately curved spine
            features[5] = np.random.normal(150, 12)      # Forward head
            features[17] = np.random.normal(0.08, 0.03)  # Moderate forward lean
            features[18] = np.random.normal(0.05, 0.02)  # Head forward
            
        elif class_name == 'lying':
            # Lying down characteristics
            features[4] = np.random.normal(120, 20)      # Very curved "spine" 
            features[6:10] = np.random.normal(0.8, 0.1, 4)  # Low vertical positions
            features[32] = np.random.normal(0.3, 0.1)    # Low aspect ratio
            features[30] = np.random.normal(0.1, 0.05)   # Small knee-ankle difference
            
        elif class_name == 'standing':
            # Standing characteristics
            features[2:4] = np.random.normal(175, 5, 2)  # Straight hip angles
            features[4] = np.random.normal(175, 5)       # Straight spine
            features[6:10] = np.random.normal(0.2, 0.1, 4)  # Higher vertical positions
            features[32] = np.random.normal(2.5, 0.3)    # High aspect ratio
            features[33] = np.random.normal(0.3, 0.05)   # Large hip-knee distance
            features[34] = np.random.normal(0.4, 0.05)   # Large knee-ankle distance
        
        return features
    
    def train_model(self, X=None, y=None):
        """Train the SitPose classifier"""
        if X is None or y is None:
            logger.info("Generating synthetic SitPose dataset...")
            X, y = self.generate_synthetic_dataset()
        
        logger.info(f"Training SitPose model with {len(X)} samples...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier (good for pose classification)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = model.score(X_test_scaled, y_test)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                        target_names=list(self.posture_classes.values())))
        
        # Save the model and scaler
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': 'SitPose RandomForest Classifier',
            'classes': self.posture_classes,
            'feature_count': 42,
            'accuracy': accuracy,
            'training_date': datetime.now().isoformat(),
            'feature_names': self._get_feature_names()
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Scaler saved to {self.scaler_path}")
        logger.info(f"Metadata saved to {self.metadata_path}")
        
        return model, scaler, accuracy
    
    def _get_feature_names(self):
        """Get descriptive names for all 42 features"""
        return [
            'left_shoulder_angle', 'right_shoulder_angle', 'left_hip_angle', 'right_hip_angle',
            'spine_angle', 'neck_angle', 'torso_length', 'shoulder_width', 'hip_width',
            'shoulder_hip_ratio', 'nose_y', 'shoulder_y', 'hip_y', 'knee_y',
            'shoulder_height_diff', 'hip_height_diff', 'knee_height_diff',
            'left_shoulder_hip_dist', 'right_shoulder_hip_dist', 'lateral_displacement',
            'forward_lean', 'head_forward', 'upper_body_center_x', 'lower_body_center_x',
            'ear_height_diff', 'ear_distance', 'head_roll', 'nose_ear_offset',
            'nose_ear_y_offset', 'ear_center_y', 'avg_wrist_y', 'wrist_shoulder_diff',
            'avg_elbow_y', 'elbow_shoulder_diff', 'avg_ankle_y', 'knee_ankle_diff',
            'body_height', 'body_width', 'aspect_ratio', 'hip_knee_distance',
            'knee_ankle_distance', 'torso_length_2'
        ]
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model = joblib.load(self.model_path)
            scaler = joblib.load(self.scaler_path)
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded SitPose model with accuracy: {metadata.get('accuracy', 'unknown')}")
            return model, scaler, metadata
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, None, None
    
    def predict_posture(self, landmarks):
        """Predict posture from pose landmarks"""
        model, scaler, metadata = self.load_model()
        
        if model is None:
            logger.warning("No trained model available, training new model...")
            self.train_model()
            model, scaler, metadata = self.load_model()
        
        if model is None:
            return 'unknown', 0.0
        
        # Extract features
        features = self.extract_sitpose_features(landmarks)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        posture_name = self.posture_classes.get(prediction, 'unknown')
        
        return posture_name, confidence

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = SitPoseTrainer()
    model, scaler, accuracy = trainer.train_model()
    
    print(f"SitPose model trained successfully with {accuracy:.3f} accuracy")
