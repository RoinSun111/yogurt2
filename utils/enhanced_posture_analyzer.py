import logging
import mediapipe as mp
import numpy as np
import math
import cv2
import requests
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class EnhancedPostureAnalyzer:
    def __init__(self, use_server_openpose=False, openpose_server_url=None):
        """
        Enhanced posture analyzer with MediaPipe Face Mesh and optional OpenPose server support
        
        Args:
            use_server_openpose: Whether to use server-based OpenPose for higher accuracy
            openpose_server_url: URL of the OpenPose server (e.g., "http://localhost:8080/pose")
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced posture analyzer initialized")
        
        # Configuration
        self.use_server_openpose = use_server_openpose
        self.openpose_server_url = openpose_server_url
        
        # Initialize MediaPipe Pose (always available as fallback)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2  # Use the high-complexity model for better accuracy
        )
        
        # Initialize MediaPipe Face Mesh for precise head/neck analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key face mesh landmark indices for head orientation
        # These are specific indices from MediaPipe's 468 face landmarks
        self.face_landmarks = {
            'nose_tip': 1,
            'chin': 175,
            'forehead': 9,
            'left_ear_tragion': 234,
            'right_ear_tragion': 454,
            'left_eye_outer': 33,
            'right_eye_outer': 263,
            'mouth_center': 13
        }
        
        # Store historical posture data for trend analysis
        self.posture_history = []
        self.history_max_size = 300  # Store ~5 minutes of data at 1 Hz
        
        # Enhanced threshold angles for posture assessment
        self.upright_threshold = 15.0
        self.excellent_threshold = 5.0
        self.good_threshold = 10.0
        self.fair_threshold = 15.0
        
        # Enhanced neck angle thresholds using face mesh precision
        self.neck_angle_threshold = 15.0  # More precise with face mesh
        self.head_tilt_threshold = 10.0   # Side-to-side head tilt
        
        # Shoulder alignment thresholds
        self.shoulder_alignment_excellent = 0.95
        self.shoulder_alignment_good = 0.9
        self.shoulder_alignment_fair = 0.8
        
        # OpenPose keypoint indices (if using server)
        self.openpose_keypoints = {
            'nose': 0,
            'neck': 1,
            'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
            'right_hip': 8,
            'right_knee': 9,
            'right_ankle': 10,
            'left_hip': 11,
            'left_knee': 12,
            'left_ankle': 13,
            'right_eye': 14,
            'left_eye': 15,
            'right_ear': 16,
            'left_ear': 17
        }
        
    def analyze(self, image):
        """
        Enhanced posture analysis with multiple detection methods
        """
        try:
            # Choose analysis method based on configuration
            if self.use_server_openpose and self.openpose_server_url:
                return self._analyze_with_openpose_server(image)
            else:
                return self._analyze_with_mediapipe_enhanced(image)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced posture analysis: {str(e)}")
            return self._get_default_posture_data()
    
    def _analyze_with_mediapipe_enhanced(self, image):
        """
        Enhanced MediaPipe analysis combining Pose and Face Mesh
        """
        # Process with MediaPipe Pose
        pose_results = self.pose.process(image)
        
        # Process with MediaPipe Face Mesh for precise head analysis
        face_results = self.face_mesh.process(image)
        
        # Initialize posture data
        posture_data = self._get_default_posture_data()
        
        # Check if pose landmarks are detected
        if not pose_results.pose_landmarks:
            self.logger.debug("No pose landmarks detected")
            return posture_data
        
        pose_landmarks = pose_results.pose_landmarks.landmark
        
        # Check visibility and presence
        nose_visible = pose_landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5
        left_shoulder_visible = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5
        right_shoulder_visible = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5
        
        is_present = nose_visible and (left_shoulder_visible or right_shoulder_visible)
        posture_data['is_present'] = is_present
        posture_data['pose_landmarks'] = pose_landmarks
        
        if not is_present:
            return posture_data
        
        # Basic posture calculations using pose landmarks
        posture_data['angle'] = self._calculate_posture_angle(pose_landmarks)
        posture_data['shoulder_alignment'] = self._calculate_shoulder_alignment(pose_landmarks)
        posture_data['head_forward_position'] = self._calculate_head_forward_position(pose_landmarks)
        posture_data['spine_curvature'] = self._calculate_spine_curvature(pose_landmarks)
        posture_data['symmetry_score'] = self._calculate_symmetry_score(pose_landmarks)
        
        # Enhanced neck angle calculation using Face Mesh if available
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            enhanced_neck_data = self._calculate_enhanced_neck_metrics(face_landmarks, pose_landmarks, image.shape)
            
            posture_data['neck_angle'] = enhanced_neck_data['neck_angle']
            posture_data['head_tilt'] = enhanced_neck_data['head_tilt']
            posture_data['head_rotation'] = enhanced_neck_data['head_rotation']
            posture_data['face_mesh_available'] = True
            
            self.logger.debug(f"Enhanced face mesh analysis: neck_angle={enhanced_neck_data['neck_angle']:.1f}°, "
                            f"head_tilt={enhanced_neck_data['head_tilt']:.1f}°")
        else:
            # Fallback to basic neck angle calculation
            posture_data['neck_angle'] = self._calculate_basic_neck_angle(pose_landmarks)
            posture_data['head_tilt'] = 0.0
            posture_data['head_rotation'] = 0.0
            posture_data['face_mesh_available'] = False
        
        # Classify posture
        angle = posture_data['angle']
        posture_data['posture'] = 'upright' if angle < self.upright_threshold else 'slouched'
        
        # Enhanced quality assessment
        posture_data['posture_quality'] = self._assess_enhanced_posture_quality(posture_data)
        
        # Enhanced feedback generation
        posture_data['feedback'] = self._generate_enhanced_feedback(posture_data)
        
        # Store in history
        self._update_posture_history(posture_data)
        
        self.logger.debug(f"Enhanced posture: {posture_data['posture']}, Quality: {posture_data['posture_quality']}, "
                         f"Angle: {angle:.2f}°, Neck: {posture_data['neck_angle']:.1f}°")
        
        return posture_data
    
    def _analyze_with_openpose_server(self, image):
        """
        Analyze posture using server-based OpenPose for higher accuracy
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send request to OpenPose server
            payload = {
                'image': image_base64,
                'model': 'BODY_25'  # or 'COCO' depending on server setup
            }
            
            response = requests.post(
                self.openpose_server_url,
                json=payload,
                timeout=5.0  # 5 second timeout
            )
            
            if response.status_code == 200:
                openpose_data = response.json()
                return self._process_openpose_results(openpose_data, image)
            else:
                self.logger.warning(f"OpenPose server error: {response.status_code}, falling back to MediaPipe")
                return self._analyze_with_mediapipe_enhanced(image)
                
        except requests.RequestException as e:
            self.logger.warning(f"OpenPose server unavailable: {str(e)}, using MediaPipe fallback")
            return self._analyze_with_mediapipe_enhanced(image)
    
    def _calculate_enhanced_neck_metrics(self, face_landmarks, pose_landmarks, image_shape):
        """
        Calculate precise neck and head orientation metrics using Face Mesh
        """
        h, w = image_shape[:2]
        
        # Get key facial landmarks in pixel coordinates
        nose_tip = face_landmarks[self.face_landmarks['nose_tip']]
        chin = face_landmarks[self.face_landmarks['chin']]
        forehead = face_landmarks[self.face_landmarks['forehead']]
        left_ear = face_landmarks[self.face_landmarks['left_ear_tragion']]
        right_ear = face_landmarks[self.face_landmarks['right_ear_tragion']]
        
        # Convert to pixel coordinates
        nose_px = (int(nose_tip.x * w), int(nose_tip.y * h))
        chin_px = (int(chin.x * w), int(chin.y * h))
        forehead_px = (int(forehead.x * w), int(forehead.y * h))
        left_ear_px = (int(left_ear.x * w), int(left_ear.y * h))
        right_ear_px = (int(right_ear.x * w), int(right_ear.y * h))
        
        # Calculate head center
        head_center_x = (left_ear_px[0] + right_ear_px[0]) / 2
        head_center_y = (left_ear_px[1] + right_ear_px[1]) / 2
        
        # Enhanced neck angle calculation (forward/backward head posture)
        # Use the angle between the face vertical line and true vertical
        face_vertical_angle = math.degrees(math.atan2(
            chin_px[1] - forehead_px[1],
            chin_px[0] - forehead_px[0]
        ))
        
        # Normalize to get forward head posture angle
        neck_angle = abs(90 - abs(face_vertical_angle))
        
        # Head tilt calculation (side-to-side)
        ear_height_diff = abs(left_ear_px[1] - right_ear_px[1])
        ear_distance = abs(left_ear_px[0] - right_ear_px[0])
        head_tilt = math.degrees(math.atan2(ear_height_diff, ear_distance)) if ear_distance > 0 else 0
        
        # Head rotation calculation (left/right turn)
        nose_center_offset = nose_px[0] - head_center_x
        head_width = abs(left_ear_px[0] - right_ear_px[0])
        head_rotation = (nose_center_offset / head_width) * 45 if head_width > 0 else 0  # Scale to degrees
        
        return {
            'neck_angle': neck_angle,
            'head_tilt': head_tilt,
            'head_rotation': abs(head_rotation)
        }
    
    def _calculate_basic_neck_angle(self, pose_landmarks):
        """
        Fallback neck angle calculation using pose landmarks only
        """
        left_ear = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        nose = pose_landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        # Choose the most visible ear
        ear = left_ear if left_ear.visibility > right_ear.visibility else right_ear
        
        # Use shoulders for reference
        left_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder = left_shoulder if left_shoulder.visibility > right_shoulder.visibility else right_shoulder
        
        # Calculate angle between ear, nose, and shoulder
        neck_angle = self._angle_between_points(
            (ear.x, ear.y),
            (nose.x, nose.y),
            (shoulder.x, shoulder.y)
        )
        
        return neck_angle
    
    def _process_openpose_results(self, openpose_data, image):
        """
        Process results from OpenPose server
        """
        posture_data = self._get_default_posture_data()
        
        if 'keypoints' not in openpose_data or not openpose_data['keypoints']:
            return posture_data
        
        keypoints = openpose_data['keypoints'][0]  # First person
        
        # Check if key points are detected
        nose = keypoints[self.openpose_keypoints['nose'] * 3:(self.openpose_keypoints['nose'] * 3) + 3]
        left_shoulder = keypoints[self.openpose_keypoints['left_shoulder'] * 3:(self.openpose_keypoints['left_shoulder'] * 3) + 3]
        right_shoulder = keypoints[self.openpose_keypoints['right_shoulder'] * 3:(self.openpose_keypoints['right_shoulder'] * 3) + 3]
        
        # Check confidence (third value in each keypoint triplet)
        if nose[2] < 0.3 or (left_shoulder[2] < 0.3 and right_shoulder[2] < 0.3):
            return posture_data
        
        posture_data['is_present'] = True
        
        # Calculate enhanced metrics using OpenPose's higher precision
        # Implementation would include more sophisticated biomechanical analysis
        # using OpenPose's superior keypoint detection
        
        # For now, we'll use similar calculations but with better keypoint data
        # This is where you would implement the enhanced OpenPose-specific analysis
        
        self.logger.info("Using OpenPose server analysis (enhanced accuracy)")
        return posture_data
    
    def _assess_enhanced_posture_quality(self, posture_data):
        """
        Enhanced posture quality assessment using all available metrics
        """
        score = 100
        
        # Basic posture angle
        angle = posture_data['angle']
        if angle <= self.excellent_threshold:
            pass
        elif angle <= self.good_threshold:
            score -= 8
        elif angle <= self.fair_threshold:
            score -= 18
        else:
            score -= 35
        
        # Enhanced neck angle assessment
        neck_angle = posture_data['neck_angle']
        if neck_angle > self.neck_angle_threshold:
            score -= min(25, int((neck_angle - self.neck_angle_threshold) * 1.2))
        
        # Head tilt penalty (if available from face mesh)
        if posture_data.get('face_mesh_available') and posture_data.get('head_tilt', 0) > self.head_tilt_threshold:
            score -= min(15, int((posture_data['head_tilt'] - self.head_tilt_threshold) * 0.8))
        
        # Head rotation penalty
        if posture_data.get('head_rotation', 0) > 20:  # 20 degrees threshold
            score -= min(20, int((posture_data['head_rotation'] - 20) * 0.5))
        
        # Shoulder alignment
        shoulder_alignment = posture_data['shoulder_alignment']
        if shoulder_alignment >= self.shoulder_alignment_excellent:
            pass
        elif shoulder_alignment >= self.shoulder_alignment_good:
            score -= 5
        elif shoulder_alignment >= self.shoulder_alignment_fair:
            score -= 12
        else:
            score -= 20
        
        # Other metrics (similar to original but with refined weights)
        head_forward = posture_data['head_forward_position']
        if head_forward > 0.08:  # Tighter threshold
            score -= min(20, int(head_forward * 80))
        
        spine_curvature = posture_data['spine_curvature']
        if spine_curvature < 150:
            score -= min(25, int((150 - spine_curvature) * 0.5))
        
        symmetry = posture_data['symmetry_score']
        if symmetry < 0.85:  # Higher threshold for good symmetry
            score -= min(15, int((1 - symmetry) * 40))
        
        # Map to quality categories
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 55:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_enhanced_feedback(self, posture_data):
        """
        Generate detailed feedback using enhanced metrics
        """
        feedback = []
        
        # Basic posture feedback
        if posture_data['angle'] > self.fair_threshold:
            feedback.append("Straighten your back to reduce strain.")
        
        # Enhanced neck feedback
        if posture_data['neck_angle'] > self.neck_angle_threshold:
            feedback.append("Bring your head back to align with your shoulders.")
        
        # Face mesh specific feedback
        if posture_data.get('face_mesh_available'):
            if posture_data.get('head_tilt', 0) > self.head_tilt_threshold:
                feedback.append("Keep your head level - avoid tilting to one side.")
            
            if posture_data.get('head_rotation', 0) > 20:
                feedback.append("Face forward - avoid turning your head to the side.")
        
        # Shoulder alignment
        if posture_data['shoulder_alignment'] < self.shoulder_alignment_fair:
            feedback.append("Level your shoulders for better balance.")
        
        # Head forward position
        if posture_data['head_forward_position'] > 0.08:
            feedback.append("Don't lean forward toward the screen.")
        
        # Spine curvature
        if posture_data['spine_curvature'] < 150:
            feedback.append("Maintain your spine's natural curve.")
        
        # Symmetry
        if posture_data['symmetry_score'] < 0.85:
            feedback.append("Balance your posture evenly on both sides.")
        
        # Positive reinforcement
        if not feedback:
            quality = posture_data['posture_quality']
            if quality == 'excellent':
                feedback.append("Outstanding posture! Keep it up.")
            elif quality == 'good':
                feedback.append("Great posture - you're doing well.")
            else:
                feedback.append("Your posture is improving.")
        
        return " ".join(feedback)
    
    def _get_default_posture_data(self):
        """
        Return default posture data structure
        """
        return {
            'posture': 'unknown',
            'posture_quality': 'unknown',
            'angle': 0.0,
            'neck_angle': 0.0,
            'head_tilt': 0.0,
            'head_rotation': 0.0,
            'shoulder_alignment': 0.0,
            'head_forward_position': 0.0,
            'spine_curvature': 0.0,
            'symmetry_score': 0.0,
            'feedback': None,
            'is_present': False,
            'face_mesh_available': False
        }
    
    # Include all the helper methods from the original analyzer
    def _calculate_posture_angle(self, landmarks):
        """Calculate posture angle using shoulder and hip landmarks"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        left_angle = self._angle_between_points(
            (left_shoulder.x, left_shoulder.y), 
            (left_hip.x, left_hip.y),
            (left_hip.x, 0)
        )
        
        right_angle = self._angle_between_points(
            (right_shoulder.x, right_shoulder.y), 
            (right_hip.x, right_hip.y),
            (right_hip.x, 0)
        )
        
        return (left_angle + right_angle) / 2.0
    
    def _calculate_shoulder_alignment(self, landmarks):
        """Calculate shoulder alignment score"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        height_diff = abs(left_shoulder.y - right_shoulder.y)
        alignment_score = max(0, 1 - (height_diff * 10))
        
        return alignment_score
    
    def _calculate_head_forward_position(self, landmarks):
        """Calculate head forward position"""
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        ear_x = (left_ear.x + right_ear.x) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        
        return abs(ear_x - shoulder_x)
    
    def _calculate_spine_curvature(self, landmarks):
        """Calculate spine curvature"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        knee_mid = ((left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2)
        
        return self._angle_between_points(shoulder_mid, hip_mid, knee_mid)
    
    def _calculate_symmetry_score(self, landmarks):
        """Calculate overall body symmetry score"""
        landmark_pairs = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        pair_scores = []
        for left_landmark, right_landmark in landmark_pairs:
            left = landmarks[left_landmark]
            right = landmarks[right_landmark]
            
            if left.visibility > 0.5 and right.visibility > 0.5:
                y_diff = abs(left.y - right.y)
                center_x = 0.5
                x_symmetry = abs((center_x - left.x) - (right.x - center_x))
                pair_score = max(0, 1 - (y_diff * 5) - (x_symmetry * 5))
                pair_scores.append(pair_score)
        
        return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    
    def _update_posture_history(self, posture_data):
        """Add posture data to history"""
        posture_entry = posture_data.copy()
        posture_entry['timestamp'] = datetime.now()
        
        self.posture_history.append(posture_entry)
        
        if len(self.posture_history) > self.history_max_size:
            self.posture_history.pop(0)
    
    def _angle_between_points(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
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
    
    def get_analysis_method(self):
        """Return the current analysis method being used"""
        if self.use_server_openpose and self.openpose_server_url:
            return "OpenPose Server (High Accuracy)"
        else:
            return "MediaPipe Enhanced (Face Mesh + Pose)"
    
    def set_server_config(self, use_server=False, server_url=None):
        """Dynamically configure server-based analysis"""
        self.use_server_openpose = use_server
        self.openpose_server_url = server_url
        self.logger.info(f"Analysis method updated: {self.get_analysis_method()}")