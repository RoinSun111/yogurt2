"""
Advanced Posture Detection System
Using MediaPipe Pose and Holistic models for accurate working posture classification.

This implementation provides real-time, accurate detection of:
1. Sitting postures (upright, slouched, leaning)
2. Standing postures  
3. Head position and neck alignment
4. Shoulder symmetry and alignment
5. Spine curvature analysis
"""

import logging
import mediapipe as mp
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PostureType(Enum):
    UPRIGHT_SITTING = "upright_sitting"
    SLOUCHED_SITTING = "slouched_sitting"
    LEANING_LEFT = "leaning_left"
    LEANING_RIGHT = "leaning_right"
    FORWARD_LEANING = "forward_leaning"
    STANDING_GOOD = "standing_good"
    STANDING_POOR = "standing_poor"
    LYING_DOWN = "lying_down"
    UNKNOWN = "unknown"

class PostureQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

@dataclass
class PostureMetrics:
    """Comprehensive posture measurement data"""
    spine_angle: float = 0.0
    neck_angle: float = 0.0
    shoulder_tilt: float = 0.0
    head_forward_distance: float = 0.0
    hip_shoulder_alignment: float = 0.0
    symmetry_score: float = 0.0
    stability_score: float = 0.0

@dataclass
class PostureResult:
    """Complete posture analysis result"""
    posture_type: PostureType
    quality: PostureQuality
    confidence: float
    metrics: PostureMetrics
    feedback: List[str]
    is_present: bool

class AdvancedPostureDetector:
    """
    Advanced posture detection using MediaPipe Holistic for comprehensive analysis.
    
    Features:
    - Accurate spine curvature calculation
    - Precise head and neck position tracking
    - Shoulder symmetry analysis
    - Real-time feedback generation
    - Multiple sitting/standing posture classification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Advanced Posture Detector")
        
        # Initialize MediaPipe Holistic (combines pose, face, and hands)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2,  # Use highest complexity for accuracy
            smooth_landmarks=True,
            enable_segmentation=False,  # Disable for performance
            smooth_segmentation=False,
            refine_face_landmarks=True
        )
        
        # Initialize drawing utilities for debug visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Anthropometric constants (average human proportions)
        self.HEAD_TO_NECK_RATIO = 0.15
        self.SHOULDER_WIDTH_RATIO = 0.25
        self.TORSO_HEIGHT_RATIO = 0.40
        
        # Posture classification thresholds (evidence-based)
        self.SPINE_THRESHOLDS = {
            'excellent': (170, 180),  # Nearly straight spine
            'good': (155, 170),       # Slight forward curve
            'fair': (140, 155),       # Moderate slouching
            'poor': (0, 140)          # Severe slouching
        }
        
        self.NECK_THRESHOLDS = {
            'excellent': (160, 180),  # Good neck alignment
            'good': (145, 160),       # Slight forward head
            'fair': (130, 145),       # Moderate forward head
            'poor': (0, 130)          # Severe forward head posture
        }
        
        self.SHOULDER_SYMMETRY_THRESHOLDS = {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        }
        
        # Movement smoothing
        self.landmark_history = []
        self.history_size = 5
        
        # Calibration data (can be personalized per user)
        self.calibration_data = None
        
    def detect_posture(self, image: np.ndarray) -> PostureResult:
        """
        Main method to detect and classify posture from an image.
        
        Args:
            image: RGB image array from camera
            
        Returns:
            PostureResult with comprehensive analysis
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Process with MediaPipe Holistic
            results = self.holistic.process(rgb_image)
            
            # Check if pose is detected
            if not results.pose_landmarks:
                return PostureResult(
                    posture_type=PostureType.UNKNOWN,
                    quality=PostureQuality.UNKNOWN,
                    confidence=0.0,
                    metrics=PostureMetrics(),
                    feedback=["No person detected in frame"],
                    is_present=False
                )
            
            # Extract and smooth landmarks
            landmarks = self._extract_landmarks(results)
            smoothed_landmarks = self._smooth_landmarks(landmarks)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(smoothed_landmarks, results)
            
            # Classify posture type
            posture_type, confidence = self._classify_posture(metrics, smoothed_landmarks)
            
            # Determine quality
            quality = self._assess_quality(metrics)
            
            # Generate feedback
            feedback = self._generate_feedback(posture_type, quality, metrics)
            
            return PostureResult(
                posture_type=posture_type,
                quality=quality,
                confidence=confidence,
                metrics=metrics,
                feedback=feedback,
                is_present=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in posture detection: {e}")
            return PostureResult(
                posture_type=PostureType.UNKNOWN,
                quality=PostureQuality.UNKNOWN,
                confidence=0.0,
                metrics=PostureMetrics(),
                feedback=[f"Detection error: {str(e)}"],
                is_present=False
            )
    
    def _extract_landmarks(self, results) -> Dict[str, np.ndarray]:
        """Extract key landmarks from MediaPipe results"""
        landmarks = {}
        
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            
            # Key body points
            landmarks.update({
                'nose': np.array([pose_landmarks[0].x, pose_landmarks[0].y, pose_landmarks[0].z]),
                'left_shoulder': np.array([pose_landmarks[11].x, pose_landmarks[11].y, pose_landmarks[11].z]),
                'right_shoulder': np.array([pose_landmarks[12].x, pose_landmarks[12].y, pose_landmarks[12].z]),
                'left_elbow': np.array([pose_landmarks[13].x, pose_landmarks[13].y, pose_landmarks[13].z]),
                'right_elbow': np.array([pose_landmarks[14].x, pose_landmarks[14].y, pose_landmarks[14].z]),
                'left_wrist': np.array([pose_landmarks[15].x, pose_landmarks[15].y, pose_landmarks[15].z]),
                'right_wrist': np.array([pose_landmarks[16].x, pose_landmarks[16].y, pose_landmarks[16].z]),
                'left_hip': np.array([pose_landmarks[23].x, pose_landmarks[23].y, pose_landmarks[23].z]),
                'right_hip': np.array([pose_landmarks[24].x, pose_landmarks[24].y, pose_landmarks[24].z]),
                'left_knee': np.array([pose_landmarks[25].x, pose_landmarks[25].y, pose_landmarks[25].z]),
                'right_knee': np.array([pose_landmarks[26].x, pose_landmarks[26].y, pose_landmarks[26].z]),
                'left_ankle': np.array([pose_landmarks[27].x, pose_landmarks[27].y, pose_landmarks[27].z]),
                'right_ankle': np.array([pose_landmarks[28].x, pose_landmarks[28].y, pose_landmarks[28].z]),
                'left_ear': np.array([pose_landmarks[7].x, pose_landmarks[7].y, pose_landmarks[7].z]),
                'right_ear': np.array([pose_landmarks[8].x, pose_landmarks[8].y, pose_landmarks[8].z])
            })
        
        return landmarks
    
    def _smooth_landmarks(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply temporal smoothing to reduce jitter"""
        if len(self.landmark_history) >= self.history_size:
            self.landmark_history.pop(0)
        
        self.landmark_history.append(landmarks)
        
        if len(self.landmark_history) < 3:
            return landmarks
        
        # Simple moving average smoothing
        smoothed = {}
        for key in landmarks.keys():
            if key in landmarks:
                history_points = [hist[key] for hist in self.landmark_history if key in hist]
                if history_points:
                    smoothed[key] = np.mean(history_points, axis=0)
                else:
                    smoothed[key] = landmarks[key]
        
        return smoothed
    
    def _calculate_metrics(self, landmarks: Dict[str, np.ndarray], results) -> PostureMetrics:
        """Calculate comprehensive posture metrics"""
        metrics = PostureMetrics()
        
        try:
            # Calculate spine angle (shoulder to hip alignment)
            if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
                hip_center = (landmarks['left_hip'] + landmarks['right_hip']) / 2
                
                # Spine vector
                spine_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1, 0])  # Negative Y is up in image coordinates
                
                # Calculate angle between spine and vertical
                cos_angle = np.dot(spine_vector[:2], vertical_vector[:2]) / (
                    np.linalg.norm(spine_vector[:2]) * np.linalg.norm(vertical_vector[:2])
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                metrics.spine_angle = math.degrees(math.acos(cos_angle))
            
            # Calculate neck angle using face landmarks if available
            if results.face_landmarks and 'nose' in landmarks:
                face_landmarks = results.face_landmarks.landmark
                
                # Use forehead and chin points for neck angle
                forehead = np.array([face_landmarks[10].x, face_landmarks[10].y, face_landmarks[10].z])
                chin = np.array([face_landmarks[152].x, face_landmarks[152].y, face_landmarks[152].z])
                
                # Head vector
                head_vector = forehead - chin
                vertical_vector = np.array([0, -1, 0])
                
                cos_angle = np.dot(head_vector[:2], vertical_vector[:2]) / (
                    np.linalg.norm(head_vector[:2]) * np.linalg.norm(vertical_vector[:2])
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                metrics.neck_angle = math.degrees(math.acos(cos_angle))
            else:
                # Fallback: use ear-to-shoulder angle
                if all(k in landmarks for k in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']):
                    ear_center = (landmarks['left_ear'] + landmarks['right_ear']) / 2
                    shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
                    
                    neck_vector = ear_center - shoulder_center
                    vertical_vector = np.array([0, -1, 0])
                    
                    cos_angle = np.dot(neck_vector[:2], vertical_vector[:2]) / (
                        np.linalg.norm(neck_vector[:2]) * np.linalg.norm(vertical_vector[:2])
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)
                    metrics.neck_angle = math.degrees(math.acos(cos_angle))
            
            # Calculate shoulder tilt
            if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
                shoulder_vector = landmarks['right_shoulder'] - landmarks['left_shoulder']
                horizontal_vector = np.array([1, 0, 0])
                
                cos_angle = np.dot(shoulder_vector[:2], horizontal_vector[:2]) / (
                    np.linalg.norm(shoulder_vector[:2]) * np.linalg.norm(horizontal_vector[:2])
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                metrics.shoulder_tilt = math.degrees(math.acos(cos_angle)) - 90  # Normalize to 0 degrees
            
            # Calculate head forward distance
            if all(k in landmarks for k in ['nose', 'left_shoulder', 'right_shoulder']):
                shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
                head_forward = landmarks['nose'][0] - shoulder_center[0]  # X-axis distance
                metrics.head_forward_distance = abs(head_forward)
            
            # Calculate hip-shoulder alignment
            if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
                hip_center = (landmarks['left_hip'] + landmarks['right_hip']) / 2
                alignment_offset = abs(shoulder_center[0] - hip_center[0])
                metrics.hip_shoulder_alignment = alignment_offset
            
            # Calculate symmetry score
            if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                # Compare left and right side distances
                left_shoulder_hip_dist = np.linalg.norm(landmarks['left_shoulder'] - landmarks['left_hip'])
                right_shoulder_hip_dist = np.linalg.norm(landmarks['right_shoulder'] - landmarks['right_hip'])
                
                if left_shoulder_hip_dist > 0 and right_shoulder_hip_dist > 0:
                    symmetry_ratio = min(left_shoulder_hip_dist, right_shoulder_hip_dist) / max(left_shoulder_hip_dist, right_shoulder_hip_dist)
                    metrics.symmetry_score = symmetry_ratio
            
            # Calculate stability score (based on landmark confidence)
            if results.pose_landmarks:
                visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
                metrics.stability_score = np.mean(visibilities)
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _classify_posture(self, metrics: PostureMetrics, landmarks: Dict[str, np.ndarray]) -> Tuple[PostureType, float]:
        """Classify the posture type based on calculated metrics"""
        confidence = metrics.stability_score
        
        # Check if standing (hip-knee-ankle alignment)
        if self._is_standing(landmarks):
            if metrics.spine_angle > 160 and abs(metrics.shoulder_tilt) < 10:
                return PostureType.STANDING_GOOD, confidence
            else:
                return PostureType.STANDING_POOR, confidence
        
        # Check if lying down (extreme angles)
        if metrics.spine_angle < 45 or metrics.neck_angle < 45:
            return PostureType.LYING_DOWN, confidence
        
        # Classify sitting postures
        if abs(metrics.shoulder_tilt) > 15:
            if metrics.shoulder_tilt > 0:
                return PostureType.LEANING_RIGHT, confidence
            else:
                return PostureType.LEANING_LEFT, confidence
        
        if metrics.spine_angle < 140 or metrics.head_forward_distance > 0.15:
            if metrics.spine_angle < 120:
                return PostureType.SLOUCHED_SITTING, confidence
            else:
                return PostureType.FORWARD_LEANING, confidence
        
        if metrics.spine_angle > 160 and metrics.neck_angle > 150:
            return PostureType.UPRIGHT_SITTING, confidence
        
        return PostureType.UNKNOWN, confidence * 0.5
    
    def _is_standing(self, landmarks: Dict[str, np.ndarray]) -> bool:
        """Determine if the person is standing based on body proportions"""
        try:
            if all(k in landmarks for k in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
                hip_center = (landmarks['left_hip'] + landmarks['right_hip']) / 2
                knee_center = (landmarks['left_knee'] + landmarks['right_knee']) / 2
                ankle_center = (landmarks['left_ankle'] + landmarks['right_ankle']) / 2
                
                # Calculate leg extension ratio
                hip_knee_dist = abs(hip_center[1] - knee_center[1])
                knee_ankle_dist = abs(knee_center[1] - ankle_center[1])
                
                # Standing typically has more extended legs
                if hip_knee_dist > 0 and knee_ankle_dist > 0:
                    leg_extension_ratio = (hip_knee_dist + knee_ankle_dist) / hip_knee_dist
                    return leg_extension_ratio > 1.8  # Threshold for standing detection
        except:
            pass
        
        return False
    
    def _assess_quality(self, metrics: PostureMetrics) -> PostureQuality:
        """Assess overall posture quality"""
        scores = []
        
        # Spine angle score
        for quality, (min_angle, max_angle) in self.SPINE_THRESHOLDS.items():
            if min_angle <= metrics.spine_angle <= max_angle:
                scores.append(quality)
                break
        
        # Neck angle score
        for quality, (min_angle, max_angle) in self.NECK_THRESHOLDS.items():
            if min_angle <= metrics.neck_angle <= max_angle:
                scores.append(quality)
                break
        
        # Symmetry score
        if metrics.symmetry_score >= self.SHOULDER_SYMMETRY_THRESHOLDS['excellent']:
            scores.append('excellent')
        elif metrics.symmetry_score >= self.SHOULDER_SYMMETRY_THRESHOLDS['good']:
            scores.append('good')
        elif metrics.symmetry_score >= self.SHOULDER_SYMMETRY_THRESHOLDS['fair']:
            scores.append('fair')
        else:
            scores.append('poor')
        
        # Return the worst quality (most conservative assessment)
        quality_order = ['excellent', 'good', 'fair', 'poor']
        for quality in reversed(quality_order):
            if quality in scores:
                return PostureQuality(quality)
        
        return PostureQuality.UNKNOWN
    
    def _generate_feedback(self, posture_type: PostureType, quality: PostureQuality, metrics: PostureMetrics) -> List[str]:
        """Generate specific feedback based on posture analysis"""
        feedback = []
        
        # Posture-specific feedback
        if posture_type == PostureType.SLOUCHED_SITTING:
            feedback.append("Sit up straighter - straighten your back against the chair")
            feedback.append("Pull your shoulders back and down")
        elif posture_type == PostureType.FORWARD_LEANING:
            feedback.append("Lean back in your chair - avoid hunching forward")
            feedback.append("Adjust screen height to reduce forward head posture")
        elif posture_type == PostureType.LEANING_LEFT:
            feedback.append("Center your body - you're leaning to the left")
            feedback.append("Distribute weight evenly on both hips")
        elif posture_type == PostureType.LEANING_RIGHT:
            feedback.append("Center your body - you're leaning to the right")
            feedback.append("Distribute weight evenly on both hips")
        elif posture_type == PostureType.STANDING_GOOD:
            feedback.append("Great standing posture! Keep it up")
        elif posture_type == PostureType.STANDING_POOR:
            feedback.append("Improve standing posture - straighten your back")
            feedback.append("Keep shoulders level and relaxed")
        elif posture_type == PostureType.UPRIGHT_SITTING:
            feedback.append("Excellent sitting posture!")
        
        # Neck-specific feedback
        if metrics.neck_angle < 140:
            feedback.append("Lift your head up - avoid looking down too much")
            feedback.append("Raise monitor to eye level to reduce neck strain")
        elif metrics.neck_angle < 150:
            feedback.append("Slightly lift your head for better neck alignment")
        
        # Shoulder-specific feedback
        if abs(metrics.shoulder_tilt) > 10:
            feedback.append("Level your shoulders - avoid tilting to one side")
        
        # Head position feedback
        if metrics.head_forward_distance > 0.12:
            feedback.append("Pull your head back - avoid forward head posture")
            feedback.append("Imagine a string pulling the top of your head upward")
        
        # Symmetry feedback
        if metrics.symmetry_score < 0.8:
            feedback.append("Maintain body symmetry - keep both sides balanced")
        
        # Quality-based general advice
        if quality == PostureQuality.POOR:
            feedback.append("Take a posture break - stand and stretch")
            feedback.append("Consider adjusting your workspace ergonomics")
        elif quality == PostureQuality.FAIR:
            feedback.append("Small adjustments needed for better posture")
        
        return feedback[:4]  # Limit to 4 most important pieces of feedback
    
    def get_posture_score(self, metrics: PostureMetrics) -> float:
        """Calculate a numerical posture score (0-100)"""
        scores = []
        
        # Spine alignment score (0-30 points)
        if metrics.spine_angle >= 170:
            scores.append(30)
        elif metrics.spine_angle >= 155:
            scores.append(25)
        elif metrics.spine_angle >= 140:
            scores.append(15)
        else:
            scores.append(5)
        
        # Neck alignment score (0-25 points)
        if metrics.neck_angle >= 160:
            scores.append(25)
        elif metrics.neck_angle >= 145:
            scores.append(20)
        elif metrics.neck_angle >= 130:
            scores.append(10)
        else:
            scores.append(2)
        
        # Symmetry score (0-25 points)
        scores.append(metrics.symmetry_score * 25)
        
        # Shoulder alignment score (0-20 points)
        if abs(metrics.shoulder_tilt) <= 5:
            scores.append(20)
        elif abs(metrics.shoulder_tilt) <= 10:
            scores.append(15)
        elif abs(metrics.shoulder_tilt) <= 15:
            scores.append(8)
        else:
            scores.append(2)
        
        return min(100, sum(scores))
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()