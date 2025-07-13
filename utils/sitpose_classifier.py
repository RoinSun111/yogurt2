"""
SitPose Posture Classifier
Accurate detection of 7 different working postures using MediaPipe landmarks.

Posture Types:
1. Sitting straight
2. Hunching over
3. Lying
4. Leaning forward
5. Left sitting
6. Right sitting
7. Standing
"""

import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import math

class SitPoseType(Enum):
    """Seven specific working posture types"""
    SITTING_STRAIGHT = "sitting_straight"
    HUNCHING_OVER = "hunching_over"
    LYING = "lying"
    LEANING_FORWARD = "leaning_forward"
    LEFT_SITTING = "left_sitting"
    RIGHT_SITTING = "right_sitting"
    STANDING = "standing"
    UNKNOWN = "unknown"

@dataclass
class PostureAnalysis:
    """Complete posture analysis result"""
    posture_type: str
    confidence: float
    spine_angle: float
    shoulder_tilt: float
    head_position: str
    is_sitting: bool
    feedback: str

class SitPoseClassifier:
    """
    Classifies working posture into 7 specific types based on body landmarks.
    Uses geometric analysis of spine, shoulders, and overall body position.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing SitPose Classifier")
        
        # MediaPipe pose landmarks
        self.mp_pose = mp.solutions.pose
        
        # Key landmark indices
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        
    def classify_posture(self, landmarks) -> PostureAnalysis:
        """
        Classify posture based on MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            PostureAnalysis with classified posture and metrics
        """
        if not landmarks:
            return PostureAnalysis(
                posture_type="unknown",
                confidence=0.0,
                spine_angle=0.0,
                shoulder_tilt=0.0,
                head_position="unknown",
                is_sitting=False,
                feedback="No person detected"
            )
        
        # Extract key points
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        right_knee = landmarks[self.RIGHT_KNEE]
        nose = landmarks[self.NOSE]
        
        # Calculate body metrics
        spine_angle = self._calculate_spine_angle(landmarks)
        shoulder_tilt = self._calculate_shoulder_tilt(left_shoulder, right_shoulder)
        hip_knee_angle = self._calculate_hip_knee_angle(landmarks)
        body_lean = self._calculate_body_lean(landmarks)
        lateral_lean = self._calculate_lateral_lean(landmarks)
        
        # Determine if sitting or standing
        is_sitting = self._is_sitting(hip_knee_angle, landmarks)
        
        # Classify posture
        posture_type, confidence = self._classify_posture_type(
            is_sitting, spine_angle, shoulder_tilt, body_lean, lateral_lean, hip_knee_angle, landmarks
        )
        
        # Generate feedback
        feedback = self._generate_feedback(posture_type, spine_angle, shoulder_tilt)
        
        # Determine head position
        head_position = self._get_head_position(nose, left_shoulder, right_shoulder)
        
        return PostureAnalysis(
            posture_type=posture_type,
            confidence=confidence,
            spine_angle=spine_angle,
            shoulder_tilt=shoulder_tilt,
            head_position=head_position,
            is_sitting=is_sitting,
            feedback=feedback
        )
    
    def _calculate_spine_angle(self, landmarks) -> float:
        """Calculate spine angle from vertical"""
        # Get midpoints
        shoulder_mid = self._get_midpoint(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.RIGHT_SHOULDER]
        )
        hip_mid = self._get_midpoint(
            landmarks[self.LEFT_HIP],
            landmarks[self.RIGHT_HIP]
        )
        
        # Calculate angle from vertical
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        angle = math.degrees(math.atan2(dx, dy))
        
        return abs(angle)
    
    def _calculate_shoulder_tilt(self, left_shoulder, right_shoulder) -> float:
        """Calculate shoulder tilt angle"""
        dy = right_shoulder.y - left_shoulder.y
        dx = right_shoulder.x - left_shoulder.x
        return math.degrees(math.atan2(dy, dx))
    
    def _calculate_hip_knee_angle(self, landmarks) -> float:
        """Calculate angle between hip and knee (for sitting detection)"""
        left_hip = landmarks[self.LEFT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        left_ankle = landmarks[self.LEFT_ANKLE]
        
        # Calculate angle at knee
        angle = self._calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )
        
        return angle
    
    def _calculate_body_lean(self, landmarks) -> float:
        """Calculate forward/backward body lean"""
        nose = landmarks[self.NOSE]
        hip_mid = self._get_midpoint(
            landmarks[self.LEFT_HIP],
            landmarks[self.RIGHT_HIP]
        )
        
        # Forward lean is positive, backward is negative
        lean = nose.x - hip_mid[0]
        return lean
    
    def _calculate_lateral_lean(self, landmarks) -> float:
        """Calculate left/right body lean"""
        shoulder_mid = self._get_midpoint(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.RIGHT_SHOULDER]
        )
        hip_mid = self._get_midpoint(
            landmarks[self.LEFT_HIP],
            landmarks[self.RIGHT_HIP]
        )
        
        # Left lean is negative, right is positive
        lean = shoulder_mid[0] - hip_mid[0]
        return lean
    
    def _is_sitting(self, hip_knee_angle: float, landmarks) -> bool:
        """Determine if person is sitting based on hip-knee angle and hip position"""
        # Sitting typically has hip-knee angle < 120 degrees
        # Also check if hips are lower than expected for standing
        
        hip_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
        knee_y = (landmarks[self.LEFT_KNEE].y + landmarks[self.RIGHT_KNEE].y) / 2
        
        # In sitting, hips and knees are at similar height
        hip_knee_y_diff = abs(hip_y - knee_y)
        
        return hip_knee_angle < 120 and hip_knee_y_diff < 0.15
    
    def _classify_posture_type(self, is_sitting: bool, spine_angle: float, 
                              shoulder_tilt: float, body_lean: float, 
                              lateral_lean: float, hip_knee_angle: float,
                              landmarks) -> Tuple[str, float]:
        """
        Classify specific posture type based on metrics.
        
        Returns:
            Tuple of (posture_type, confidence)
        """
        confidence = 0.9  # Base confidence
        
        # Check for lying down first (most distinctive)
        if self._is_lying(landmarks):
            return SitPoseType.LYING.value, 0.95
        
        # Standing vs sitting classification
        if not is_sitting:
            return SitPoseType.STANDING.value, confidence
        
        # Sitting posture classification
        
        # Left/Right sitting (lateral lean)
        if abs(lateral_lean) > 0.05:  # Significant lateral lean
            if lateral_lean < 0:
                return SitPoseType.LEFT_SITTING.value, confidence
            else:
                return SitPoseType.RIGHT_SITTING.value, confidence
        
        # Forward leaning vs hunching
        if body_lean > 0.1:  # Significant forward lean
            if spine_angle > 25:  # Large spine curvature
                return SitPoseType.HUNCHING_OVER.value, confidence
            else:
                return SitPoseType.LEANING_FORWARD.value, confidence
        
        # Sitting straight (default good sitting posture)
        if spine_angle < 15 and abs(shoulder_tilt) < 5:
            return SitPoseType.SITTING_STRAIGHT.value, 0.95
        
        # Default to hunching if spine angle is bad
        if spine_angle > 20:
            return SitPoseType.HUNCHING_OVER.value, 0.85
        
        return SitPoseType.SITTING_STRAIGHT.value, 0.8
    
    def _is_lying(self, landmarks) -> bool:
        """Check if person is lying down"""
        # In lying position, shoulders and hips are at similar y-coordinate
        shoulder_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
        hip_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
        
        # Also check if body is more horizontal than vertical
        shoulder_x_diff = abs(landmarks[self.LEFT_SHOULDER].x - landmarks[self.RIGHT_SHOULDER].x)
        shoulder_y_diff = abs(landmarks[self.LEFT_SHOULDER].y - landmarks[self.RIGHT_SHOULDER].y)
        
        is_horizontal = shoulder_x_diff > shoulder_y_diff * 2
        similar_height = abs(shoulder_y - hip_y) < 0.1
        
        return is_horizontal and similar_height
    
    def _generate_feedback(self, posture_type: str, spine_angle: float, shoulder_tilt: float) -> str:
        """Generate posture-specific feedback"""
        if posture_type == SitPoseType.SITTING_STRAIGHT.value:
            return "Great posture! Keep your back straight and shoulders relaxed."
        elif posture_type == SitPoseType.HUNCHING_OVER.value:
            return "You're hunching over. Sit back, straighten your spine, and pull your shoulders back."
        elif posture_type == SitPoseType.LEANING_FORWARD.value:
            return "You're leaning forward. Try to sit back and maintain a neutral spine position."
        elif posture_type == SitPoseType.LEFT_SITTING.value:
            return "You're leaning to the left. Center your weight and sit evenly on both sides."
        elif posture_type == SitPoseType.RIGHT_SITTING.value:
            return "You're leaning to the right. Center your weight and sit evenly on both sides."
        elif posture_type == SitPoseType.LYING.value:
            return "You appear to be lying down. Consider sitting up for better work posture."
        elif posture_type == SitPoseType.STANDING.value:
            if spine_angle > 10:
                return "Standing posture could be improved. Stand tall with shoulders back."
            return "Good standing posture! Keep your weight balanced and spine neutral."
        else:
            return "Unable to determine posture. Ensure you're visible to the camera."
    
    def _get_head_position(self, nose, left_shoulder, right_shoulder) -> str:
        """Determine head position relative to shoulders"""
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        
        if nose.x < shoulder_mid_x - 0.05:
            return "left"
        elif nose.x > shoulder_mid_x + 0.05:
            return "right"
        else:
            return "center"
    
    def _get_midpoint(self, point1, point2) -> Tuple[float, float]:
        """Calculate midpoint between two landmarks"""
        return ((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)
    
    def _calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle ABC (angle at point B)"""
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        
        cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
            math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
        )
        
        angle = math.acos(np.clip(cosine_angle, -1.0, 1.0))
        return math.degrees(angle)