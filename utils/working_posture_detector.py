"""
Working Posture Detector - Simplified and accurate detection for 7 working postures
Optimized for real-world desk scenarios with more practical thresholds.
"""

import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math

@dataclass
class PostureResult:
    """Simplified posture analysis result"""
    posture_type: str
    confidence: float
    spine_angle: float
    shoulder_tilt: float
    head_position: str
    is_sitting: bool
    feedback: str

class WorkingPostureDetector:
    """
    Simplified posture detector focused on practical working scenarios.
    Detects: sitting_straight, hunching_over, leaning_forward, left_sitting, 
    right_sitting, lying, standing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Working Posture Detector initialized")
        
        # MediaPipe pose landmarks indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        
    def detect_posture(self, landmarks) -> PostureResult:
        """
        Main posture detection method with simplified logic
        """
        if not landmarks:
            return PostureResult(
                posture_type="unknown",
                confidence=0.0,
                spine_angle=0.0,
                shoulder_tilt=0.0,
                head_position="unknown",
                is_sitting=False,
                feedback="No person detected"
            )
        
        try:
            # Calculate basic metrics
            spine_angle = self._get_spine_angle(landmarks)
            shoulder_tilt = self._get_shoulder_tilt(landmarks)
            body_lean = self._get_body_lean(landmarks)
            lateral_lean = self._get_lateral_lean(landmarks)
            
            # Simple sitting vs standing detection
            is_sitting = self._detect_sitting(landmarks)
            
            # Classify posture type
            posture_type, confidence = self._classify_posture(
                is_sitting, spine_angle, shoulder_tilt, body_lean, lateral_lean
            )
            
            # Generate feedback
            feedback = self._get_feedback(posture_type, spine_angle)
            
            # Head position
            head_position = self._get_head_position(landmarks)
            
            self.logger.debug(f"Detected: {posture_type} (sitting: {is_sitting}, spine: {spine_angle:.1f}°)")
            
            return PostureResult(
                posture_type=posture_type,
                confidence=confidence,
                spine_angle=spine_angle,
                shoulder_tilt=shoulder_tilt,
                head_position=head_position,
                is_sitting=is_sitting,
                feedback=feedback
            )
            
        except Exception as e:
            self.logger.error(f"Error in posture detection: {e}")
            return PostureResult(
                posture_type="unknown",
                confidence=0.0,
                spine_angle=0.0,
                shoulder_tilt=0.0,
                head_position="unknown",
                is_sitting=False,
                feedback="Error detecting posture"
            )
    
    def _detect_sitting(self, landmarks) -> bool:
        """Simplified sitting detection based on body proportions"""
        try:
            # Get key positions
            shoulder_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
            hip_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
            knee_y = (landmarks[self.LEFT_KNEE].y + landmarks[self.RIGHT_KNEE].y) / 2
            
            # Key indicator: in sitting, torso is more prominent in frame
            # and legs are less visible/compressed
            torso_height = abs(shoulder_y - hip_y)
            leg_height = abs(hip_y - knee_y)
            
            # In sitting: torso is more visible, legs are compressed
            if torso_height > 0 and leg_height >= 0:
                torso_to_leg_ratio = torso_height / (leg_height + 0.1)  # Avoid division by zero
                
                # Sitting indicators:
                # 1. High torso-to-leg ratio (legs compressed when sitting)
                sitting_ratio = torso_to_leg_ratio > 1.5
                
                # 2. Knees are close to or below hips (typical sitting posture)
                knee_position = knee_y >= hip_y - 0.1
                
                # 3. Body appears more compact vertically
                total_body = abs(shoulder_y - knee_y)
                is_compact = total_body < 0.7  # Normalized coordinate threshold
                
                # Consider sitting if multiple indicators match
                return sum([sitting_ratio, knee_position, is_compact]) >= 2
            
            return False
            
        except Exception:
            return False
    
    def _get_spine_angle(self, landmarks) -> float:
        """Calculate spine angle from vertical"""
        try:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            shoulder_mid_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            hip_mid_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
            
            # Calculate angle from vertical
            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y
            
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            return angle
            
        except Exception:
            return 0.0
    
    def _get_shoulder_tilt(self, landmarks) -> float:
        """Calculate shoulder tilt angle"""
        try:
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            dy = right_shoulder.y - left_shoulder.y
            dx = right_shoulder.x - left_shoulder.x
            
            return math.degrees(math.atan2(dy, dx))
            
        except Exception:
            return 0.0
    
    def _get_body_lean(self, landmarks) -> float:
        """Calculate forward/backward body lean"""
        try:
            nose = landmarks[self.NOSE]
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            
            # Positive = forward lean, negative = backward lean
            return nose.x - hip_mid_x
            
        except Exception:
            return 0.0
    
    def _get_lateral_lean(self, landmarks) -> float:
        """Calculate left/right body lean"""
        try:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            
            # Positive = right lean, negative = left lean
            return shoulder_mid_x - hip_mid_x
            
        except Exception:
            return 0.0
    
    def _classify_posture(self, is_sitting: bool, spine_angle: float, 
                         shoulder_tilt: float, body_lean: float, lateral_lean: float) -> Tuple[str, float]:
        """Classify posture with practical thresholds"""
        
        # Check for lying (extreme horizontal position)
        if abs(shoulder_tilt) > 60:  # Very tilted shoulders indicate lying
            return "lying", 0.9
        
        # Standing postures
        if not is_sitting:
            if spine_angle > 15:
                return "standing", 0.8
            else:
                return "standing", 0.9
        
        # Sitting postures - use more practical thresholds
        
        # Strong lateral lean
        if abs(lateral_lean) > 0.08:  # Increased sensitivity
            if lateral_lean < -0.08:
                return "left_sitting", 0.85
            else:
                return "right_sitting", 0.85
        
        # Forward postures
        if body_lean > 0.15:  # Strong forward lean
            if spine_angle > 30:  # Poor posture
                return "hunching_over", 0.9
            else:
                return "leaning_forward", 0.85
        
        # Moderate forward lean or poor spine angle
        if spine_angle > 25 or body_lean > 0.08:
            return "hunching_over", 0.8
        
        # Good sitting posture
        if spine_angle <= 15 and abs(lateral_lean) <= 0.05:
            return "sitting_straight", 0.95
        
        # Default moderate posture
        return "sitting_straight", 0.7
    
    def _get_feedback(self, posture_type: str, spine_angle: float) -> str:
        """Generate feedback based on posture"""
        feedback_map = {
            "sitting_straight": "Excellent posture! Keep your back straight and shoulders relaxed.",
            "hunching_over": "You're hunching over. Sit back, straighten your spine, and relax your shoulders.",
            "leaning_forward": "Leaning forward detected. Try to sit back and maintain an upright position.",
            "left_sitting": "You're leaning to the left. Center your weight evenly on both sides.",
            "right_sitting": "You're leaning to the right. Center your weight evenly on both sides.",
            "lying": "Lying position detected. Consider sitting up for better work posture.",
            "standing": "Good standing posture!" if spine_angle <= 10 else "Standing detected. Keep your back straight and shoulders back.",
            "unknown": "Unable to determine posture. Ensure you're clearly visible to the camera."
        }
        
        return feedback_map.get(posture_type, "Posture analysis complete.")
    
    def _get_head_position(self, landmarks) -> str:
        """Determine head position relative to shoulders"""
        try:
            nose = landmarks[self.NOSE]
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            
            offset = nose.x - shoulder_mid_x
            
            if offset < -0.05:
                return "left"
            elif offset > 0.05:
                return "right"
            else:
                return "center"
                
        except Exception:
            return "center"