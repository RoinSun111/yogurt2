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
            
            self.logger.debug(f"Detected: {posture_type} (sitting: {is_sitting}, spine: {spine_angle:.1f}°, shoulder_tilt: {shoulder_tilt:.1f}°, body_lean: {body_lean:.3f}, lateral: {lateral_lean:.3f})")
            
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
            
            # In a typical webcam view, if someone is sitting at a desk:
            # - Their shoulders will be higher in the frame (lower y value)
            # - Hips will be visible but lower
            # - Knees may be partially visible or at frame edge
            
            # Key indicator: body proportions and positioning
            torso_height = abs(shoulder_y - hip_y)
            
            # More relaxed sitting detection - focus on key indicators
            # 1. Torso is visible and prominent in frame
            prominent_torso = shoulder_y < 0.5  # Shoulders in upper half of frame
            
            # 2. Hip position is reasonable for sitting
            hip_position_ok = 0.2 < hip_y < 0.9  # More lenient hip positioning
            
            # 3. Good torso visibility
            good_torso_proportion = torso_height > 0.12  # More lenient torso requirement
            
            # 4. Body composition suggests sitting (more compact)
            total_visible_body = abs(shoulder_y - knee_y)
            compact_body = total_visible_body < 0.8  # Person appears more compact
            
            # Consider sitting if multiple indicators match (more lenient)
            sitting_indicators = [
                prominent_torso,
                hip_position_ok, 
                good_torso_proportion,
                compact_body
            ]
            
            return sum(sitting_indicators) >= 2  # Only need 2 out of 4 indicators
            
        except Exception:
            return False
    
    def _get_spine_angle(self, landmarks) -> float:
        """Calculate spine angle from vertical (0 = perfectly upright)"""
        try:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            shoulder_mid_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            hip_mid_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
            
            # Calculate the vector from hip to shoulder
            dx = shoulder_mid_x - hip_mid_x
            dy = hip_mid_y - shoulder_mid_y  # Note: y coordinates are inverted (smaller y = higher)
            
            # Calculate angle from vertical (90 degrees minus the angle from horizontal)
            if abs(dy) < 0.001:  # Avoid division by zero
                return 90.0  # Completely horizontal
            
            # Angle from horizontal
            angle_from_horizontal = math.degrees(math.atan2(abs(dx), abs(dy)))
            
            # Convert to angle from vertical (0 = perfectly upright, 90 = horizontal)
            return angle_from_horizontal
            
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
        
        # Lying detection: spine should be very horizontal (close to 90 degrees from vertical)
        if spine_angle > 70:  # Very horizontal spine indicates lying
            return "lying", 0.9
        
        # Standing postures
        if not is_sitting:
            if spine_angle > 15:
                return "standing", 0.8
            else:
                return "standing", 0.9
        
        # Sitting postures - focus on practical distinctions
        
        # Strong lateral lean (side sitting)
        if abs(lateral_lean) > 0.15:  # Even more restrictive
            if lateral_lean < -0.15:
                return "left_sitting", 0.85
            else:
                return "right_sitting", 0.85
        
        # Forward postures (hunching and leaning)
        if body_lean > 0.25:  # Very restrictive for forward lean
            if spine_angle > 40:  # Very high threshold for hunching
                return "hunching_over", 0.9
            else:
                return "leaning_forward", 0.85
        
        # Moderate poor posture
        if spine_angle > 35 or body_lean > 0.20:
            return "hunching_over", 0.8
        
        # Good sitting posture (quite lenient)
        if spine_angle <= 25 and abs(lateral_lean) <= 0.10:
            return "sitting_straight", 0.95
        
        # Default to sitting straight for normal positions
        return "sitting_straight", 0.8
    
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