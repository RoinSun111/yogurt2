"""
Clean MediaPipe-based posture detection system
Accurately detects 7 working postures using head-to-shoulder angle analysis
"""

import logging
import math
import numpy as np
import mediapipe as mp
from datetime import datetime


class PostureDetector:
    """
    Real-time posture detection using MediaPipe pose landmarks
    Detects: sitting_straight, hunching_over, left_sitting, right_sitting, leaning_forward, lying, standing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("PostureDetector initialized for real-time detection")
        
        # MediaPipe pose setup optimized for real-time processing
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.3    # Lower for continuous tracking
        )
        
        # Refined thresholds for stable detection (hip logic removed)
        self.thresholds = {
            'lying_head_angle': 75,       # head-shoulder angle > 75° = lying 
            'hunching_head_angle': 30,    # head forward angle > 30° = hunching
            'lean_forward_head': 20,      # head angle > 20° forward = leaning forward
            'lateral_lean_head': 15,      # head tilt > 15° = side sitting
            'standing_spine_angle': 25    # spine angle < 25° = standing (new logic)
        }
        
        # Stability tracking for consistent detection
        self.recent_detections = []
        self.stability_window = 3  # Number of frames to consider for stability
        
        # MediaPipe landmark indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
    
    def detect_posture(self, image):
        """
        Detect posture from RGB image
        Returns: dict with posture type, angles, confidence, and feedback
        """
        try:
            # Process image with MediaPipe
            results = self.pose.process(image)
            
            if not results.pose_landmarks:
                return self._no_person_detected()
            
            landmarks = results.pose_landmarks.landmark
            
            # Calculate key metrics
            metrics = self._calculate_metrics(landmarks)
            
            # Classify posture based on metrics
            raw_posture = self._classify_posture(metrics)
            
            # Apply stability filtering
            posture_type = self._apply_stability_filter(raw_posture)
            
            # Generate feedback
            feedback = self._generate_feedback(posture_type, metrics)
            
            # Determine quality
            quality = self._determine_quality(metrics['spine_angle'])
            
            return {
                'posture': posture_type,
                'posture_quality': quality,
                'angle': metrics['spine_angle'],
                'head_to_shoulder_angle': metrics['head_tilt_angle'],  # Main detection metric
                'head_forward_angle': metrics['head_forward_angle'],   # Forward lean metric  
                'neck_angle': metrics['neck_angle'],
                'shoulder_alignment': metrics['shoulder_alignment'],
                'head_forward_position': metrics['head_forward'],
                'spine_curvature': 180 - metrics['spine_angle'],  # For compatibility
                'symmetry_score': metrics['shoulder_alignment'],
                'feedback': feedback,
                'is_present': True,
                'confidence': metrics['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Error in posture detection: {e}")
            return self._error_response()
    
    def _calculate_metrics(self, landmarks):
        """Calculate posture metrics using only head and shoulder landmarks"""
        
        # Get key landmark positions (no hip/knee needed)
        nose = np.array([landmarks[self.NOSE].x, landmarks[self.NOSE].y])
        left_shoulder = np.array([landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y])
        
        # Calculate shoulder midpoint
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        
        # 1. Shoulder angle (shoulder line vs horizontal - indicates standing/sitting)
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_angle = abs(math.degrees(math.atan2(shoulder_vector[1], shoulder_vector[0])))
        
        # 2. Head to shoulder angle (primary metric for posture detection)
        head_shoulder_vector = nose - shoulder_mid
        # Calculate tilt angle - how much head deviates from center
        head_tilt_angle = math.degrees(math.atan2(head_shoulder_vector[0], abs(head_shoulder_vector[1]) + 0.001))
        
        # 3. Head forward/backward angle 
        head_forward_angle = math.degrees(math.atan2(abs(head_shoulder_vector[1]), abs(head_shoulder_vector[0]) + 0.001))
        
        # 4. Neck angle (vertical alignment)
        neck_vector = nose - shoulder_mid
        neck_angle = abs(math.degrees(math.atan2(neck_vector[0], neck_vector[1] + 0.001)))
        
        # 5. Shoulder alignment (how level shoulders are)
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        shoulder_alignment = max(0, 1 - (shoulder_diff * 10))  # 0-1 score
        
        # 6. Head forward position
        head_forward = abs(nose[0] - shoulder_mid[0])
        
        # 7. Confidence based on landmark visibility and consistency
        confidence = self._calculate_confidence(landmarks)
        
        return {
            'spine_angle': shoulder_angle,           # Now based on shoulder line
            'head_tilt_angle': head_tilt_angle,      # Key metric for left/right detection
            'head_forward_angle': head_forward_angle, # Key metric for forward lean
            'neck_angle': neck_angle,
            'shoulder_alignment': shoulder_alignment,
            'head_forward': head_forward,
            'confidence': confidence
        }
    
    def _classify_posture(self, metrics):
        """Classify posture using head-to-shoulder angle analysis (no hip logic)"""
        
        shoulder_angle = metrics['spine_angle']  # Now represents shoulder line angle
        head_tilt = metrics['head_tilt_angle']
        head_forward = metrics['head_forward_angle']
        
        # Debug logging for angle analysis (removed hip-knee)
        self.logger.debug(f"Angles - Shoulder: {shoulder_angle:.1f}°, Head Tilt: {head_tilt:.1f}°, Head Forward: {head_forward:.1f}°")
        
        # 1. Check for lying down (head very tilted indicates lying)
        if abs(head_tilt) > self.thresholds['lying_head_angle']:
            return 'lying'
        
        # 2. Check for left/right head tilt FIRST (prioritize head position)
        if abs(head_tilt) > self.thresholds['lateral_lean_head']:
            if head_tilt < 0:
                return 'left_sitting' 
            else:
                return 'right_sitting'
        
        # 3. Check for standing (based on shoulder line angle)
        if shoulder_angle < self.thresholds['standing_spine_angle']:
            return 'standing'
        
        # 4. Check for hunching (forward head posture)
        if head_forward > self.thresholds['hunching_head_angle']:
            return 'hunching_over'
        
        # 5. Check for leaning forward (moderate forward head angle)
        if head_forward > self.thresholds['lean_forward_head']:
            return 'leaning_forward'
        
        # 6. Default to sitting straight (good alignment)
        return 'sitting_straight'
    
    def _calculate_confidence(self, landmarks):
        """Calculate confidence score based on landmark quality (head/shoulders only)"""
        try:
            # Check visibility of key landmarks (removed hip/knee)
            key_landmarks = [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER]
            
            visible_count = 0
            total_visibility = 0
            
            for idx in key_landmarks:
                if hasattr(landmarks[idx], 'visibility'):
                    visibility = landmarks[idx].visibility
                    if visibility > 0.5:
                        visible_count += 1
                    total_visibility += visibility
                else:
                    visible_count += 1  # Assume visible if no visibility score
                    total_visibility += 1.0
            
            # Calculate confidence (0.0 to 1.0)
            confidence = (visible_count / len(key_landmarks)) * (total_visibility / len(key_landmarks))
            return min(1.0, max(0.0, confidence))
            
        except:
            return 0.8  # Default confidence
    
    def _determine_quality(self, spine_angle):
        """Determine posture quality based on spine angle"""
        if spine_angle <= 10:
            return 'excellent'
        elif spine_angle <= 15:
            return 'good'
        elif spine_angle <= 25:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_feedback(self, posture_type, metrics):
        """Generate specific feedback based on posture and metrics"""
        head_tilt = metrics.get('head_tilt_angle', 0)
        head_forward = metrics.get('head_forward_angle', 0)
        
        feedback_map = {
            'sitting_straight': f'Excellent posture! Head tilt: {head_tilt:.1f}°',
            'hunching_over': f'Hunching detected. Head forward: {head_forward:.1f}°. Straighten up!',
            'leaning_forward': f'Leaning forward. Head angle: {head_forward:.1f}°. Sit back straight.',
            'left_sitting': f'Head tilted left {abs(head_tilt):.1f}°. Center yourself.',
            'right_sitting': f'Head tilted right {head_tilt:.1f}°. Center yourself.',
            'lying': f'Lying detected. Head tilt: {abs(head_tilt):.1f}°. Sit up properly.',
            'standing': f'Standing detected. Good break! Head tilt: {head_tilt:.1f}°',
            'unknown': 'Unable to detect posture clearly.'
        }
        
        return feedback_map.get(posture_type, 'Monitor your posture regularly.')
    
    def _apply_stability_filter(self, raw_posture):
        """Apply stability filtering to reduce rapid posture changes"""
        
        # Add current detection to recent detections
        self.recent_detections.append(raw_posture)
        
        # Keep only the last N detections
        if len(self.recent_detections) > self.stability_window:
            self.recent_detections.pop(0)
        
        # If we don't have enough data yet, return the raw detection
        if len(self.recent_detections) < self.stability_window:
            return raw_posture
        
        # Count occurrences of each posture in recent detections
        posture_counts = {}
        for posture in self.recent_detections:
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
        
        # Find the most common posture
        most_common_posture = max(posture_counts, key=posture_counts.get)
        most_common_count = posture_counts[most_common_posture]
        
        # Require majority consensus for posture change
        if most_common_count >= 2:  # At least 2 out of 3 frames agree
            return most_common_posture
        else:
            # No consensus, return the previous stable detection or current
            return self.recent_detections[-2] if len(self.recent_detections) >= 2 else raw_posture
    
    def _no_person_detected(self):
        """Return response when no person is detected"""
        return {
            'posture': 'unknown',
            'posture_quality': 'unknown',
            'angle': 160,
            'head_to_shoulder_angle': 0,
            'neck_angle': 0,
            'shoulder_alignment': 0,
            'head_forward_position': 0,
            'spine_curvature': 180,
            'symmetry_score': 0,
            'feedback': 'No person detected in frame',
            'is_present': False,
            'confidence': 0.0
        }
    
    def _error_response(self):
        """Return response when an error occurs"""
        return {
            'posture': 'unknown',
            'posture_quality': 'unknown',
            'angle': 160,
            'head_to_shoulder_angle': 0,
            'neck_angle': 0,
            'shoulder_alignment': 0,
            'head_forward_position': 0,
            'spine_curvature': 180,
            'symmetry_score': 0,
            'feedback': 'Error analyzing posture',
            'is_present': False,
            'confidence': 0.0
        }