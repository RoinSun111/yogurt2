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
        self.logger.info("PostureDetector initialized")
        
        # MediaPipe pose setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Posture classification thresholds (in degrees)
        self.thresholds = {
            'lying_spine_angle': 60,      # spine angle > 60° = lying down
            'hunching_spine_angle': 25,   # spine angle > 25° = hunching
            'lean_forward_head': 15,      # head angle > 15° forward = leaning forward
            'lateral_lean': 0.08,         # lateral position difference > 8% = side sitting
            'standing_hip_knee': 0.15     # hip-knee distance > 15% = standing
        }
        
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
            posture_type = self._classify_posture(metrics)
            
            # Generate feedback
            feedback = self._generate_feedback(posture_type, metrics)
            
            # Determine quality
            quality = self._determine_quality(metrics['spine_angle'])
            
            return {
                'posture': posture_type,
                'posture_quality': quality,
                'angle': metrics['spine_angle'],
                'head_to_shoulder_angle': metrics['head_shoulder_angle'],
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
        """Calculate all posture-related metrics from landmarks"""
        
        # Get key landmark positions
        nose = np.array([landmarks[self.NOSE].x, landmarks[self.NOSE].y])
        left_shoulder = np.array([landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y])
        left_hip = np.array([landmarks[self.LEFT_HIP].x, landmarks[self.LEFT_HIP].y])
        right_hip = np.array([landmarks[self.RIGHT_HIP].x, landmarks[self.RIGHT_HIP].y])
        left_knee = np.array([landmarks[self.LEFT_KNEE].x, landmarks[self.LEFT_KNEE].y])
        right_knee = np.array([landmarks[self.RIGHT_KNEE].x, landmarks[self.RIGHT_KNEE].y])
        
        # Calculate midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        knee_mid = (left_knee + right_knee) / 2
        
        # 1. Spine angle (shoulder-hip line vs vertical)
        spine_vector = shoulder_mid - hip_mid
        spine_angle = abs(math.degrees(math.atan2(abs(spine_vector[0]), abs(spine_vector[1]))))
        
        # 2. Head to shoulder angle (key metric for posture)
        head_shoulder_vector = nose - shoulder_mid
        head_shoulder_angle = math.degrees(math.atan2(head_shoulder_vector[1], abs(head_shoulder_vector[0])))
        
        # 3. Neck angle (nose to shoulder midpoint)
        neck_vector = nose - shoulder_mid
        neck_angle = abs(math.degrees(math.atan2(neck_vector[0], neck_vector[1])))
        
        # 4. Shoulder alignment (how level shoulders are)
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        shoulder_alignment = max(0, 1 - (shoulder_diff * 10))  # 0-1 score
        
        # 5. Head forward position
        head_forward = abs(nose[0] - shoulder_mid[0])
        
        # 6. Lateral lean (side-to-side position)
        lateral_lean = hip_mid[0] - shoulder_mid[0]
        
        # 7. Standing indicator (hip-knee distance)
        hip_knee_distance = abs(hip_mid[1] - knee_mid[1])
        
        # 8. Confidence based on landmark visibility and consistency
        confidence = self._calculate_confidence(landmarks)
        
        return {
            'spine_angle': spine_angle,
            'head_shoulder_angle': head_shoulder_angle,
            'neck_angle': neck_angle,
            'shoulder_alignment': shoulder_alignment,
            'head_forward': head_forward,
            'lateral_lean': lateral_lean,
            'hip_knee_distance': hip_knee_distance,
            'confidence': confidence
        }
    
    def _classify_posture(self, metrics):
        """Classify posture based on calculated metrics"""
        
        spine_angle = metrics['spine_angle']
        head_shoulder_angle = metrics['head_shoulder_angle']
        lateral_lean = metrics['lateral_lean']
        hip_knee_distance = metrics['hip_knee_distance']
        
        # 1. Check for lying down (spine very horizontal)
        if spine_angle > self.thresholds['lying_spine_angle']:
            return 'lying'
        
        # 2. Check for standing (large hip-knee distance)
        if hip_knee_distance > self.thresholds['standing_hip_knee']:
            return 'standing'
        
        # 3. Check for lateral leaning (left/right sitting)
        if abs(lateral_lean) > self.thresholds['lateral_lean']:
            if lateral_lean < 0:
                return 'left_sitting'
            else:
                return 'right_sitting'
        
        # 4. Check for hunching (poor spine angle)
        if spine_angle > self.thresholds['hunching_spine_angle']:
            return 'hunching_over'
        
        # 5. Check for leaning forward (head angle)
        if abs(head_shoulder_angle) > self.thresholds['lean_forward_head']:
            return 'leaning_forward'
        
        # 6. Default to sitting straight
        return 'sitting_straight'
    
    def _calculate_confidence(self, landmarks):
        """Calculate confidence score based on landmark quality"""
        try:
            # Check visibility of key landmarks
            key_landmarks = [self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 
                           self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_KNEE, self.RIGHT_KNEE]
            
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
        feedback_map = {
            'sitting_straight': 'Excellent sitting posture! Keep it up.',
            'hunching_over': f'Hunching detected (spine angle: {metrics["spine_angle"]:.1f}°). Straighten your back and shoulders.',
            'leaning_forward': f'Leaning forward detected. Sit back and align your spine properly.',
            'left_sitting': 'Leaning to the left. Center yourself in your chair.',
            'right_sitting': 'Leaning to the right. Center yourself in your chair.',
            'lying': 'Lying position detected. Sit up for proper work posture.',
            'standing': 'Standing detected. Good for taking breaks!',
            'unknown': 'Unable to clearly detect posture.'
        }
        
        return feedback_map.get(posture_type, 'Monitor your posture regularly.')
    
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