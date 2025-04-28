import logging
import mediapipe as mp
import numpy as np
import math
import cv2

class PostureAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Posture analyzer initialized")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Use the lightweight model for efficiency
        )
        
        # Threshold angle for upright posture (degrees)
        self.upright_threshold = 15.0
        
    def analyze(self, image):
        """
        Analyze posture from an image using MediaPipe Pose
        Returns: dict with posture classification, angle, and presence
        """
        try:
            # Process the image with MediaPipe
            results = self.pose.process(image)
            
            # Default values if no pose is detected
            posture_data = {
                'posture': 'unknown',
                'angle': 0.0,
                'is_present': False
            }
            
            # Check if pose landmarks are detected
            if not results.pose_landmarks:
                self.logger.debug("No pose landmarks detected")
                return posture_data
            
            # Extract key landmarks for posture analysis
            landmarks = results.pose_landmarks.landmark
            
            # Check if the person is visible (using visibility of key points)
            nose_visible = landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5
            left_shoulder_visible = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5
            right_shoulder_visible = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5
            
            is_present = nose_visible and (left_shoulder_visible or right_shoulder_visible)
            posture_data['is_present'] = is_present
            
            if not is_present:
                self.logger.debug("Person not clearly visible")
                return posture_data
            
            # Calculate posture angle (average of left and right side)
            angle = self._calculate_posture_angle(landmarks)
            posture_data['angle'] = angle
            
            # Classify posture based on angle
            if angle < self.upright_threshold:
                posture_data['posture'] = 'upright'
            else:
                posture_data['posture'] = 'slouched'
            
            self.logger.debug(f"Posture: {posture_data['posture']}, Angle: {angle:.2f} degrees")
            return posture_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing posture: {str(e)}")
            return {
                'posture': 'unknown',
                'angle': 0.0,
                'is_present': False
            }
    
    def _calculate_posture_angle(self, landmarks):
        """
        Calculate the posture angle using shoulder and hip landmarks
        """
        # Get shoulder and hip coordinates
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate angles on both sides
        left_angle = self._angle_between_points(
            (left_shoulder.x, left_shoulder.y), 
            (left_hip.x, left_hip.y),
            (left_hip.x, 0)  # Vertical reference point
        )
        
        right_angle = self._angle_between_points(
            (right_shoulder.x, right_shoulder.y), 
            (right_hip.x, right_hip.y),
            (right_hip.x, 0)  # Vertical reference point
        )
        
        # Average the two angles
        avg_angle = (left_angle + right_angle) / 2.0
        return avg_angle
    
    def _angle_between_points(self, p1, p2, p3):
        """
        Calculate angle between three points in degrees
        """
        # Vectors
        v1 = [p1[0] - p2[0], p1[1] - p2[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        # Dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Magnitudes
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Angle in degrees
        angle = math.degrees(math.acos(dot_product / (mag1 * mag2)))
        
        return angle
