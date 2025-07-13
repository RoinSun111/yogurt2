"""
New Posture Analyzer - Using SitPose Classifier for accurate posture detection
Integrates with existing SmartDesk system while providing accurate detection of 7 posture types.
"""

import logging
import numpy as np
import mediapipe as mp
from datetime import datetime, timedelta
from .sitpose_classifier import SitPoseClassifier, SitPoseType

class PostureAnalyzer:
    """
    Posture analyzer using the SitPose classifier for accurate detection of:
    - sitting straight
    - hunching over  
    - lying
    - leaning forward
    - left sitting
    - right sitting
    - standing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("New Posture analyzer initialized")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize the SitPose classifier
        self.classifier = SitPoseClassifier()
        
        # Quality thresholds based on spine angle
        self.quality_thresholds = {
            "excellent": 10,  # < 10 degrees
            "good": 15,      # < 15 degrees
            "fair": 20,      # < 20 degrees
            "poor": float('inf')  # >= 20 degrees
        }
        
        # Initialize posture history for trend tracking
        self.posture_history = []
    
    def analyze(self, image):
        """
        Analyze posture from an image using the SitPose classifier.
        Returns: dict with posture classification, angles, and feedback
        """
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image_rgb = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image_rgb = image[:, :, :3]
            else:
                image_rgb = image
            
            # Process image with MediaPipe
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return {
                    'posture': 'unknown',
                    'posture_quality': 'unknown',
                    'angle': 160,  # Default angle for no detection
                    'neck_angle': 0,
                    'shoulder_alignment': 0,
                    'head_forward_position': 0,
                    'spine_curvature': 180,
                    'symmetry_score': 0,
                    'feedback': 'No person detected in frame',
                    'is_present': False
                }
            
            # Use SitPose classifier
            posture_result = self.classifier.classify_posture(results.pose_landmarks.landmark)
            
            # Determine quality based on spine angle
            quality = self._determine_quality(posture_result.spine_angle)
            
            # Calculate additional metrics for compatibility
            shoulder_alignment = self._calculate_shoulder_alignment(results.pose_landmarks.landmark)
            neck_angle = self._calculate_neck_angle(results.pose_landmarks.landmark)
            head_forward = self._calculate_head_forward(results.pose_landmarks.landmark)
            
            # Convert spine angle to curvature (180 - angle for compatibility)
            spine_curvature = 180 - posture_result.spine_angle
            
            result = {
                'posture': posture_result.posture_type,
                'posture_quality': quality,
                'angle': posture_result.spine_angle,
                'neck_angle': neck_angle,
                'shoulder_alignment': shoulder_alignment,
                'head_forward_position': head_forward,
                'spine_curvature': spine_curvature,
                'symmetry_score': shoulder_alignment,  # Use shoulder alignment as symmetry
                'feedback': posture_result.feedback,
                'is_present': True
            }
            
            # Add to history for trend tracking
            self.posture_history.append({
                'timestamp': datetime.now(),
                'posture': posture_result.posture_type,
                'posture_quality': quality,
                'angle': posture_result.spine_angle
            })
            
            # Keep only last 2 hours of history
            cutoff = datetime.now() - timedelta(hours=2)
            self.posture_history = [entry for entry in self.posture_history 
                                   if entry['timestamp'] > cutoff]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing posture: {str(e)}")
            return {
                'posture': 'unknown',
                'posture_quality': 'unknown',
                'angle': 160,
                'neck_angle': 0,
                'shoulder_alignment': 0,
                'head_forward_position': 0,
                'spine_curvature': 180,
                'symmetry_score': 0,
                'feedback': 'Error analyzing posture',
                'is_present': False
            }
    
    def _determine_quality(self, spine_angle: float) -> str:
        """Determine posture quality based on spine angle"""
        for quality, threshold in self.quality_thresholds.items():
            if spine_angle < threshold:
                return quality
        return "poor"
    
    def _calculate_shoulder_alignment(self, landmarks) -> float:
        """Calculate shoulder alignment score (0-1)"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Calculate vertical difference
            y_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Convert to alignment score (1 = perfect alignment)
            alignment = max(0, 1 - (y_diff * 10))  # Scale factor of 10
            
            return alignment
        except:
            return 0.5
    
    def _calculate_neck_angle(self, landmarks) -> float:
        """Calculate neck angle from vertical"""
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Get shoulder midpoint
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate angle
            dx = nose.x - shoulder_mid_x
            dy = nose.y - shoulder_mid_y
            
            angle = np.degrees(np.arctan2(dx, dy))
            return abs(angle)
        except:
            return 0
    
    def _calculate_head_forward(self, landmarks) -> float:
        """Calculate head forward position"""
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Forward position relative to shoulders
            forward_distance = nose.x - shoulder_mid_x
            
            return abs(forward_distance)
        except:
            return 0
    
    def get_posture_trend(self, minutes=60):
        """
        Analyze posture trend over the specified time period
        Returns a dict with trend data and recommendations
        """
        if not self.posture_history:
            return {
                'trend': 'neutral',
                'quality_counts': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unknown': 0},
                'average_angle': 0,
                'recommendation': "Not enough posture data collected yet."
            }
        
        # Filter history for specified time period
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_history = [entry for entry in self.posture_history 
                         if entry['timestamp'] > cutoff_time]
        
        if not recent_history:
            return {
                'trend': 'neutral',
                'quality_counts': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unknown': 0},
                'average_angle': 0,
                'recommendation': "No recent posture data available."
            }
        
        # Count quality ratings
        quality_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unknown': 0}
        for entry in recent_history:
            quality = entry.get('posture_quality', 'unknown')
            quality_counts[quality] += 1
        
        # Calculate average angle
        angles = [entry['angle'] for entry in recent_history if entry['angle'] > 0]
        avg_angle = sum(angles) / len(angles) if angles else 0
        
        # Determine trend
        if quality_counts['excellent'] + quality_counts['good'] > quality_counts['fair'] + quality_counts['poor']:
            trend = 'positive'
            recommendation = "Your posture has been good. Keep it up!"
        elif quality_counts['poor'] > (quality_counts['excellent'] + quality_counts['good'] + quality_counts['fair']) / 2:
            trend = 'negative'
            recommendation = "Your posture needs improvement. Try to sit up straighter and take regular breaks."
        else:
            trend = 'neutral'
            recommendation = "Your posture is average. Making small adjustments throughout the day can help."
        
        return {
            'trend': trend,
            'quality_counts': quality_counts,
            'average_angle': avg_angle,
            'recommendation': recommendation
        }