"""
New Posture Analyzer - Simplified wrapper for the Advanced Posture Detector
Integrates with existing SmartDesk system while providing accurate posture detection.
"""

import logging
import numpy as np
from .advanced_posture_detector import AdvancedPostureDetector, PostureType, PostureQuality

class PostureAnalyzer:
    """
    Simplified posture analyzer that wraps the AdvancedPostureDetector
    to maintain compatibility with existing system interfaces.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("New Posture analyzer initialized")
        
        # Initialize the advanced detector
        self.detector = AdvancedPostureDetector()
        
        # Mapping from advanced detector types to simplified types for compatibility
        self.posture_mapping = {
            PostureType.UPRIGHT_SITTING: "sitting_straight",
            PostureType.SLOUCHED_SITTING: "slouched",
            PostureType.LEANING_LEFT: "left_sitting",
            PostureType.LEANING_RIGHT: "right_sitting", 
            PostureType.FORWARD_LEANING: "hunching_over",
            PostureType.STANDING_GOOD: "standing",
            PostureType.STANDING_POOR: "standing_poor",
            PostureType.LYING_DOWN: "lying",
            PostureType.UNKNOWN: "unknown"
        }
        
        # Quality mapping
        self.quality_mapping = {
            PostureQuality.EXCELLENT: "excellent",
            PostureQuality.GOOD: "good", 
            PostureQuality.FAIR: "fair",
            PostureQuality.POOR: "poor",
            PostureQuality.UNKNOWN: "unknown"
        }
    
    def analyze(self, image):
        """
        Analyze posture from an image using the advanced detector.
        Returns: dict with posture classification, angles, and feedback
        """
        try:
            # Use the advanced detector
            result = self.detector.detect_posture(image)
            
            # Map to simplified format for compatibility
            simplified_posture = self.posture_mapping.get(result.posture_type, "unknown")
            simplified_quality = self.quality_mapping.get(result.quality, "unknown")
            
            # Calculate overall posture score
            posture_score = self.detector.get_posture_score(result.metrics)
            
            # Format feedback as single string
            feedback_text = None
            if result.feedback:
                feedback_text = " ".join(result.feedback[:3])  # Take first 3 feedback items
            
            # Calculate spine angle for backward compatibility
            spine_angle = result.metrics.spine_angle
            if spine_angle == 0:  # If no spine angle calculated, derive from posture type
                if result.posture_type == PostureType.UPRIGHT_SITTING:
                    spine_angle = 175.0
                elif result.posture_type == PostureType.SLOUCHED_SITTING:
                    spine_angle = 130.0
                elif result.posture_type == PostureType.FORWARD_LEANING:
                    spine_angle = 145.0
                elif result.posture_type in [PostureType.STANDING_GOOD, PostureType.STANDING_POOR]:
                    spine_angle = 170.0
                else:
                    spine_angle = 160.0
            
            # Return format compatible with existing system
            return {
                'posture': simplified_posture,
                'posture_quality': simplified_quality,
                'angle': spine_angle,
                'neck_angle': result.metrics.neck_angle,
                'shoulder_alignment': result.metrics.symmetry_score,
                'head_forward_position': result.metrics.head_forward_distance,
                'spine_curvature': max(0, 180 - result.metrics.spine_angle),  # Convert to curvature
                'symmetry_score': result.metrics.symmetry_score,
                'feedback': feedback_text,
                'is_present': result.is_present,
                'confidence': result.confidence,
                'posture_score': posture_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in posture analysis: {e}")
            
            # Return default values on error
            return {
                'posture': 'unknown',
                'posture_quality': 'unknown',
                'angle': 0.0,
                'neck_angle': 0.0,
                'shoulder_alignment': 0.0,
                'head_forward_position': 0.0,
                'spine_curvature': 0.0,
                'symmetry_score': 0.0,
                'feedback': None,
                'is_present': False,
                'confidence': 0.0,
                'posture_score': 0.0
            }
    
    def get_posture_trend(self, minutes=60):
        """
        Get posture trend data for the specified time period.
        Returns trend analysis and recommendations.
        """
        try:
            from datetime import datetime, timedelta
            from models import PostureStatus
            
            # Get recent posture data
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            
            recent_postures = PostureStatus.query.filter(
                PostureStatus.timestamp >= start_time,
                PostureStatus.timestamp <= end_time
            ).order_by(PostureStatus.timestamp.desc()).limit(100).all()
            
            if not recent_postures:
                return {
                    'trend': 'neutral',
                    'recommendation': 'Start using the system to track your posture trends'
                }
            
            # Analyze trend
            good_postures = ['excellent', 'good']
            recent_quality = [p.posture_quality for p in recent_postures if hasattr(p, 'posture_quality')]
            
            if len(recent_quality) < 3:
                return {
                    'trend': 'neutral',
                    'recommendation': 'Keep monitoring your posture for better insights'
                }
            
            # Calculate trend
            recent_third = recent_quality[:len(recent_quality)//3]
            older_third = recent_quality[-len(recent_quality)//3:]
            
            recent_good_ratio = sum(1 for q in recent_third if q in good_postures) / len(recent_third)
            older_good_ratio = sum(1 for q in older_third if q in good_postures) / len(older_third)
            
            if recent_good_ratio > older_good_ratio + 0.1:
                trend = 'improving'
                recommendation = 'Great progress! Your posture is getting better'
            elif recent_good_ratio < older_good_ratio - 0.1:
                trend = 'declining'
                recommendation = 'Take a break and focus on your posture'
            else:
                trend = 'stable'
                if recent_good_ratio > 0.7:
                    recommendation = 'Excellent! Keep maintaining good posture'
                else:
                    recommendation = 'Try to maintain better posture consistently'
            
            return {
                'trend': trend,
                'recommendation': recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Error getting posture trend: {e}")
            return {
                'trend': 'neutral',
                'recommendation': 'Unable to analyze posture trend'
            }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'detector'):
            self.detector.close()