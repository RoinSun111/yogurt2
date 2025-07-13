"""
Main posture analyzer that integrates with the SmartDesk system
Uses the new PostureDetector for accurate real-time posture monitoring
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from .posture_detector import PostureDetector


class PostureAnalyzer:
    """
    Main posture analyzer interface for the SmartDesk system
    Provides compatibility with existing system while using improved detection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Posture analyzer initialized with new detector")
        
        # Initialize the new posture detector
        self.detector = PostureDetector()
        
        # History tracking
        self.posture_history = []
        
    def analyze(self, image):
        """
        Analyze posture from image and return results compatible with existing system
        """
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image_rgb = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image_rgb = image[:, :, :3]
            else:
                image_rgb = image
            
            # Detect posture using new detector
            result = self.detector.detect_posture(image_rgb)
            
            # Add to history
            if result['is_present']:
                self.posture_history.append({
                    'timestamp': datetime.now(),
                    'posture': result['posture'],
                    'quality': result['posture_quality'],
                    'angle': result['angle']
                })
                
                # Keep only last 2 hours
                cutoff = datetime.now() - timedelta(hours=2)
                self.posture_history = [entry for entry in self.posture_history 
                                      if entry['timestamp'] > cutoff]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in posture analysis: {e}")
            return self.detector._error_response()
    
    def get_posture_trend(self, minutes=60):
        """Get posture trend analysis for the specified time period"""
        if not self.posture_history:
            return {
                'trend': 'neutral',
                'quality_counts': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
                'average_angle': 0,
                'recommendation': "Not enough posture data collected yet."
            }
        
        # Filter recent history
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_history = [entry for entry in self.posture_history 
                         if entry['timestamp'] > cutoff_time]
        
        if not recent_history:
            return {
                'trend': 'neutral',
                'quality_counts': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
                'average_angle': 0,
                'recommendation': "No recent posture data available."
            }
        
        # Count quality ratings
        quality_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for entry in recent_history:
            quality = entry.get('quality', 'unknown')
            if quality in quality_counts:
                quality_counts[quality] += 1
        
        # Calculate average angle
        angles = [entry['angle'] for entry in recent_history if entry['angle'] > 0]
        avg_angle = sum(angles) / len(angles) if angles else 0
        
        # Determine trend
        good_postures = quality_counts['excellent'] + quality_counts['good']
        poor_postures = quality_counts['fair'] + quality_counts['poor']
        
        if good_postures > poor_postures * 1.5:
            trend = 'positive'
            recommendation = "Your posture has been good! Keep maintaining proper alignment."
        elif poor_postures > good_postures * 1.5:
            trend = 'negative'
            recommendation = "Your posture needs improvement. Focus on sitting up straight and taking breaks."
        else:
            trend = 'neutral'
            recommendation = "Your posture is moderate. Small adjustments can make a big difference."
        
        return {
            'trend': trend,
            'quality_counts': quality_counts,
            'average_angle': avg_angle,
            'recommendation': recommendation
        }