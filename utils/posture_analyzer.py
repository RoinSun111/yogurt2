import logging
import mediapipe as mp
import numpy as np
import math
import cv2
from datetime import datetime, timedelta
from .posture_calibrator import PostureCalibrator

class PostureAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Posture analyzer initialized")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Use the medium model for better posture analysis
        )
        
        # Store historical posture data for trend analysis
        self.posture_history = []
        self.history_max_size = 300  # Store ~5 minutes of data at 1 Hz
        
        # Threshold angles for posture assessment
        self.upright_threshold = 15.0  # Angle for basic upright vs slouched classification
        self.excellent_threshold = 5.0  # Very upright posture
        self.good_threshold = 10.0     # Good posture
        self.fair_threshold = 15.0     # Acceptable posture
        # Anything above fair_threshold is considered poor
        
        # Threshold for neck angle (head tilt)
        self.neck_angle_threshold = 20.0  # Degree threshold for neck angle
        
        # Thresholds for shoulder alignment (higher is better, 1.0 is perfect)
        self.shoulder_alignment_excellent = 0.95
        self.shoulder_alignment_good = 0.9
        self.shoulder_alignment_fair = 0.8
        
    def analyze(self, image):
        """
        Analyze posture from an image using MediaPipe Pose
        Returns: dict with detailed posture classification, angles, and feedback
        """
        try:
            # Process the image with MediaPipe
            results = self.pose.process(image)
            
            # Default values if no pose is detected
            posture_data = {
                'posture': 'unknown',
                'posture_quality': 'unknown',
                'angle': 0.0,
                'neck_angle': 0.0,
                'shoulder_alignment': 0.0,
                'head_forward_position': 0.0,
                'spine_curvature': 0.0,
                'symmetry_score': 0.0,
                'feedback': None,
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
            
            # Include pose landmarks in the result for activity analysis
            posture_data['pose_landmarks'] = landmarks
            
            if not is_present:
                self.logger.debug("Person not clearly visible")
                return posture_data
            
            # Calculate basic posture angle (average of left and right side)
            angle = self._calculate_posture_angle(landmarks)
            posture_data['angle'] = angle
            
            # Neck angle (head tilt) calculation
            neck_angle = self._calculate_neck_angle(landmarks)
            posture_data['neck_angle'] = neck_angle
            
            # Calculate shoulder alignment score
            shoulder_alignment = self._calculate_shoulder_alignment(landmarks)
            posture_data['shoulder_alignment'] = shoulder_alignment
            
            # Calculate head forward position
            head_forward = self._calculate_head_forward_position(landmarks)
            posture_data['head_forward_position'] = head_forward
            
            # Calculate estimated spine curvature
            spine_curvature = self._calculate_spine_curvature(landmarks)
            posture_data['spine_curvature'] = spine_curvature
            
            # Calculate overall symmetry score
            symmetry_score = self._calculate_symmetry_score(landmarks)
            posture_data['symmetry_score'] = symmetry_score
            
            # Classify posture based on angle
            if angle < self.upright_threshold:
                posture_data['posture'] = 'upright'
            else:
                posture_data['posture'] = 'slouched'
            
            # Determine posture quality (finer-grained classification)
            posture_data['posture_quality'] = self._assess_posture_quality(
                angle, neck_angle, shoulder_alignment, head_forward, spine_curvature, symmetry_score
            )
            
            # Generate specific feedback based on metrics
            posture_data['feedback'] = self._generate_posture_feedback(
                angle, neck_angle, shoulder_alignment, head_forward, spine_curvature, symmetry_score
            )
            
            # Store posture data in history for trend analysis
            self._update_posture_history(posture_data)
            
            self.logger.debug(f"Posture: {posture_data['posture']}, Quality: {posture_data['posture_quality']}, Angle: {angle:.2f} degrees")
            return posture_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing posture: {str(e)}")
            return {
                'posture': 'unknown',
                'posture_quality': 'unknown',
                'angle': 0.0,
                'neck_angle': 0.0,
                'shoulder_alignment': 0.0,
                'head_forward_position': 0.0,
                'spine_curvature': 0.0,
                'symmetry_score': 0.0,
                'feedback': "Error analyzing posture.",
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
    
    def _calculate_neck_angle(self, landmarks):
        """
        Calculate the neck angle (head tilt) using ear, nose, and shoulder landmarks
        A higher angle indicates forward head posture or "text neck"
        """
        # Use the most visible ear for calculation
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Choose the most visible ear
        ear = left_ear if left_ear.visibility > right_ear.visibility else right_ear
        
        # Get nose and shoulder landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        # Use the most visible shoulder
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder = left_shoulder if left_shoulder.visibility > right_shoulder.visibility else right_shoulder
        
        # Calculate angle between ear, nose, and shoulder
        neck_angle = self._angle_between_points(
            (ear.x, ear.y),
            (nose.x, nose.y),
            (shoulder.x, shoulder.y)
        )
        
        return neck_angle
    
    def _calculate_shoulder_alignment(self, landmarks):
        """
        Calculate the shoulder alignment score (0-1)
        1 means shoulders are perfectly level, lower values indicate asymmetry
        """
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate height difference in normalized coordinates
        height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # Convert to a score where 1 is perfect alignment and 0 is poor alignment
        # Typically, height_diff should be small (< 0.05) for good alignment
        alignment_score = max(0, 1 - (height_diff * 10))
        
        return alignment_score
    
    def _calculate_head_forward_position(self, landmarks):
        """
        Calculate the head forward position relative to shoulders
        Higher values indicate head is too far forward (poor posture)
        """
        # Get ear and shoulder positions
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate average positions
        ear_x = (left_ear.x + right_ear.x) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # Calculate forward distance (in normalized space)
        # Use absolute value for consistent meaning regardless of camera orientation
        forward_position = abs(ear_x - shoulder_x)
        
        return forward_position
    
    def _calculate_spine_curvature(self, landmarks):
        """
        Estimate spine curvature using shoulder, hip, and knee landmarks
        Higher values indicate more curvature (potentially worse posture)
        """
        # Get landmarks for spine estimation
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate midpoints
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        knee_mid = ((left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2)
        
        # Calculate angle along the spine
        spine_angle = self._angle_between_points(shoulder_mid, hip_mid, knee_mid)
        
        return spine_angle
    
    def _calculate_symmetry_score(self, landmarks):
        """
        Calculate overall body symmetry score (0-1)
        1 means perfect symmetry, lower values indicate asymmetry
        """
        # Pairs of landmarks to compare for symmetry
        landmark_pairs = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Calculate symmetry scores for each pair
        pair_scores = []
        for left_landmark, right_landmark in landmark_pairs:
            left = landmarks[left_landmark]
            right = landmarks[right_landmark]
            
            # Only use points with good visibility
            if left.visibility > 0.5 and right.visibility > 0.5:
                # Calculate vertical symmetry (y-axis)
                y_diff = abs(left.y - right.y)
                
                # Calculate horizontal symmetry around a center line
                # Assuming the image is normalized such that x ranges from 0 to 1
                center_x = 0.5
                x_symmetry = abs((center_x - left.x) - (right.x - center_x))
                
                # Combine into a single score (0-1)
                pair_score = max(0, 1 - (y_diff * 5) - (x_symmetry * 5))
                pair_scores.append(pair_score)
        
        # Return average symmetry score if we have valid pairs, otherwise return 0
        if pair_scores:
            return sum(pair_scores) / len(pair_scores)
        else:
            return 0.0
    
    def _assess_posture_quality(self, angle, neck_angle, shoulder_alignment, head_forward, spine_curvature, symmetry):
        """
        Determine posture quality based on multiple metrics
        Returns: 'excellent', 'good', 'fair', or 'poor'
        """
        # Start with a high score and deduct points for issues
        score = 100
        
        # Deduct points based on overall back angle
        if angle <= self.excellent_threshold:
            pass  # No deduction for excellent posture
        elif angle <= self.good_threshold:
            score -= 10
        elif angle <= self.fair_threshold:
            score -= 20
        else:
            score -= 35
        
        # Deduct points based on neck angle (text neck)
        if neck_angle > self.neck_angle_threshold:
            score -= min(30, int((neck_angle - self.neck_angle_threshold) * 1.5))
        
        # Deduct points based on shoulder alignment
        if shoulder_alignment >= self.shoulder_alignment_excellent:
            pass  # No deduction for excellent alignment
        elif shoulder_alignment >= self.shoulder_alignment_good:
            score -= 5
        elif shoulder_alignment >= self.shoulder_alignment_fair:
            score -= 15
        else:
            score -= 25
        
        # Deduct points for forward head position
        if head_forward > 0.1:
            score -= min(25, int(head_forward * 100))
        
        # Deduct points for spine curvature
        if spine_curvature > 160:
            score -= 5
        elif spine_curvature > 150:
            score -= 15
        elif spine_curvature > 140:
            score -= 25
        
        # Deduct points for asymmetry
        if symmetry < 0.8:
            score -= min(20, int((1 - symmetry) * 50))
        
        # Map score to quality categories
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 50:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_posture_feedback(self, angle, neck_angle, shoulder_alignment, head_forward, spine_curvature, symmetry):
        """
        Generate specific feedback based on posture metrics
        Returns a string with actionable advice
        """
        feedback = []
        
        # Check back angle (slouching)
        if angle > self.fair_threshold:
            feedback.append("Sit up straighter to reduce back strain.")
        
        # Check for forward head position ("text neck")
        if neck_angle > self.neck_angle_threshold:
            feedback.append("Bring your head back to align with your shoulders.")
        
        # Check shoulder alignment (one shoulder higher than the other)
        if shoulder_alignment < self.shoulder_alignment_fair:
            feedback.append("Level your shoulders to improve alignment.")
        
        # Check head forward position
        if head_forward > 0.1:
            feedback.append("Pull your head back - don't lean forward toward the screen.")
        
        # Check spine curvature
        if spine_curvature < 150:
            feedback.append("Maintain a natural curve in your spine - don't hunch forward.")
        
        # Check symmetry
        if symmetry < 0.8:
            feedback.append("Try to balance your posture more evenly on both sides.")
        
        # If no specific issues, give positive reinforcement based on quality
        if not feedback:
            quality = self._assess_posture_quality(angle, neck_angle, shoulder_alignment, head_forward, spine_curvature, symmetry)
            if quality == 'excellent':
                feedback.append("Excellent posture! Keep it up.")
            elif quality == 'good':
                feedback.append("Good posture - minor adjustments could make it excellent.")
            else:
                feedback.append("Your posture could use some small improvements.")
        
        return " ".join(feedback)
    
    def _update_posture_history(self, posture_data):
        """
        Add posture data to history for trend analysis and limit history size
        """
        # Add timestamp to posture data
        posture_entry = posture_data.copy()
        posture_entry['timestamp'] = datetime.now()
        
        # Add to history
        self.posture_history.append(posture_entry)
        
        # Limit history size
        if len(self.posture_history) > self.history_max_size:
            self.posture_history.pop(0)
    
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
        
        # Prevent division by zero
        if mag1 * mag2 == 0:
            return 0
        
        # Ensure the value is in valid range for acos
        cos_angle = min(1.0, max(-1.0, dot_product / (mag1 * mag2)))
        
        # Angle in degrees
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
