import logging
import mediapipe as mp
import numpy as np
import math
import cv2
from datetime import datetime, timedelta

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
        
        # Initialize MediaPipe Face Mesh for precise neck angle detection
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_mesh_available = True
        except Exception as e:
            self.logger.warning(f"Face Mesh not available: {e}")
            self.face_mesh_available = False
        
        # Define key facial landmarks for head orientation
        # These landmarks are used to calculate precise head pose
        self.face_landmarks_3d = {
            'nose_tip': 1,
            'chin': 152,
            'left_eye_corner': 33,
            'right_eye_corner': 362,
            'left_mouth_corner': 61,
            'right_mouth_corner': 291,
            'forehead': 10
        }
        
        # Store historical posture data for trend analysis
        self.posture_history = []
        self.history_max_size = 300  # Store ~5 minutes of data at 1 Hz
        
        # Threshold angles for posture assessment (more realistic thresholds)
        self.upright_threshold = 20.0  # Angle for basic upright vs slouched classification
        self.excellent_threshold = 8.0  # Very upright posture
        self.good_threshold = 15.0     # Good posture
        self.fair_threshold = 22.0     # Acceptable posture
        # Anything above fair_threshold is considered poor
        
        # Threshold for neck angle (head tilt) - more forgiving
        self.neck_angle_threshold = 25.0  # Degree threshold for neck angle
        
        # Thresholds for shoulder alignment (more realistic for real-world conditions)
        self.shoulder_alignment_excellent = 0.90
        self.shoulder_alignment_good = 0.80
        self.shoulder_alignment_fair = 0.65
        
    def analyze(self, image):
        """
        Analyze posture from an image using MediaPipe Pose and Face Mesh
        Returns: dict with detailed posture classification, angles, and feedback
        """
        try:
            # Process the image with MediaPipe Pose and Face Mesh
            pose_results = self.pose.process(image)
            face_results = None
            if self.face_mesh_available:
                face_results = self.face_mesh.process(image)
            
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
            if not pose_results.pose_landmarks:
                self.logger.debug("No pose landmarks detected")
                return posture_data
            
            # Extract key landmarks for posture analysis
            landmarks = pose_results.pose_landmarks.landmark
            
            # Check if the person is visible (using visibility of key points)
            nose_visible = landmarks[self.mp_pose.PoseLandmark.NOSE].visibility > 0.5
            left_shoulder_visible = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5
            right_shoulder_visible = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5
            
            is_present = nose_visible and (left_shoulder_visible or right_shoulder_visible)
            posture_data['is_present'] = is_present
            
            # Include pose landmarks in the result for frontend visualization
            # Convert landmarks to serializable format
            serializable_landmarks = []
            for landmark in landmarks:
                serializable_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            posture_data['pose_landmarks'] = serializable_landmarks
            
            # Also keep raw landmarks for internal analysis
            posture_data['_raw_landmarks'] = landmarks
            
            if not is_present:
                self.logger.debug("Person not clearly visible")
                return posture_data
            
            # Calculate basic posture angle (average of left and right side)
            angle = self._calculate_posture_angle(landmarks)
            posture_data['angle'] = angle
            
            # Enhanced neck angle calculation using Face Mesh if available
            if self.face_mesh_available and face_results and face_results.multi_face_landmarks:
                neck_angle = self._calculate_precise_neck_angle_face_mesh(face_results.multi_face_landmarks[0], landmarks)
            else:
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
        
        # Deduct points based on overall back angle (more forgiving)
        if angle <= self.excellent_threshold:
            pass  # No deduction for excellent posture
        elif angle <= self.good_threshold:
            score -= 5  # Reduced from 10
        elif angle <= self.fair_threshold:
            score -= 12  # Reduced from 20
        else:
            score -= 25  # Reduced from 35
        
        # Deduct points based on neck angle (text neck) - more forgiving
        if neck_angle > self.neck_angle_threshold:
            score -= min(20, int((neck_angle - self.neck_angle_threshold) * 1.0))  # Reduced penalty
        
        # Deduct points based on shoulder alignment (more forgiving)
        if shoulder_alignment >= self.shoulder_alignment_excellent:
            pass  # No deduction for excellent alignment
        elif shoulder_alignment >= self.shoulder_alignment_good:
            score -= 3  # Reduced from 5
        elif shoulder_alignment >= self.shoulder_alignment_fair:
            score -= 8  # Reduced from 15
        else:
            score -= 15  # Reduced from 25
        
        # Deduct points for forward head position (more forgiving)
        if head_forward > 0.15:  # Increased threshold
            score -= min(15, int(head_forward * 50))  # Reduced penalty
        
        # Deduct points for spine curvature (more forgiving)
        if spine_curvature > 155:
            score -= 3  # Reduced from 5
        elif spine_curvature > 145:
            score -= 8  # Reduced from 15
        elif spine_curvature > 135:
            score -= 15  # Reduced from 25
        
        # Deduct points for asymmetry (more forgiving)
        if symmetry < 0.7:  # Lowered threshold
            score -= min(15, int((1 - symmetry) * 30))  # Reduced penalty
        
        # Map score to quality categories (adjusted thresholds)
        if score >= 80:  # Lowered from 85
            return 'excellent'
        elif score >= 65:  # Lowered from 70
            return 'good'
        elif score >= 45:  # Lowered from 50
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
    
    def _calculate_precise_neck_angle_face_mesh(self, face_landmarks, pose_landmarks):
        """
        Calculate precise neck angle using MediaPipe Face Mesh (468 landmarks)
        This provides much more accurate head orientation than pose landmarks alone
        
        Args:
            face_landmarks: Face mesh landmarks from MediaPipe
            pose_landmarks: Pose landmarks for reference points
            
        Returns:
            float: Precise neck angle in degrees
        """
        try:
            # Convert face landmarks to numpy array for easier processing
            face_points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # Key facial landmarks for head pose estimation
            # These landmarks are stable across different face orientations
            nose_tip = face_points[self.face_landmarks_3d['nose_tip']]  # Tip of nose
            chin = face_points[self.face_landmarks_3d['chin']]  # Bottom of chin
            forehead = face_points[self.face_landmarks_3d['forehead']]  # Center of forehead
            left_eye = face_points[self.face_landmarks_3d['left_eye_corner']]  # Left eye corner
            right_eye = face_points[self.face_landmarks_3d['right_eye_corner']]  # Right eye corner
            
            # Calculate head orientation vectors
            # Vertical face vector (forehead to chin)
            face_vertical = chin - forehead
            
            # Horizontal face vector (left to right eye)
            face_horizontal = right_eye - left_eye
            
            # Calculate normal vector to face plane
            face_normal = np.cross(face_horizontal, face_vertical)
            face_normal = face_normal / np.linalg.norm(face_normal)  # Normalize
            
            # Reference vectors for neutral position
            # Assume neutral head position has face normal pointing forward (0, 0, -1)
            neutral_normal = np.array([0, 0, -1])
            
            # Calculate pitch (up/down tilt) - this is our neck angle
            # Project face normal onto the vertical plane (XZ plane)
            face_normal_xz = np.array([face_normal[0], 0, face_normal[2]])
            face_normal_xz = face_normal_xz / np.linalg.norm(face_normal_xz)
            
            # Calculate angle between neutral and current head orientation
            dot_product = np.dot(neutral_normal, face_normal_xz)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range
            
            pitch_angle = np.degrees(np.arccos(dot_product))
            
            # Determine if head is tilted up or down
            if face_normal[1] > 0:  # Head tilted up
                pitch_angle = -pitch_angle
            
            # Calculate yaw (left/right turn) for additional context
            face_normal_xy = np.array([face_normal[0], face_normal[1], 0])
            if np.linalg.norm(face_normal_xy) > 0:
                face_normal_xy = face_normal_xy / np.linalg.norm(face_normal_xy)
                yaw_angle = np.degrees(np.arcsin(np.clip(face_normal[0], -1.0, 1.0)))
            else:
                yaw_angle = 0
            
            # Calculate roll (head tilt side to side)
            roll_angle = np.degrees(np.arctan2(face_horizontal[1], 
                                             np.sqrt(face_horizontal[0]**2 + face_horizontal[2]**2)))
            
            # Combine pitch and roll for comprehensive neck angle
            # Pitch is primary for forward head posture, roll adds side tilt
            combined_neck_angle = np.sqrt(pitch_angle**2 + roll_angle**2)
            
            # Add some contribution from yaw if head is turned significantly
            if abs(yaw_angle) > 15:  # Head turned more than 15 degrees
                combined_neck_angle += abs(yaw_angle) * 0.3  # Weighted contribution
            
            return abs(combined_neck_angle)
            
        except Exception as e:
            self.logger.warning(f"Error in precise neck angle calculation: {e}")
            # Fallback to basic pose-based calculation
            return self._calculate_neck_angle(pose_landmarks)
