import logging
import cv2
import numpy as np
import mediapipe as mp
import math
from datetime import datetime

class EyeTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Eye tracker initialized")
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define eye landmarks indices
        # Left eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Pupil landmarks
        self.LEFT_PUPIL = [468, 469, 470, 471, 472]
        self.RIGHT_PUPIL = [473, 474, 475, 476, 477]
        
        # Eye state tracking
        self.blink_threshold = 0.2  # Threshold for eye aspect ratio to detect blink
        self.gaze_history = []  # Track recent gaze directions
        self.blink_history = []  # Track recent blinks
        self.history_max_size = 30  # Max size of history (about 3 seconds at 10fps)
        
        # Thresholds
        self.screen_looking_threshold = 0.3  # Threshold for determining if looking at screen
    
    def analyze(self, image):
        """
        Analyze eye gaze from an image using MediaPipe Face Mesh
        
        Returns: dict with gaze direction, focus status, and blink information
        """
        result = {
            'is_looking_at_screen': False,
            'eyes_open': False,
            'gaze_direction': 'unknown',  # left, right, center, up, down, unknown
            'blink_detected': False,
            'left_eye_ratio': 0,
            'right_eye_ratio': 0,
            'gaze_score': 0,  # 0-100 score indicating focus level
            'face_found': False
        }
        
        try:
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            face_results = self.face_mesh.process(rgb_image)
            
            # If no face detected, return default result
            if not face_results.multi_face_landmarks:
                return result
            
            result['face_found'] = True
            
            # Get the first face
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Get landmark coordinates
            h, w, _ = image.shape
            landmarks = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]
            
            # Calculate eye aspect ratios
            left_ear = self._calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE)
            right_ear = self._calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            result['left_eye_ratio'] = left_ear
            result['right_eye_ratio'] = right_ear
            
            # Detect if eyes are open
            is_eyes_open = left_ear > self.blink_threshold and right_ear > self.blink_threshold
            result['eyes_open'] = is_eyes_open
            
            # Track blinks 
            # If previously open and now closed
            current_time = datetime.now()
            if len(self.blink_history) > 0:
                last_state = self.blink_history[-1]['eyes_open']
                if last_state and not is_eyes_open:
                    result['blink_detected'] = True
            
            # Add current eye state to history
            self.blink_history.append({
                'timestamp': current_time,
                'eyes_open': is_eyes_open
            })
            
            # Cleanup old blink records
            self.blink_history = [b for b in self.blink_history 
                                 if (current_time - b['timestamp']).total_seconds() < 3]
            
            # If eyes are closed, we can't determine gaze direction
            if not is_eyes_open:
                result['gaze_direction'] = 'eyes_closed'
                result['is_looking_at_screen'] = False
                result['gaze_score'] = 0
                return result
            
            # Analyze gaze direction
            gaze_direction, is_looking = self._analyze_gaze(landmarks)
            result['gaze_direction'] = gaze_direction
            result['is_looking_at_screen'] = is_looking
            
            # Add to gaze history
            self.gaze_history.append({
                'timestamp': current_time,
                'direction': gaze_direction,
                'is_looking': is_looking
            })
            
            # Limit history size
            if len(self.gaze_history) > self.history_max_size:
                self.gaze_history.pop(0)
            
            # Calculate gaze score based on history (how much time spent looking at screen)
            if len(self.gaze_history) > 0:
                looking_count = sum(1 for g in self.gaze_history if g['is_looking'])
                result['gaze_score'] = int((looking_count / len(self.gaze_history)) * 100)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing eye gaze: {str(e)}")
            return result
    
    def _calculate_eye_aspect_ratio(self, landmarks, eye_points):
        """
        Calculate the eye aspect ratio as a measure of eye openness
        
        EAR = (h1 + h2) / (2 * w)
        where h1, h2 are the heights and w is the width of the eye
        """
        try:
            # Get eye landmarks
            eye = [landmarks[point] for point in eye_points]
            
            # Get the height of the eye (average of two measurements)
            h1 = self._distance(eye[1], eye[5])
            h2 = self._distance(eye[2], eye[4])
            
            # Get the width of the eye
            w = self._distance(eye[0], eye[3])
            
            # Calculate EAR
            if w == 0:
                return 0
            
            return (h1 + h2) / (2.0 * w)
        except Exception as e:
            self.logger.error(f"Error calculating eye aspect ratio: {str(e)}")
            return 0
    
    def _analyze_gaze(self, landmarks):
        """
        Analyze the gaze direction based on the relationship between
        eye landmarks and pupil position
        """
        try:
            # Get pupil landmarks
            left_pupil = [landmarks[idx] for idx in self.LEFT_PUPIL]
            right_pupil = [landmarks[idx] for idx in self.RIGHT_PUPIL]
            
            # Calculate pupil centers
            left_pupil_center = self._get_center(left_pupil)
            right_pupil_center = self._get_center(right_pupil)
            
            # Get eye corners
            left_eye_left = landmarks[self.LEFT_EYE[0]]
            left_eye_right = landmarks[self.LEFT_EYE[8]]
            right_eye_left = landmarks[self.RIGHT_EYE[0]]
            right_eye_right = landmarks[self.RIGHT_EYE[8]]
            
            # Calculate relative positions
            left_eye_width = self._distance(left_eye_left, left_eye_right)
            right_eye_width = self._distance(right_eye_left, right_eye_right)
            
            if left_eye_width == 0 or right_eye_width == 0:
                return "unknown", False
            
            # Horizontal gaze ratio for left and right eye
            left_ratio = (left_pupil_center[0] - left_eye_left[0]) / left_eye_width
            right_ratio = (right_pupil_center[0] - right_eye_left[0]) / right_eye_width
            
            # Average horizontal gaze ratio (0.5 means center)
            horizontal_ratio = (left_ratio + right_ratio) / 2.0
            
            # Determine general direction
            if horizontal_ratio < 0.35:
                direction = "left"
                is_looking = False
            elif horizontal_ratio > 0.65:
                direction = "right"
                is_looking = False
            else:
                # Looking generally center, let's check vertical gaze
                # Get eye top and bottom points
                left_eye_top = landmarks[self.LEFT_EYE[12]]
                left_eye_bottom = landmarks[self.LEFT_EYE[4]]
                right_eye_top = landmarks[self.RIGHT_EYE[12]]
                right_eye_bottom = landmarks[self.RIGHT_EYE[4]]
                
                # Calculate eye heights
                left_eye_height = self._distance(left_eye_top, left_eye_bottom)
                right_eye_height = self._distance(right_eye_top, right_eye_bottom)
                
                if left_eye_height == 0 or right_eye_height == 0:
                    return "center", True
                
                # Vertical gaze ratio
                left_vertical = (left_pupil_center[1] - left_eye_top[1]) / left_eye_height
                right_vertical = (right_pupil_center[1] - right_eye_top[1]) / right_eye_height
                
                # Average vertical ratio (0.5 means center)
                vertical_ratio = (left_vertical + right_vertical) / 2.0
                
                if vertical_ratio < 0.35:
                    direction = "up"
                    is_looking = vertical_ratio > 0.2  # Only if not extreme
                elif vertical_ratio > 0.65:
                    direction = "down"
                    is_looking = vertical_ratio < 0.8  # Only if not extreme
                else:
                    direction = "center"
                    is_looking = True
            
            return direction, is_looking
            
        except Exception as e:
            self.logger.error(f"Error analyzing gaze: {str(e)}")
            return "unknown", False
    
    def _get_center(self, points):
        """Calculate the center point of a set of points"""
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return (x, y)
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)