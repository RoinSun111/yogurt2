import logging
import numpy as np
import cv2
import time
import math
from datetime import datetime
from utils.eye_tracker import EyeTracker

class ActivityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Activity analyzer initialized")
        
        # State tracking
        self.previous_landmarks = None
        self.previous_timestamp = None
        self.current_state = "unknown"
        self.current_substate = None
        self.time_in_current_state = 0
        self.state_start_time = time.time()
        
        # Store frame history for movement detection
        self.frame_history = []
        self.max_history_frames = 5
        
        # Tracking for hand movement detection
        self.hand_positions = []
        self.typing_threshold = 5  # Number of movements per 10 seconds for typing
        self.writing_threshold = 3  # Number of movements per 10 seconds for writing
        
        # Movement thresholds
        self.movement_threshold = 0.02  # 2% change in keypoints for movement detection
        
        # State transition time thresholds (in seconds)
        self.working_to_not_working_threshold = 30  # No hand movement for 30 seconds
        self.not_working_to_idle_threshold = 120  # Inactivity for 2 minutes
        self.on_break_threshold = 60  # Person absent for 1 minute
        self.distraction_threshold = 5  # Head turned for 5 seconds
    
    def analyze(self, image, pose_landmarks, face_detections=None):
        """
        Analyze the user's activity based on pose landmarks and face detections
        
        Args:
            image: The RGB image from the camera
            pose_landmarks: MediaPipe pose landmarks
            face_detections: Optional face detection results for counting people
            
        Returns:
            dict: Activity state and metrics
        """
        current_time = time.time()
        
        # Initialize result with default values
        result = {
            'activity_state': self.current_state,
            'working_substate': self.current_substate,
            'head_angle': 0.0,
            'movement_level': 0.0,
            'people_detected': 1 if pose_landmarks else 0,
            'time_in_state': int(current_time - self.state_start_time)
        }
        
        # If no landmarks detected, check if user is on break
        if not pose_landmarks:
            # If no person for more than 1 minute, transition to "on_break"
            if self.current_state != "on_break" and (current_time - self.state_start_time) > self.on_break_threshold:
                self._transition_state("on_break")
            return result
        
        # Add the current frame to history for movement analysis
        if len(self.frame_history) >= self.max_history_frames:
            self.frame_history.pop(0)
        self.frame_history.append(image)
        
        # Calculate head angle from pose landmarks
        head_angle = self._calculate_head_angle(pose_landmarks)
        result['head_angle'] = head_angle
        
        # Calculate movement level between current and previous landmarks
        movement_level = self._calculate_movement(pose_landmarks)
        result['movement_level'] = movement_level
        
        # Detect number of people (if face detection is available)
        if face_detections and len(face_detections) > 0:
            result['people_detected'] = len(face_detections)
        
        # Update the current state based on analysis
        self._update_state(head_angle, movement_level, result['people_detected'])
        
        # If in working state, detect the specific working activity
        if self.current_state == "working":
            self._detect_working_activity(movement_level, head_angle)
            result['working_substate'] = self.current_substate
        
        # Update result with current state
        result['activity_state'] = self.current_state
        result['time_in_state'] = int(current_time - self.state_start_time)
        
        # Store current landmarks for next comparison
        self.previous_landmarks = pose_landmarks
        self.previous_timestamp = current_time
        
        return result
    
    def _calculate_head_angle(self, landmarks):
        """
        Calculate the head angle from center (how much the user is looking away)
        """
        try:
            # Get nose and eyes landmarks
            nose = landmarks[0]
            left_eye = landmarks[2]
            right_eye = landmarks[5]
            
            # Calculate center of eyes
            eye_center_x = (left_eye.x + right_eye.x) / 2
            
            # Calculate angle between nose and eye center to get head turn
            # A perfectly centered head would have nose.x == eye_center_x
            angle = abs(np.rad2deg(np.arctan2(nose.x - eye_center_x, 0.1)))
            
            return min(90, angle * 3)  # Scale angle for better sensitivity, cap at 90 degrees
        except (IndexError, AttributeError) as e:
            self.logger.debug(f"Error calculating head angle: {str(e)}")
            return 0.0
    
    def _calculate_movement(self, landmarks):
        """
        Calculate the movement level by comparing current landmarks to previous ones
        """
        if not self.previous_landmarks:
            return 0.0
        
        try:
            # Calculate movement between current and previous landmarks
            total_movement = 0.0
            landmark_count = len(landmarks)
            
            # Focus on key points for hands to detect typing/writing movements
            hand_landmarks = [
                4,   # Right wrist
                8,   # Right index finger
                12,  # Right middle finger
                16,  # Right ring finger
                20,  # Right pinky
                7,   # Left wrist
                11,  # Left index finger
                15,  # Left middle finger
                19,  # Left ring finger
                23   # Left pinky
            ]
            
            # Calculate total movement across all landmarks
            for i in range(landmark_count):
                curr_x, curr_y = landmarks[i].x, landmarks[i].y
                prev_x, prev_y = self.previous_landmarks[i].x, self.previous_landmarks[i].y
                
                # Calculate Euclidean distance
                distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                total_movement += distance
                
                # Track hand position changes for typing/writing detection
                if i in hand_landmarks and distance > 0.01:  # Only count significant hand movements
                    self.hand_positions.append({
                        'position': (curr_x, curr_y),
                        'timestamp': time.time()
                    })
            
            # Normalize movement by number of landmarks
            normalized_movement = (total_movement / landmark_count) * 100
            
            # Cleanup old hand position records (older than 10 seconds)
            current_time = time.time()
            self.hand_positions = [p for p in self.hand_positions if current_time - p['timestamp'] <= 10]
            
            return min(100, normalized_movement)  # Cap at 100
        except Exception as e:
            self.logger.error(f"Error calculating movement: {str(e)}")
            return 0.0
    
    def _update_state(self, head_angle, movement_level, people_detected):
        """
        Update the current activity state based on metrics
        """
        current_time = time.time()
        
        # If coming back from "on_break", reset to appropriate state
        if self.current_state == "on_break" and movement_level > self.movement_threshold:
            if movement_level > 5.0:  # Significant movement
                self._transition_state("working")
            else:
                self._transition_state("not_working")
            return
        
        # Check for multiple people -> distracted by others
        if people_detected > 1 and self.current_state != "distracted_by_others":
            self._transition_state("distracted_by_others")
            return
            
        # Check head angle for distraction (head turned significantly)
        if head_angle > 60 and self.current_state != "distracted_by_others":
            if self.previous_timestamp and (current_time - self.previous_timestamp) > self.distraction_threshold:
                self._transition_state("distracted_by_others")
            return
                
        # State transitions based on current state
        if self.current_state == "working":
            # Transition to not working if no movement for threshold period
            if movement_level < self.movement_threshold and len(self.hand_positions) == 0:
                time_inactive = 0
                if self.previous_timestamp:
                    time_inactive = current_time - self.previous_timestamp
                if time_inactive > self.working_to_not_working_threshold:
                    self._transition_state("not_working")
            
        elif self.current_state == "not_working":
            # Transition to working if movement detected
            if movement_level > self.movement_threshold and len(self.hand_positions) > 2:
                self._transition_state("working")
            # Transition to idle if inactive for long period
            elif movement_level < self.movement_threshold:
                time_inactive = current_time - self.state_start_time
                if time_inactive > self.not_working_to_idle_threshold:
                    self._transition_state("idle")
                    
        elif self.current_state == "idle":
            # Transition to working if significant movement detected
            if movement_level > 2 * self.movement_threshold:
                self._transition_state("working")
            # Transition to not_working if some movement detected
            elif movement_level > self.movement_threshold:
                self._transition_state("not_working")
                
        elif self.current_state == "distracted_by_others":
            # Transition back to working if head angle returns to normal
            # and no multiple people detected
            if head_angle < 30 and people_detected <= 1 and movement_level > self.movement_threshold:
                self._transition_state("working")
            
        # If state is still unknown, initialize based on current metrics
        if self.current_state == "unknown":
            if movement_level > self.movement_threshold:
                self._transition_state("working")
            else:
                self._transition_state("not_working")
    
    def _detect_working_activity(self, movement_level, head_angle):
        """
        Detect the specific working activity (typing, writing, reading)
        when in the "working" state
        """
        # Count hand movements in last 10 seconds
        hand_movement_count = len(self.hand_positions)
        
        # Detect typing (many hand movements)
        if hand_movement_count >= self.typing_threshold:
            if self.current_substate != "typing":
                self.current_substate = "typing"
                self.logger.debug(f"Working substate: typing (movements: {hand_movement_count})")
        
        # Detect writing (medium hand movements)
        elif hand_movement_count >= self.writing_threshold:
            if self.current_substate != "writing":
                self.current_substate = "writing"
                self.logger.debug(f"Working substate: writing (movements: {hand_movement_count})")
        
        # Detect reading (low hand movements, centered head)
        elif head_angle < 10 and movement_level < 2.0:
            if self.current_substate != "reading":
                self.current_substate = "reading"
                self.logger.debug(f"Working substate: reading (head angle: {head_angle:.1f}°)")
        
        # Default case - generic working
        else:
            if self.current_substate is not None:
                self.current_substate = None
                self.logger.debug("Working substate: generic working")
    
    def _transition_state(self, new_state):
        """
        Transition to a new activity state
        """
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # Reset substate when leaving working state
        if new_state != "working":
            self.current_substate = None
            
        self.logger.info(f"Activity state transition: {old_state} -> {new_state}")