import logging
import numpy as np
import cv2
import time
import math
from datetime import datetime
import os
import random
from typing import Dict, List, Tuple, Optional

# Try to import tflite_runtime (our preferred small TFLite runtime)
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
except ImportError:
    # Fall back to full TensorFlow if tflite_runtime is not available
    try:
        import tensorflow as tf
        USING_TFLITE_RUNTIME = False
    except ImportError:
        # Neither TF nor TFLite available - will use fallback heuristics
        USING_TFLITE_RUNTIME = False

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
        self.landmark_history = []
        self.max_landmark_history = 30  # Store about 3 seconds of pose data at 10fps
        
        # Movement thresholds (used as fallback)
        self.movement_threshold = 0.02  # 2% change in keypoints for movement detection
        
        # State transition time thresholds (in seconds)
        self.working_to_not_working_threshold = 30  # No activity for 30 seconds
        self.not_working_to_idle_threshold = 120  # Inactivity for 2 minutes
        self.on_break_threshold = 60  # Person absent for 1 minute
        self.distraction_threshold = 5  # Head turned for 5 seconds
        
        # Initialize TinyML model for activity recognition
        self._initialize_ml_model()
        
        # Activity labels
        self.activity_states = [
            "working", "not_working", "distracted_by_others", 
            "on_break", "idle"
        ]
        
        # Working substates
        self.working_substates = [
            "typing", "writing", "reading", "thinking", 
            "phone_use", "computer_use"
        ]
    
    def _initialize_ml_model(self):
        """Initialize TensorFlow Lite model for activity recognition"""
        try:
            # In a real implementation, we would load an actual TinyML model here
            # But for this demo, we'll use heuristic detection
            self.model_loaded = False
            self.use_tf = False
            
            # Set up thresholds for activity detection
            self.gaze_focus_threshold = 15.0  # Head angle threshold for focus detection
            self.typing_threshold = 5  # Number of movements per 10 seconds for typing
            self.writing_threshold = 3  # Number of movements per 10 seconds for writing
            
            # Additional counters for new requirements
            self.distraction_count = 0
            self.last_distraction_time = 0
            self.distraction_events = []
            self.away_time_start = None
            self.away_time_total = 0
            self.break_count = 0
            
            # Focus time tracking (for the enhanced focus metrics)
            self.focus_time_start = None
            self.focus_time_total = 0
            self.active_work_time_total = 0
            
            self.logger.info("Activity analyzer initialized with enhanced focus tracking")
            
        except Exception as e:
            self.logger.error(f"Error initializing activity analyzer: {str(e)}")
    
    def analyze(self, image, pose_landmarks, face_detections=None):
        """
        Analyze the user's activity based on pose landmarks and face detections using TinyML
        
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
            'time_in_state': int(current_time - self.state_start_time),
            # New focus metrics
            'is_focused': False,
            'focus_time_total': self.focus_time_total,
            'active_work_time_total': self.active_work_time_total,
            'focus_rate': 0.0,
            'distraction_count': self.distraction_count,
            'break_count': self.break_count,
            'away_time_total': self.away_time_total
        }
        
        # If no landmarks detected, check if user is on break (away)
        if not pose_landmarks:
            # Track away time (new requirement)
            if self.away_time_start is None:
                self.away_time_start = current_time
                self.logger.debug("User away time started")
            
            # If no person for more than 30 seconds, transition to "on_break"
            if self.current_state != "on_break" and (current_time - self.state_start_time) > self.on_break_threshold:
                self._transition_state("on_break")
                # Increment break counter
                self.break_count += 1
            
            return result
        else:
            # If user was away and now is back, update away time
            if self.away_time_start is not None:
                away_duration = current_time - self.away_time_start
                self.away_time_total += away_duration
                self.away_time_start = None
                self.logger.debug(f"User away time ended, duration: {away_duration:.1f}s")
        
        # Add the current frame to history for movement analysis
        if len(self.frame_history) >= self.max_history_frames:
            self.frame_history.pop(0)
        self.frame_history.append(image)
        
        # Store landmark data for ML model input
        self._update_landmark_history(pose_landmarks)
        
        # Calculate head angle from pose landmarks
        head_angle = self._calculate_head_angle(pose_landmarks)
        result['head_angle'] = head_angle
        
        # Check if user is focused based on head angle
        is_focused = self._is_user_focused(head_angle)
        result['is_focused'] = is_focused
        
        # Calculate movement level between current and previous landmarks
        movement_level = self._calculate_movement(pose_landmarks)
        result['movement_level'] = movement_level
        
        # Detect number of people (if face detection is available)
        if face_detections and len(face_detections) > 0:
            result['people_detected'] = len(face_detections)
        
        # Check for phone use (another distraction type)
        is_phone_use = self._detect_phone_use(pose_landmarks)
        if is_phone_use and self.current_state == "working":
            self.current_substate = "phone_use"
        
        # Use heuristic detection for activity states
        self._update_state(head_angle, movement_level, result['people_detected'])
        
        # If in working state, detect the specific working activity
        if self.current_state == "working" and not is_phone_use:
            self._detect_working_activity(movement_level, head_angle)
        
        # Update result with current state and substate
        result['working_substate'] = self.current_substate
        result['activity_state'] = self.current_state
        result['time_in_state'] = int(current_time - self.state_start_time)
        
        # Update focus metrics
        result['focus_time_total'] = self.focus_time_total
        if self.focus_time_start is not None:
            # Add current focus session
            result['focus_time_total'] += (current_time - self.focus_time_start)
        
        result['active_work_time_total'] = self.active_work_time_total
        
        # Calculate focus rate (Focus Time / Active Work Time)
        if result['active_work_time_total'] > 0:
            result['focus_rate'] = min(100.0, (result['focus_time_total'] / result['active_work_time_total']) * 100.0)
            
        result['distraction_count'] = self.distraction_count
        result['break_count'] = self.break_count
        result['away_time_total'] = self.away_time_total
        
        # Store current landmarks for next comparison
        self.previous_landmarks = pose_landmarks
        self.previous_timestamp = current_time
        
        return result
        
    def _update_landmark_history(self, landmarks):
        """Store landmark history for temporal ML features"""
        # Extract x, y coordinates from landmarks for ML input
        landmark_features = []
        for lm in landmarks:
            landmark_features.extend([lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0])
        
        # Add timestamp
        self.landmark_history.append({
            'features': landmark_features,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.landmark_history) > self.max_landmark_history:
            self.landmark_history.pop(0)
            
    def _is_user_focused(self, head_angle):
        """
        Determine if the user is focused based on head angle
        Implements the 'Focus Time Tracking' requirement
        """
        # User is considered focused if head angle is below the threshold
        is_focused = head_angle < self.gaze_focus_threshold
        
        current_time = time.time()
        
        # Track focus time
        if is_focused:
            # If we weren't already tracking focus time, start now
            if self.focus_time_start is None:
                self.focus_time_start = current_time
                self.logger.debug("User focus started")
        else:
            # If we were tracking focus time, update total and reset
            if self.focus_time_start is not None:
                focus_duration = current_time - self.focus_time_start
                self.focus_time_total += focus_duration
                self.focus_time_start = None
                self.logger.debug(f"User focus ended, duration: {focus_duration:.1f}s")
                
            # Check for distraction events
            if head_angle > 30:  # More significant head turn
                # Only count as a new distraction if it's been a while since the last one
                if current_time - self.last_distraction_time > 10:  # At least 10 seconds between distinct distractions
                    self.distraction_count += 1
                    self.last_distraction_time = current_time
                    self.distraction_events.append({
                        'timestamp': current_time,
                        'head_angle': head_angle
                    })
                    self.logger.debug(f"Distraction event detected: head angle {head_angle:.1f}°")
        
        # Update active work time whenever the user is present
        # (Used for Focus Rate calculation - Focus Time / Active Work Time)
        if self.current_state in ["working", "not_working"]:
            # If we have a previous timestamp, add the elapsed time
            if self.previous_timestamp:
                self.active_work_time_total += (current_time - self.previous_timestamp)
        
        return is_focused
        
    def _detect_phone_use(self, pose_landmarks):
        """
        Detect if the user is holding a phone
        Part of the 'Distraction Event Detection' requirement
        
        In a real implementation, this would use object detection or hand position analysis
        """
        # Simple heuristic for demo - check if one hand is near the face
        # Real implementation would use TinyML for object detection
        try:
            # Check if landmarks for hand and face are available
            if len(pose_landmarks) >= 20:
                # Get wrist and ear positions
                right_wrist = pose_landmarks[4]
                left_wrist = pose_landmarks[7]
                right_ear = pose_landmarks[8]
                left_ear = pose_landmarks[7]
                
                # Calculate distances from wrists to ears
                right_distance = np.sqrt((right_wrist.x - right_ear.x)**2 + (right_wrist.y - right_ear.y)**2)
                left_distance = np.sqrt((left_wrist.x - left_ear.x)**2 + (left_wrist.y - left_ear.y)**2)
                
                # Check if either wrist is close to the ear (phone call position)
                is_phone_near_ear = min(right_distance, left_distance) < 0.15
                
                if is_phone_near_ear:
                    # Count as a phone distraction event
                    current_time = time.time()
                    if current_time - self.last_distraction_time > 10:  # At least 10s between events
                        self.distraction_count += 1
                        self.last_distraction_time = current_time
                        self.distraction_events.append({
                            'timestamp': current_time,
                            'type': 'phone_use'
                        })
                        self.logger.debug("Phone use detected")
                        
                        # Set working substate if we're in working state
                        if self.current_state == "working":
                            self.current_substate = "phone_use"
                
                return is_phone_near_ear
            
            return False
        except Exception as e:
            self.logger.error(f"Error detecting phone use: {str(e)}")
            return False
    
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