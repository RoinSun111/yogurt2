import logging
from datetime import datetime, timedelta

class FocusCalculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Focus calculator initialized")
        
    def is_focused(self, is_present, posture, activity_state=None, head_angle=0):
        """
        Determine if the user is focused based on presence, posture, and activity
        
        Args:
            is_present (bool): Whether the user is detected in the frame
            posture (str): The classified posture ('sitting_straight', 'standing', 'leaning_forward', etc.)
            activity_state (str): Activity state ('working', 'not_working', 'distracted_by_others', etc.)
            head_angle (float): Head angle from center (0 = looking straight)
            
        Returns:
            bool: True if focused, False otherwise
        """
        # Define focused postures (good working positions)
        focused_postures = ['sitting_straight', 'standing']
        
        # Define working activity states
        working_states = ['working', 'unknown']  # Include unknown as neutral working state
        
        # User is focused if:
        # 1. Present in frame
        # 2. Has good posture OR is in working activity state
        # 3. Head is relatively centered (not looking away significantly)
        is_good_posture = posture in focused_postures
        is_working_activity = activity_state in working_states if activity_state else True
        is_looking_forward = abs(head_angle) < 30  # Allow some head movement
        
        # Primary focus determination: present + (good posture OR working activity) + looking forward
        is_focused = is_present and (is_good_posture or is_working_activity) and is_looking_forward
        
        self.logger.debug(f"Focus calculation: Present={is_present}, Posture={posture}, Activity={activity_state}, HeadAngle={head_angle:.1f}°, Focused={is_focused}")
        return is_focused
    
    def calculate_daily_score(self, focus_entries):
        """
        Calculate the daily focus score based on focus entries
        
        Args:
            focus_entries (list): List of FocusScore model instances
            
        Returns:
            float: Focus score as a percentage (0-100)
        """
        if not focus_entries:
            return 0.0
        
        # Count focused entries
        focused_count = sum(1 for entry in focus_entries if entry.is_focused)
        total_count = len(focus_entries)
        
        # Calculate score as percentage
        focus_score = (focused_count / total_count) * 100 if total_count > 0 else 0
        
        self.logger.debug(f"Daily focus score: {focus_score:.2f}% ({focused_count}/{total_count} entries)")
        return focus_score
