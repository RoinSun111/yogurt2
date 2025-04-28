import logging
from datetime import datetime, timedelta

class FocusCalculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Focus calculator initialized")
        
    def is_focused(self, is_present, posture):
        """
        Determine if the user is focused based on presence and posture
        
        Args:
            is_present (bool): Whether the user is detected in the frame
            posture (str): The classified posture ('upright', 'slouched', 'unknown')
            
        Returns:
            bool: True if focused, False otherwise
        """
        # User is focused if present and with upright posture
        is_focused = is_present and posture == 'upright'
        
        self.logger.debug(f"Focus calculation: Present={is_present}, Posture={posture}, Focused={is_focused}")
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
