from app import db
from datetime import datetime

class FocusScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_focused = db.Column(db.Boolean, default=False)
    is_present = db.Column(db.Boolean, default=False)
    date = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class PostureStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    posture = db.Column(db.String(50), default="unknown")  # "upright", "slouched", "unknown"
    posture_quality = db.Column(db.String(50), default="unknown")  # "excellent", "good", "fair", "poor", "unknown"
    angle = db.Column(db.Float, default=0.0)  # Shoulder-hip angle in degrees
    neck_angle = db.Column(db.Float, default=0.0)  # Neck angle (head tilt) in degrees
    shoulder_alignment = db.Column(db.Float, default=0.0)  # Shoulder alignment score (0-1)
    head_forward_position = db.Column(db.Float, default=0.0)  # Head forward position in normalized units
    spine_curvature = db.Column(db.Float, default=0.0)  # Estimated spine curvature angle
    symmetry_score = db.Column(db.Float, default=0.0)  # Body symmetry score (0-1)
    feedback = db.Column(db.String(255), nullable=True)  # Specific feedback about posture issues
    timestamp = db.Column(db.DateTime, default=datetime.now)

class WaterIntake(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, default=0.0)  # Amount in milliliters
    date = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class ActivityStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # Main activity state
    # "working", "not_working", "distracted_by_others", "on_break", "idle"
    activity_state = db.Column(db.String(50), default="unknown")
    
    # Working sub-state (if in "working" state)
    # "typing", "writing", "reading", or null if not in working state
    working_substate = db.Column(db.String(50), nullable=True)
    
    # Additional metrics
    head_angle = db.Column(db.Float, default=0.0)  # Head angle from center
    movement_level = db.Column(db.Float, default=0.0)  # Movement intensity (0-100)
    people_detected = db.Column(db.Integer, default=0)  # Number of people in frame
    
    # Eye tracking metrics
    gaze_direction = db.Column(db.String(20), default="center")  # Direction of gaze: center, left, right, up, down, etc.
    eye_focus_score = db.Column(db.Float, default=1.0)  # Eye focus score (0-1)
    blink_rate = db.Column(db.Float, default=0.0)  # Blinks per minute
    
    # Time in current state (seconds)
    time_in_state = db.Column(db.Integer, default=0)
    
    # Timestamps
    date = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class DistractionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50))  # distraction, away, talking, phone_use
    duration = db.Column(db.Integer, default=0)  # Duration in seconds
    start_time = db.Column(db.DateTime, default=datetime.now)
    end_time = db.Column(db.DateTime, nullable=True)
    date = db.Column(db.Date, default=datetime.now().date)
    details = db.Column(db.String(255), nullable=True)  # Additional context about the event
