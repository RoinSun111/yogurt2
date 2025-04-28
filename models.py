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
    angle = db.Column(db.Float, default=0.0)  # Shoulder-hip angle in degrees
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
    
    # Time in current state (seconds)
    time_in_state = db.Column(db.Integer, default=0)
    
    # Timestamps
    date = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)
