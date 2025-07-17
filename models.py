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

class MoodboardSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    widget_type = db.Column(db.String(50))  # 'mood', 'calendar', 'todo', 'focus_score', 'water', 'posture_score'
    is_enabled = db.Column(db.Boolean, default=True)
    position_x = db.Column(db.Integer, default=0)  # Grid position X
    position_y = db.Column(db.Integer, default=0)  # Grid position Y
    width = db.Column(db.Integer, default=1)  # Grid width
    height = db.Column(db.Integer, default=1)  # Grid height
    config = db.Column(db.Text, nullable=True)  # JSON configuration for the widget
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class MoodEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mood_text = db.Column(db.String(200))  # Customizable mood text
    date = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class CalendarEvent(db.Model):
    """Calendar events including meetings, appointments, and deadlines"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    event_type = db.Column(db.String(50), default='meeting')  # meeting, appointment, deadline, reminder
    location = db.Column(db.String(200), nullable=True)
    attendees = db.Column(db.Text, nullable=True)  # JSON string of attendee emails
    is_all_day = db.Column(db.Boolean, default=False)
    is_ai_created = db.Column(db.Boolean, default=False)  # Created via AI assistant
    source = db.Column(db.String(50), default='manual')  # manual, google, lark, caldav
    external_id = db.Column(db.String(200), nullable=True)  # ID from external calendar
    priority = db.Column(db.String(20), default='medium')  # low, medium, high
    reminder_minutes = db.Column(db.Integer, default=15)  # Minutes before event to remind
    color = db.Column(db.String(7), default='#3498db')  # Hex color for calendar display
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class TodoItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text, nullable=True)
    is_completed = db.Column(db.Boolean, default=False)
    priority = db.Column(db.String(20), default='medium')  # low, medium, high
    due_date = db.Column(db.Date, nullable=True)
    due_time = db.Column(db.Time, nullable=True)  # Optional specific time
    calendar_event_id = db.Column(db.Integer, db.ForeignKey('calendar_event.id'), nullable=True)  # Link to calendar event
    created_at = db.Column(db.DateTime, default=datetime.now)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationship to calendar event
    calendar_event = db.relationship('CalendarEvent', backref='linked_todos')


class Achievement(db.Model):
    """Track user achievements and milestones"""
    id = db.Column(db.Integer, primary_key=True)
    achievement_type = db.Column(db.String(50), nullable=False)  # 'elite_performer', 'streak_3', 'streak_7', 'perfect_posture', etc.
    date_earned = db.Column(db.Date, default=datetime.now().date)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    score = db.Column(db.Float, nullable=True)  # Score when achievement was earned
    streak_count = db.Column(db.Integer, default=0)  # For streak-based achievements
    achievement_data = db.Column(db.Text, nullable=True)  # JSON string for additional data


class DailyStreak(db.Model):
    """Track daily streaks for various metrics"""
    id = db.Column(db.Integer, primary_key=True)
    streak_type = db.Column(db.String(50), nullable=False)  # 'elite_performer', 'good_posture', 'hydration_goal'
    current_streak = db.Column(db.Integer, default=0)
    longest_streak = db.Column(db.Integer, default=0)
    last_update_date = db.Column(db.Date, default=datetime.now().date)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class CalendarIntegration(db.Model):
    """Store calendar integration settings"""
    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(50), nullable=False)  # google, lark, caldav
    account_name = db.Column(db.String(100), nullable=False)
    access_token = db.Column(db.Text, nullable=True)  # Encrypted access token
    refresh_token = db.Column(db.Text, nullable=True)  # Encrypted refresh token
    calendar_url = db.Column(db.String(500), nullable=True)  # For CalDAV
    sync_enabled = db.Column(db.Boolean, default=True)
    sync_interval = db.Column(db.Integer, default=15)  # Minutes between syncs
    last_sync = db.Column(db.DateTime, nullable=True)
    settings = db.Column(db.Text, nullable=True)  # JSON settings for the integration
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class AIConversation(db.Model):
    """Store AI calendar conversation history"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    intent = db.Column(db.String(50), nullable=True)  # create, update, delete, list, find
    confidence_score = db.Column(db.Float, default=0.0)
    event_id = db.Column(db.Integer, db.ForeignKey('calendar_event.id'), nullable=True)
    is_voice_input = db.Column(db.Boolean, default=False)
    processing_time_ms = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
