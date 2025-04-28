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
