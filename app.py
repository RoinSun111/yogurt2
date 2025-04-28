import os
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///smartdesk.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

# Import utils after app initialization
from utils.camera_processor import CameraProcessor
from utils.posture_analyzer import PostureAnalyzer
from utils.focus_calculator import FocusCalculator

# Initialize the processors
camera_processor = CameraProcessor()
posture_analyzer = PostureAnalyzer()
focus_calculator = FocusCalculator()

with app.app_context():
    # Import models here
    import models
    db.create_all()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/focus_score')
def get_focus_score():
    today = datetime.now().date()
    focus_entries = models.FocusScore.query.filter_by(date=today).all()
    if not focus_entries:
        return jsonify({'score': 0, 'working_time': 0, 'total_time': 0})
    
    total_entries = len(focus_entries)
    working_entries = sum(1 for entry in focus_entries if entry.is_focused)
    
    if total_entries == 0:
        score = 0
    else:
        score = (working_entries / total_entries) * 100
    
    return jsonify({
        'score': round(score, 1),
        'working_time': working_entries * 6,  # 6 seconds per entry
        'total_time': total_entries * 6  # 6 seconds per entry
    })

@app.route('/api/posture')
def get_posture():
    posture_status = models.PostureStatus.query.order_by(models.PostureStatus.timestamp.desc()).first()
    if not posture_status:
        return jsonify({'posture': 'unknown', 'angle': 0})
    
    return jsonify({
        'posture': posture_status.posture,
        'angle': posture_status.angle
    })

@app.route('/api/water_intake')
def get_water_intake():
    today = datetime.now().date()
    water_entries = models.WaterIntake.query.filter_by(date=today).all()
    total_intake = sum(entry.amount for entry in water_entries)
    
    return jsonify({
        'total_intake': total_intake,
        'goal': 2000,  # 2 liters per day
        'last_drink': water_entries[-1].timestamp.strftime('%H:%M') if water_entries else 'No drinks today'
    })

@app.route('/api/add_water', methods=['POST'])
def add_water():
    data = request.json
    amount = data.get('amount', 250)  # Default to 250ml
    
    new_intake = models.WaterIntake(
        amount=amount,
        date=datetime.now().date(),
        timestamp=datetime.now()
    )
    db.session.add(new_intake)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    frame_file = request.files['frame']
    frame = camera_processor.process_image_file(frame_file)
    
    if frame is None:
        return jsonify({'error': 'Invalid frame'}), 400
    
    # Analyze posture
    posture_results = posture_analyzer.analyze(frame)
    is_present = posture_results['is_present']
    posture = posture_results['posture']
    angle = posture_results['angle']
    
    # Calculate focus
    is_focused = focus_calculator.is_focused(is_present, posture)
    
    # Save posture status
    new_posture = models.PostureStatus(
        posture=posture,
        angle=angle,
        timestamp=datetime.now()
    )
    db.session.add(new_posture)
    
    # Save focus score
    new_focus = models.FocusScore(
        is_focused=is_focused,
        is_present=is_present,
        date=datetime.now().date(),
        timestamp=datetime.now()
    )
    db.session.add(new_focus)
    db.session.commit()
    
    return jsonify({
        'posture': posture,
        'is_focused': is_focused,
        'angle': angle,
        'is_present': is_present
    })

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500
