import os
import logging
from datetime import datetime, timedelta
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
from utils.activity_analyzer import ActivityAnalyzer

# Initialize the processors
camera_processor = CameraProcessor()
posture_analyzer = PostureAnalyzer()
focus_calculator = FocusCalculator()
activity_analyzer = ActivityAnalyzer()

with app.app_context():
    # Import models here
    import models
    db.create_all()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/activity-records')
def activity_records():
    """Render the activity records page with distraction tracking"""
    # Get distraction events from today
    today = datetime.now().date()
    distraction_events = models.DistractionEvent.query.filter_by(date=today).order_by(models.DistractionEvent.start_time.desc()).all()
    
    # Group events by type
    events_by_type = {
        'distraction': [],
        'away': [],
        'talking': [],
        'phone_use': []
    }
    
    for event in distraction_events:
        if event.event_type in events_by_type:
            events_by_type[event.event_type].append(event)
        else:
            events_by_type['distraction'].append(event)
    
    # Get activity summary
    activity_summary = {
        'distraction_count': len(events_by_type['distraction']),
        'away_count': len(events_by_type['away']),
        'talking_count': len(events_by_type['talking']),
        'phone_use_count': len(events_by_type['phone_use']),
        'total_distraction_time': sum([e.duration or 0 for e in distraction_events]),
        'date': today.strftime('%Y-%m-%d')
    }
    
    return render_template('activity_records.html', 
                          events=distraction_events, 
                          events_by_type=events_by_type,
                          summary=activity_summary)
                          
@app.route('/api/demo-distractions', methods=['POST'])
def demo_distractions():
    """Generate demo distraction data for testing the UI"""
    # Clear any existing data
    models.DistractionEvent.query.delete()
    db.session.commit()
    
    # Current time for realistic timestamps
    now = datetime.now()
    today = now.date()
    
    # Demo events with realistic timings and durations
    demo_events = [
        # Phone usage events
        {
            'event_type': 'phone_use',
            'start_time': now - timedelta(minutes=30),
            'end_time': now - timedelta(minutes=28),
            'details': 'User checking phone',
            'date': today
        },
        {
            'event_type': 'phone_use',
            'start_time': now - timedelta(hours=2, minutes=15),
            'end_time': now - timedelta(hours=2, minutes=13),
            'details': 'User checking phone',
            'date': today
        },
        {
            'event_type': 'phone_use',
            'start_time': now - timedelta(hours=4, minutes=5),
            'end_time': now - timedelta(hours=4),
            'details': 'User checking phone',
            'date': today
        },
        
        # Away from desk events
        {
            'event_type': 'away',
            'start_time': now - timedelta(hours=1),
            'end_time': now - timedelta(minutes=50),
            'details': 'User left desk',
            'date': today
        },
        {
            'event_type': 'away',
            'start_time': now - timedelta(hours=3, minutes=30),
            'end_time': now - timedelta(hours=3, minutes=20),
            'details': 'User left desk',
            'date': today
        },
        
        # Talking to others
        {
            'event_type': 'talking',
            'start_time': now - timedelta(minutes=45),
            'end_time': now - timedelta(minutes=40),
            'details': 'User talking to others',
            'date': today
        },
        {
            'event_type': 'talking',
            'start_time': now - timedelta(hours=2, minutes=45),
            'end_time': now - timedelta(hours=2, minutes=40),
            'details': 'User talking to others',
            'date': today
        },
        
        # General distractions
        {
            'event_type': 'distraction',
            'start_time': now - timedelta(hours=1, minutes=15),
            'end_time': now - timedelta(hours=1, minutes=13),
            'details': 'User became distracted',
            'date': today
        },
        {
            'event_type': 'distraction',
            'start_time': now - timedelta(hours=3),
            'end_time': now - timedelta(hours=2, minutes=57),
            'details': 'User became distracted',
            'date': today
        }
    ]
    
    # Add events to database
    for event_data in demo_events:
        event = models.DistractionEvent(**event_data)
        # Calculate duration
        if event.end_time and event.start_time:
            event.duration = int((event.end_time - event.start_time).total_seconds())
        db.session.add(event)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Demo distraction data created successfully',
        'event_count': len(demo_events)
    })

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
    # Use try/except to handle potential missing columns during migration
    try:
        posture_status = models.PostureStatus.query.order_by(models.PostureStatus.timestamp.desc()).first()
        if not posture_status:
            return jsonify({
                'posture': 'unknown', 
                'angle': 0,
                'posture_quality': 'unknown',
                'neck_angle': 0,
                'shoulder_alignment': 0,
                'head_forward_position': 0,
                'spine_curvature': 0,
                'symmetry_score': 0,
                'feedback': None
            })
        
        # Get posture trend data for the last hour
        posture_trend = posture_analyzer.get_posture_trend(minutes=60)
        
        return jsonify({
            'posture': posture_status.posture,
            'posture_quality': getattr(posture_status, 'posture_quality', 'unknown'),
            'angle': posture_status.angle,
            'neck_angle': getattr(posture_status, 'neck_angle', 0),
            'shoulder_alignment': getattr(posture_status, 'shoulder_alignment', 0),
            'head_forward_position': getattr(posture_status, 'head_forward_position', 0),
            'spine_curvature': getattr(posture_status, 'spine_curvature', 0),
            'symmetry_score': getattr(posture_status, 'symmetry_score', 0),
            'feedback': getattr(posture_status, 'feedback', None),
            'trend': posture_trend.get('trend', 'neutral'),
            'recommendation': posture_trend.get('recommendation', '')
        })
    except Exception as e:
        logging.error(f"Error retrieving posture data: {str(e)}")
        # Fallback to basic data if columns don't exist yet
        return jsonify({
            'posture': 'unknown', 
            'angle': 0,
            'posture_quality': 'unknown',
            'feedback': "Posture data is being updated. Please refresh in a moment."
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

@app.route('/api/activity_status')
def get_activity_status():
    """Get the current activity status of the user"""
    activity_status = models.ActivityStatus.query.order_by(models.ActivityStatus.timestamp.desc()).first()
    
    if not activity_status:
        return jsonify({
            'activity_state': 'unknown',
            'working_substate': None,
            'head_angle': 0,
            'movement_level': 0,
            'people_detected': 0,
            'time_in_state': 0
        })
    
    return jsonify({
        'activity_state': activity_status.activity_state,
        'working_substate': activity_status.working_substate,
        'head_angle': activity_status.head_angle,
        'movement_level': activity_status.movement_level,
        'people_detected': activity_status.people_detected,
        'time_in_state': activity_status.time_in_state
    })

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
    try:
        # Try to save with all new attributes (in case migration has been done)
        if isinstance(posture_results, dict) and posture_results.get('posture_quality'):
            new_posture = models.PostureStatus(
                posture=posture,
                posture_quality=posture_results.get('posture_quality', 'unknown'),
                angle=angle,
                neck_angle=posture_results.get('neck_angle', 0.0),
                shoulder_alignment=posture_results.get('shoulder_alignment', 0.0),
                head_forward_position=posture_results.get('head_forward_position', 0.0),
                spine_curvature=posture_results.get('spine_curvature', 0.0),
                symmetry_score=posture_results.get('symmetry_score', 0.0),
                feedback=posture_results.get('feedback', None),
                timestamp=datetime.now()
            )
        else:
            # Fallback to basic attributes if migration is not complete
            new_posture = models.PostureStatus(
                posture=posture,
                angle=angle,
                timestamp=datetime.now()
            )
        db.session.add(new_posture)
    except Exception as e:
        # If there's a database error, log it and continue with basic data
        logging.error(f"Error saving posture data: {str(e)}")
        # Save only the basic posture data
        try:
            new_posture = models.PostureStatus(
                posture=posture,
                angle=angle,
                timestamp=datetime.now()
            )
            db.session.add(new_posture)
        except:
            logging.error("Failed to save even basic posture data")
    
    # Save focus score
    new_focus = models.FocusScore(
        is_focused=is_focused,
        is_present=is_present,
        date=datetime.now().date(),
        timestamp=datetime.now()
    )
    db.session.add(new_focus)
    
    # Initialize default activity data based on presence status
    if not is_present:
        # User is not at the desk
        activity_data = {
            'activity_state': 'not_at_desk',
            'working_substate': None,
            'head_angle': 0.0,
            'movement_level': 0.0,
            'people_detected': 0,
            'time_in_state': 0
        }
    else:
        # User is present but analyze activity if pose landmarks are available
        if posture_results.get('pose_landmarks'):
            # Get the MediaPipe pose landmarks from posture analyzer's results
            pose_landmarks = posture_results.get('pose_landmarks')
            
            # Analyze activity
            activity_data = activity_analyzer.analyze(
                image=frame, 
                pose_landmarks=pose_landmarks,
                face_detections=None  # Currently not implementing face detection
            )
        else:
            # User is detected but no reliable pose landmarks
            activity_data = {
                'activity_state': 'unknown',
                'working_substate': None,
                'head_angle': 0.0,
                'movement_level': 0.0,
                'people_detected': 1,
                'time_in_state': 0
            }
    
    # Save activity status
    new_activity = models.ActivityStatus(
        activity_state=activity_data['activity_state'],
        working_substate=activity_data['working_substate'],
        head_angle=activity_data['head_angle'],
        movement_level=activity_data['movement_level'],
        people_detected=activity_data['people_detected'],
        time_in_state=activity_data['time_in_state'],
        date=datetime.now().date(),
        timestamp=datetime.now()
    )
    db.session.add(new_activity)
    
    # Track distraction events
    previous_activity = db.session.query(models.ActivityStatus).order_by(
        models.ActivityStatus.id.desc()).offset(1).limit(1).first()
    
    if previous_activity:
        # Handle transition: working → distracted_by_others (talking to someone)
        if previous_activity.activity_state == "working" and activity_data['activity_state'] == "distracted_by_others":
            new_event = models.DistractionEvent(
                event_type="talking",
                start_time=datetime.now(),
                details="User talking to others"
            )
            db.session.add(new_event)
            
        # Handle transition: any state → phone_use (detected phone usage)
        if activity_data['working_substate'] == "phone_use" and previous_activity.working_substate != "phone_use":
            new_event = models.DistractionEvent(
                event_type="phone_use",
                start_time=datetime.now(),
                details="User checking phone"
            )
            db.session.add(new_event)
            
        # Handle transition: any state → not_at_desk (user left)
        if activity_data['activity_state'] == "not_at_desk" and previous_activity.activity_state != "not_at_desk":
            new_event = models.DistractionEvent(
                event_type="away",
                start_time=datetime.now(),
                details="User left desk"
            )
            db.session.add(new_event)
        
        # Close open events when the user returns to working
        if activity_data['activity_state'] == "working" and previous_activity.activity_state != "working":
            open_events = db.session.query(models.DistractionEvent).filter(
                models.DistractionEvent.end_time == None
            ).all()
            
            for event in open_events:
                event.end_time = datetime.now()
                event.duration = int((event.end_time - event.start_time).total_seconds())
    
    db.session.commit()
    
    # Prepare response
    response_data = {
        'posture': posture,
        'is_focused': bool(is_focused),
        'angle': float(angle),
        'is_present': bool(is_present),
        'activity': {
            'activity_state': str(activity_data['activity_state']),
            'working_substate': activity_data['working_substate'],
            'head_angle': float(activity_data['head_angle']),
            'movement_level': float(activity_data['movement_level']),
            'people_detected': int(activity_data['people_detected']),
            'time_in_state': int(activity_data['time_in_state'])
        }
    }
    
    return jsonify(response_data)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500
