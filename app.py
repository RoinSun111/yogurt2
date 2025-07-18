import os
import logging
import json
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, Response, session
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

# Import AI calendar utilities
from utils import ai_calendar
from utils.voice_processor import voice_processor

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
    
    # Calculate health score for gamification
    focus_entries = models.FocusScore.query.filter_by(date=today).all()
    posture_entries = models.PostureStatus.query.filter(
        models.PostureStatus.timestamp >= datetime.combine(today, datetime.min.time())
    ).all()
    water_entries = models.WaterIntake.query.filter_by(date=today).all()
    
    # Calculate component scores
    focus_score = 0
    if focus_entries:
        focused_entries = sum(1 for entry in focus_entries if entry.is_focused)
        focus_score = (focused_entries / len(focus_entries)) * 100
    
    posture_score = 0
    if posture_entries:
        good_posture_count = sum(1 for entry in posture_entries 
                               if getattr(entry, 'posture_quality', 'unknown') in ['excellent', 'good'])
        posture_score = (good_posture_count / len(posture_entries)) * 100
    
    water_score = 0
    if water_entries:
        total_water = sum(entry.amount for entry in water_entries)
        water_score = min(100, (total_water / 2000) * 100)  # 2000ml daily goal
    
    # Distraction penalty
    distraction_penalty = min(50, len(distraction_events) * 5)  # Up to 50 points penalty
    
    # Calculate overall health score
    health_score = max(0, (focus_score * 0.4 + posture_score * 0.3 + water_score * 0.3) - distraction_penalty)
    
    # Handle achievements and streaks
    is_elite = health_score > 95
    achievements_earned = []
    
    # Check and update elite performer streak
    elite_streak = models.DailyStreak.query.filter_by(streak_type='elite_performer').first()
    if not elite_streak:
        elite_streak = models.DailyStreak(streak_type='elite_performer')
        db.session.add(elite_streak)
    
    if is_elite:
        if elite_streak.last_update_date != today:
            elite_streak.current_streak += 1
            elite_streak.longest_streak = max(elite_streak.longest_streak, elite_streak.current_streak)
            elite_streak.last_update_date = today
            
            # Award achievements for streaks
            if elite_streak.current_streak == 1:
                achievement = models.Achievement(
                    achievement_type='elite_performer',
                    score=health_score
                )
                db.session.add(achievement)
                achievements_earned.append('Elite Performer')
            elif elite_streak.current_streak == 3:
                achievement = models.Achievement(
                    achievement_type='elite_streak_3',
                    score=health_score,
                    streak_count=3
                )
                db.session.add(achievement)
                achievements_earned.append('3-Day Elite Streak')
            elif elite_streak.current_streak == 7:
                achievement = models.Achievement(
                    achievement_type='elite_streak_7',
                    score=health_score,
                    streak_count=7
                )
                db.session.add(achievement)
                achievements_earned.append('Week Champion')
    else:
        # Reset streak if not elite
        if elite_streak.last_update_date != today:
            elite_streak.current_streak = 0
            elite_streak.last_update_date = today
    
    # Check for perfect posture achievement
    if posture_score == 100:
        today_perfect_posture = models.Achievement.query.filter_by(
            achievement_type='perfect_posture',
            date_earned=today
        ).first()
        if not today_perfect_posture:
            achievement = models.Achievement(
                achievement_type='perfect_posture',
                score=posture_score
            )
            db.session.add(achievement)
            achievements_earned.append('Perfect Posture')
    
    # Check for hydration hero achievement
    if water_score >= 100:
        today_hydration = models.Achievement.query.filter_by(
            achievement_type='hydration_hero',
            date_earned=today
        ).first()
        if not today_hydration:
            achievement = models.Achievement(
                achievement_type='hydration_hero',
                score=water_score
            )
            db.session.add(achievement)
            achievements_earned.append('Hydration Hero')
    
    db.session.commit()
    
    # Get recent achievements for display
    recent_achievements = models.Achievement.query.filter(
        models.Achievement.date_earned >= today - timedelta(days=7)
    ).order_by(models.Achievement.timestamp.desc()).limit(5).all()
    
    # Get historical health scores (last 30 days)
    health_history = []
    for i in range(29, -1, -1):  # Last 30 days including today
        date = today - timedelta(days=i)
        
        # Get data for this date
        daily_focus = models.FocusScore.query.filter_by(date=date).all()
        daily_posture = models.PostureStatus.query.filter(
            models.PostureStatus.timestamp >= datetime.combine(date, datetime.min.time()),
            models.PostureStatus.timestamp < datetime.combine(date + timedelta(days=1), datetime.min.time())
        ).all()
        daily_water = models.WaterIntake.query.filter_by(date=date).all()
        daily_distractions = models.DistractionEvent.query.filter_by(date=date).all()
        
        # Calculate daily scores
        daily_focus_score = 0
        if daily_focus:
            focused_count = sum(1 for entry in daily_focus if entry.is_focused)
            daily_focus_score = (focused_count / len(daily_focus)) * 100
        
        daily_posture_score = 0
        if daily_posture:
            good_posture_count = sum(1 for entry in daily_posture 
                                   if getattr(entry, 'posture_quality', 'unknown') in ['excellent', 'good'])
            daily_posture_score = (good_posture_count / len(daily_posture)) * 100
        
        daily_water_score = 0
        if daily_water:
            total_water = sum(entry.amount for entry in daily_water)
            daily_water_score = min(100, (total_water / 2000) * 100)
        
        daily_distraction_penalty = min(50, len(daily_distractions) * 5)
        daily_health_score = max(0, (daily_focus_score * 0.4 + daily_posture_score * 0.3 + daily_water_score * 0.3) - daily_distraction_penalty)
        
        health_history.append({
            'date': date.strftime('%Y-%m-%d'),
            'date_short': date.strftime('%m/%d'),
            'health_score': round(daily_health_score, 1),
            'focus_score': round(daily_focus_score, 1),
            'posture_score': round(daily_posture_score, 1),
            'water_score': round(daily_water_score, 1),
            'is_elite': daily_health_score > 95
        })
    
    # Get activity summary
    activity_summary = {
        'distraction_count': len(events_by_type['distraction']),
        'away_count': len(events_by_type['away']),
        'talking_count': len(events_by_type['talking']),
        'phone_use_count': len(events_by_type['phone_use']),
        'total_distraction_time': sum([e.duration or 0 for e in distraction_events]),
        'date': today.strftime('%Y-%m-%d'),
        'health_score': round(health_score, 1),
        'focus_score': round(focus_score, 1),
        'posture_score': round(posture_score, 1),
        'water_score': round(water_score, 1),
        'is_elite': is_elite,
        'elite_streak': elite_streak.current_streak,
        'longest_streak': elite_streak.longest_streak,
        'achievements_earned': achievements_earned,
        'recent_achievements': recent_achievements,
        'health_history': health_history
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
            'neck_angle': 0,
            'shoulder_alignment': 0,
            'head_forward_position': 0,
            'spine_curvature': 0,
            'symmetry_score': 0,
            'feedback': "Posture data is being updated. Please refresh in a moment.",
            'trend': 'neutral',
            'recommendation': 'Please position yourself in front of the camera.'
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
    
    # Analyze posture and activity
    analysis = posture_analyzer.analyze(frame)
    is_present = analysis['is_present']
    posture = analysis['posture']
    angle = analysis['angle']
    
    # Get activity state and head angle from analysis
    activity_state = analysis.get('activity', {}).get('activity_state', 'unknown')
    head_angle = analysis.get('activity', {}).get('head_angle', 0)
    
    # Calculate focus with enhanced logic
    is_focused = focus_calculator.is_focused(is_present, posture, activity_state, head_angle)
    
    # Save posture status
    try:
        # Try to save with all new attributes (in case migration has been done)
        if isinstance(analysis, dict) and analysis.get('posture_quality'):
            new_posture = models.PostureStatus(
                posture=posture,
                posture_quality=analysis.get('posture_quality', 'unknown'),
                angle=angle,
                neck_angle=analysis.get('neck_angle', 0.0),
                shoulder_alignment=analysis.get('shoulder_alignment', 0.0),
                head_forward_position=analysis.get('head_forward_position', 0.0),
                spine_curvature=analysis.get('spine_curvature', 0.0),
                symmetry_score=analysis.get('symmetry_score', 0.0),
                feedback=analysis.get('feedback', None),
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
    
    # Get activity data from analysis (already calculated in posture_analyzer.analyze)
    activity_data = analysis.get('activity', {
        'activity_state': 'unknown',
        'working_substate': None,
        'head_angle': 0.0,
        'movement_level': 0.0,
        'people_detected': 1 if is_present else 0,
        'time_in_state': 0
    })
    
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
        # Include advanced posture metrics if available
        'posture_quality': analysis.get('posture_quality', 'unknown'),
        'neck_angle': float(analysis.get('neck_angle', 0.0)),
        'shoulder_alignment': float(analysis.get('shoulder_alignment', 0.0)),
        'head_forward_position': float(analysis.get('head_forward_position', 0.0)),
        'spine_curvature': float(analysis.get('spine_curvature', 0.0)),
        'symmetry_score': float(analysis.get('symmetry_score', 0.0)),
        'feedback': analysis.get('feedback', None),
        # Include pose landmarks for frontend visualization
        'pose_landmarks': analysis.get('pose_landmarks', []),
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

# Moodboard routes
@app.route('/moodboard')
def moodboard():
    """Display the moodboard with configured widgets"""
    # Get widget configurations
    widgets = models.MoodboardSettings.query.filter_by(is_enabled=True).all()
    
    # Get current mood
    current_mood = models.MoodEntry.query.filter_by(date=datetime.now().date()).first()
    
    # Get todos
    todos = models.TodoItem.query.filter_by(is_completed=False).order_by(models.TodoItem.priority.desc()).limit(5).all()
    
    # Get focus score for today
    today = datetime.now().date()
    focus_entries = models.FocusScore.query.filter_by(date=today).all()
    
    total_entries = len(focus_entries)
    working_entries = sum(1 for entry in focus_entries if entry.is_focused)
    focus_score = (working_entries / total_entries) * 100 if total_entries > 0 else 0
    
    # Get water intake for today
    water_entries = models.WaterIntake.query.filter_by(date=today).all()
    total_water = sum(entry.amount for entry in water_entries)
    
    # Get latest posture
    latest_posture = models.PostureStatus.query.order_by(models.PostureStatus.timestamp.desc()).first()
    
    return render_template('moodboard/display.html', 
                         widgets=widgets,
                         current_mood=current_mood,
                         todos=todos,
                         focus_score=round(focus_score, 1),
                         total_water=total_water,
                         latest_posture=latest_posture)

@app.route('/moodboard/customize')
def moodboard_customize():
    """Display the moodboard customization page"""
    widgets = models.MoodboardSettings.query.all()
    return render_template('moodboard/customize.html', widgets=widgets)

@app.route('/api/moodboard/mood', methods=['POST'])
def update_mood():
    """Update the current mood text"""
    data = request.get_json()
    mood_text = data.get('mood_text', '')
    
    today = datetime.now().date()
    existing_mood = models.MoodEntry.query.filter_by(date=today).first()
    
    if existing_mood:
        existing_mood.mood_text = mood_text
        existing_mood.timestamp = datetime.now()
    else:
        new_mood = models.MoodEntry(mood_text=mood_text, date=today)
        db.session.add(new_mood)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/moodboard/calendar/events')
def get_calendar_events():
    """Get calendar events for a specific date or today"""
    try:
        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            target_date = datetime.now().date()
            
        events = models.CalendarEvent.query.filter(
            db.func.date(models.CalendarEvent.start_time) == target_date
        ).order_by(models.CalendarEvent.start_time).all()
        
        events_data = []
        for event in events:
            events_data.append({
                'id': event.id,
                'title': event.title,
                'time': event.start_time.strftime('%H:%M') if not event.is_all_day else 'All Day',
                'end_time': event.end_time.strftime('%H:%M') if event.end_time and not event.is_all_day else None,
                'type': event.event_type,
                'location': event.location
            })
        
        return jsonify({'success': True, 'events': events_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/moodboard/calendar/events/<int:event_id>', methods=['PUT'])
def update_calendar_event(event_id):
    """Update a calendar event"""
    try:
        data = request.get_json()
        event = models.CalendarEvent.query.get_or_404(event_id)
        
        # Update event fields
        if 'title' in data:
            event.title = data['title']
        if 'description' in data:
            event.description = data['description']
        if 'location' in data:
            event.location = data['location']
        if 'time' in data and data['time']:
            # Update time while keeping the same date
            new_time = datetime.strptime(data['time'], '%H:%M').time()
            event.start_time = datetime.combine(event.start_time.date(), new_time)
            # Update end time if it exists (add 1 hour by default)
            if event.end_time:
                event.end_time = datetime.combine(event.start_time.date(), 
                                                 (datetime.combine(datetime.min, new_time) + timedelta(hours=1)).time())
        
        event.updated_at = datetime.now()
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Event updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/moodboard/calendar/events/<int:event_id>', methods=['DELETE'])
def delete_calendar_event(event_id):
    """Delete a calendar event"""
    try:
        event = models.CalendarEvent.query.get_or_404(event_id)
        
        # Also delete any linked todos
        linked_todos = models.TodoItem.query.filter_by(calendar_event_id=event_id).all()
        for todo in linked_todos:
            todo.calendar_event_id = None  # Unlink instead of deleting the todo
        
        db.session.delete(event)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Event deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/moodboard/todos')
def get_todos():
    """Get todos, optionally filtered by date"""
    try:
        date_str = request.args.get('date')
        completed = request.args.get('completed', 'false').lower() == 'true'
        
        query = models.TodoItem.query
        
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(models.TodoItem.due_date == target_date)
        
        if not completed:
            query = query.filter(models.TodoItem.is_completed == False)
        
        todos = query.order_by(models.TodoItem.created_at.desc()).all()
        
        todos_data = []
        for todo in todos:
            todos_data.append({
                'id': todo.id,
                'title': todo.title,
                'description': todo.description,
                'is_completed': todo.is_completed,
                'priority': todo.priority,
                'due_date': todo.due_date.strftime('%Y-%m-%d') if todo.due_date else None,
                'due_time': todo.due_time.strftime('%H:%M') if todo.due_time else None,
                'has_calendar_event': todo.calendar_event_id is not None
            })
        
        return jsonify({'success': True, 'todos': todos_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/moodboard/todo', methods=['POST'])
def add_todo():
    """Add a new todo item with optional calendar integration"""
    try:
        data = request.get_json()
        
        # Parse due date and time
        due_date = None
        due_time = None
        if data.get('due_date'):
            due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date()
        if data.get('due_time'):
            due_time = datetime.strptime(data['due_time'], '%H:%M').time()
        
        # Create calendar event if both date and time are provided and requested
        calendar_event_id = None
        if due_date and due_time and data.get('create_calendar_event'):
            start_datetime = datetime.combine(due_date, due_time)
            calendar_event = models.CalendarEvent(
                title=f"TODO: {data.get('title', '')}",
                description=data.get('description', ''),
                start_time=start_datetime,
                event_type='deadline'
            )
            db.session.add(calendar_event)
            db.session.flush()
            calendar_event_id = calendar_event.id
        
        new_todo = models.TodoItem(
            title=data.get('title', ''),
            description=data.get('description', ''),
            priority=data.get('priority', 'medium'),
            due_date=due_date,
            due_time=due_time,
            calendar_event_id=calendar_event_id
        )
        db.session.add(new_todo)
        db.session.commit()
        
        return jsonify({'success': True, 'id': new_todo.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/moodboard/todo/<int:todo_id>/complete', methods=['POST'])
def complete_todo(todo_id):
    """Mark a todo as completed"""
    todo = models.TodoItem.query.get_or_404(todo_id)
    todo.is_completed = True
    todo.completed_at = datetime.now()
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/moodboard/widgets', methods=['GET'])
def get_widget_settings():
    """Get current widget configuration"""
    widgets = models.MoodboardSettings.query.all()
    
    widget_list = []
    for widget in widgets:
        widget_list.append({
            'type': widget.widget_type,
            'enabled': widget.is_enabled,
            'x': widget.position_x,
            'y': widget.position_y,
            'width': widget.width,
            'height': widget.height,
            'config': widget.config
        })
    
    return jsonify({
        'success': True,
        'widgets': widget_list
    })

@app.route('/api/moodboard/widgets', methods=['POST'])
def update_widget_settings():
    """Update widget configuration"""
    data = request.get_json()
    
    # Clear existing widgets first
    models.MoodboardSettings.query.delete()
    
    # Add new widgets from the configuration
    for widget_data in data.get('widgets', []):
        new_widget = models.MoodboardSettings(
            widget_type=widget_data['type'],
            is_enabled=widget_data.get('enabled', True),
            position_x=widget_data.get('x', 0),
            position_y=widget_data.get('y', 0),
            width=widget_data.get('width', 1),
            height=widget_data.get('height', 1),
            config=widget_data.get('config', '')
        )
        db.session.add(new_widget)
    
    db.session.commit()
    return jsonify({'success': True})

# AI Calendar Routes
@app.route('/api/calendar/events')
def get_ai_calendar_events():
    """Get calendar events for AI calendar interface"""
    start_date = request.args.get('start', datetime.now().strftime('%Y-%m-%d'))
    end_date = request.args.get('end', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        
        events = models.CalendarEvent.query.filter(
            models.CalendarEvent.start_time >= start_dt,
            models.CalendarEvent.start_time < end_dt
        ).order_by(models.CalendarEvent.start_time).all()
        
        events_data = []
        for event in events:
            events_data.append({
                'id': event.id,
                'title': event.title,
                'description': event.description,
                'start': event.start_time.isoformat(),
                'end': event.end_time.isoformat() if event.end_time else None,
                'type': event.event_type,
                'location': event.location,
                'color': event.color,
                'priority': event.priority,
                'is_ai_created': event.is_ai_created
            })
        
        return jsonify({'success': True, 'events': events_data})
    except Exception as e:
        logging.error(f"Error fetching events: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calendar/ai-chat', methods=['POST'])
def ai_calendar_chat():
    """Handle AI calendar conversation"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = session.get('calendar_session_id')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            session['calendar_session_id'] = session_id
        
        # Parse the natural language input
        event_request = ai_calendar.parse_natural_language(user_message)
        
        response_text = ""
        created_event = None
        
        # Handle different actions
        if event_request.action == 'create' and event_request.confidence > 0.5:
            # Check for conflicts
            existing_events = []
            if event_request.start_time:
                start_dt = datetime.fromisoformat(event_request.start_time)
                day_start = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)
                
                existing = models.CalendarEvent.query.filter(
                    models.CalendarEvent.start_time >= day_start,
                    models.CalendarEvent.start_time < day_end
                ).all()
                
                existing_events = [{
                    'title': e.title,
                    'start_time': e.start_time.isoformat(),
                    'end_time': e.end_time.isoformat() if e.end_time else None
                } for e in existing]
            
            conflict_info = ai_calendar.detect_conflicts(event_request, existing_events)
            
            if conflict_info.has_conflict:
                response_text = f"I found a scheduling conflict: {conflict_info.reasoning}\n\nWould you like me to suggest alternative times?"
            else:
                # Create the event
                start_time = datetime.fromisoformat(event_request.start_time) if event_request.start_time else None
                end_time = None
                
                if event_request.end_time:
                    end_time = datetime.fromisoformat(event_request.end_time)
                elif start_time and event_request.duration:
                    end_time = start_time + timedelta(minutes=event_request.duration)
                elif start_time:
                    end_time = start_time + timedelta(hours=1)  # Default 1 hour
                
                if start_time:
                    new_event = models.CalendarEvent(
                        title=event_request.title,
                        description=event_request.description,
                        start_time=start_time,
                        end_time=end_time,
                        location=event_request.location,
                        event_type=event_request.event_type,
                        attendees=json.dumps(event_request.attendees) if event_request.attendees else None,
                        is_ai_created=True,
                        priority='medium'
                    )
                    
                    db.session.add(new_event)
                    db.session.commit()
                    created_event = new_event
                    
                    response_text = f"Perfect! I've scheduled '{event_request.title}' for {start_time.strftime('%B %d at %I:%M %p')}."
                else:
                    response_text = "I'd be happy to schedule that for you! What time would you prefer?"
        
        elif event_request.action == 'list':
            # List upcoming events
            upcoming = models.CalendarEvent.query.filter(
                models.CalendarEvent.start_time >= datetime.now()
            ).order_by(models.CalendarEvent.start_time).limit(5).all()
            
            if upcoming:
                event_list = []
                for event in upcoming:
                    event_list.append(f"• {event.title} - {event.start_time.strftime('%B %d at %I:%M %p')}")
                response_text = "Here are your upcoming events:\n" + "\n".join(event_list)
            else:
                response_text = "You don't have any upcoming events scheduled."
        
        else:
            # General conversation
            context = {
                'recent_events': [],
                'current_time': datetime.now().isoformat()
            }
            response_text = ai_calendar.handle_conversation(user_message, context)
        
        # Save conversation
        conversation = models.AIConversation(
            session_id=session_id,
            user_message=user_message,
            ai_response=response_text,
            intent=event_request.action,
            confidence_score=event_request.confidence,
            event_id=created_event.id if created_event else None
        )
        db.session.add(conversation)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': response_text,
            'intent': event_request.action,
            'confidence': event_request.confidence,
            'event_created': created_event.id if created_event else None
        })
        
    except Exception as e:
        logging.error(f"Error in AI calendar chat: {e}")
        return jsonify({'success': False, 'error': 'Sorry, I had trouble processing that request. Could you try again?'})

@app.route('/api/calendar/voice', methods=['POST'])
def process_voice_command():
    """Process voice commands for calendar"""
    try:
        # Get audio data from request
        audio_data = request.files.get('audio')
        if not audio_data:
            return jsonify({'success': False, 'error': 'No audio data provided'})
        
        # Process audio (placeholder - in real implementation, use speech-to-text)
        transcript = "Schedule a meeting with John tomorrow at 2 PM"  # Placeholder
        
        # Parse the voice command
        event_request = ai_calendar.extract_voice_command(transcript)
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'intent': event_request.action,
            'confidence': event_request.confidence,
            'event_data': event_request.dict()
        })
        
    except Exception as e:
        logging.error(f"Error processing voice command: {e}")
        return jsonify({'success': False, 'error': 'Error processing voice command'})

@app.route('/api/calendar/voice-schedule', methods=['POST'])
def process_voice_schedule():
    """Process voice commands for moodboard calendar quick schedule"""
    try:
        # Get audio data from request
        audio_data = request.files.get('audio')
        if not audio_data:
            return jsonify({'success': False, 'error': 'No audio data provided'})
        
        # Read audio data
        audio_bytes = audio_data.read()
        
        # Process audio using voice processor
        transcript = voice_processor.process_audio_blob(audio_bytes)
        
        if transcript:
            return jsonify({
                'success': True,
                'transcript': transcript
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not process audio'
            })
        
    except Exception as e:
        logging.error(f"Error processing voice schedule: {e}")
        return jsonify({'success': False, 'error': 'Error processing voice input'})

@app.route('/calendar')
def ai_calendar_view():
    """Render the AI calendar page"""
    return render_template('ai_calendar.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500
