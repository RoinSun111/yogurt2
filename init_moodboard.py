#!/usr/bin/env python3
"""
Initialize default moodboard widgets and sample data
"""

from app import app, db
import models
from datetime import datetime

def init_default_widgets():
    """Initialize default moodboard widget configurations"""
    with app.app_context():
        # Check if widgets already exist
        existing_widgets = models.MoodboardSettings.query.count()
        if existing_widgets > 0:
            print("Moodboard widgets already initialized")
            return
        
        # Create default widget configurations
        default_widgets = [
            {
                'widget_type': 'mood',
                'is_enabled': True,
                'position_x': 0,
                'position_y': 0,
                'width': 1,
                'height': 1
            },
            {
                'widget_type': 'calendar',
                'is_enabled': True,
                'position_x': 1,
                'position_y': 0,
                'width': 1,
                'height': 1
            },
            {
                'widget_type': 'todo',
                'is_enabled': True,
                'position_x': 2,
                'position_y': 0,
                'width': 1,
                'height': 1
            },
            {
                'widget_type': 'focus_score',
                'is_enabled': True,
                'position_x': 0,
                'position_y': 1,
                'width': 1,
                'height': 1
            },
            {
                'widget_type': 'water',
                'is_enabled': True,
                'position_x': 1,
                'position_y': 1,
                'width': 1,
                'height': 1
            },
            {
                'widget_type': 'posture_score',
                'is_enabled': False,
                'position_x': 2,
                'position_y': 1,
                'width': 1,
                'height': 1
            }
        ]
        
        for widget_config in default_widgets:
            widget = models.MoodboardSettings(**widget_config)
            db.session.add(widget)
        
        # Add a sample mood entry for today
        today = datetime.now().date()
        existing_mood = models.MoodEntry.query.filter_by(date=today).first()
        if not existing_mood:
            sample_mood = models.MoodEntry(
                mood_text="Ready to focus and be productive!",
                date=today
            )
            db.session.add(sample_mood)
        
        # Add sample todo items
        existing_todos = models.TodoItem.query.count()
        if existing_todos == 0:
            sample_todos = [
                {
                    'title': 'Review daily focus metrics',
                    'description': 'Check progress on productivity goals',
                    'priority': 'high'
                },
                {
                    'title': 'Take posture breaks',
                    'description': 'Stand and stretch every hour',
                    'priority': 'medium'
                },
                {
                    'title': 'Stay hydrated',
                    'description': 'Drink 8 glasses of water today',
                    'priority': 'medium'
                }
            ]
            
            for todo_data in sample_todos:
                todo = models.TodoItem(**todo_data)
                db.session.add(todo)
        
        db.session.commit()
        print("✓ Default moodboard widgets and sample data initialized")

if __name__ == '__main__':
    init_default_widgets()