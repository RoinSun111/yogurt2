#!/usr/bin/env python3
"""
Initialize sample calendar events and todos for demonstration
"""

import os
import sys
from datetime import datetime, timedelta, time
from app import app, db
from models import CalendarEvent, TodoItem

def create_sample_calendar_events():
    """Create sample calendar events for demonstration"""
    
    # Clear existing sample events
    CalendarEvent.query.filter(CalendarEvent.title.like('Sample%')).delete()
    
    today = datetime.now().date()
    
    # Today's events
    events = [
        {
            'title': 'Team Standup',
            'description': 'Daily team synchronization meeting',
            'start_time': datetime.combine(today, time(9, 0)),
            'end_time': datetime.combine(today, time(9, 30)),
            'event_type': 'meeting',
            'location': 'Conference Room A'
        },
        {
            'title': 'Project Review',
            'description': 'Review project progress with stakeholders',
            'start_time': datetime.combine(today, time(14, 30)),
            'end_time': datetime.combine(today, time(15, 30)),
            'event_type': 'meeting',
            'location': 'Meeting Room B'
        },
        {
            'title': 'Code Review Session',
            'description': 'Review pending pull requests',
            'start_time': datetime.combine(today, time(16, 0)),
            'end_time': datetime.combine(today, time(17, 0)),
            'event_type': 'meeting',
            'location': 'Development Office'
        }
    ]
    
    # Tomorrow's events
    tomorrow = today + timedelta(days=1)
    events.extend([
        {
            'title': 'Client Call',
            'description': 'Weekly check-in with client',
            'start_time': datetime.combine(tomorrow, time(10, 0)),
            'end_time': datetime.combine(tomorrow, time(11, 0)),
            'event_type': 'meeting',
            'location': 'Video Conference'
        },
        {
            'title': 'Sprint Planning',
            'description': 'Plan next sprint objectives',
            'start_time': datetime.combine(tomorrow, time(13, 0)),
            'end_time': datetime.combine(tomorrow, time(14, 0)),
            'event_type': 'meeting',
            'location': 'Planning Room'
        }
    ])
    
    # This week's events
    for i in range(2, 7):
        future_date = today + timedelta(days=i)
        events.append({
            'title': f'Sample Meeting {i}',
            'description': f'Sample meeting for {future_date.strftime("%A")}',
            'start_time': datetime.combine(future_date, time(11, 0)),
            'end_time': datetime.combine(future_date, time(12, 0)),
            'event_type': 'meeting',
            'location': 'Office'
        })
    
    # Add events to database
    for event_data in events:
        event = CalendarEvent(**event_data)
        db.session.add(event)
    
    print(f"Created {len(events)} sample calendar events")

def create_sample_todos():
    """Create sample todos with calendar integration"""
    
    # Clear existing sample todos
    TodoItem.query.filter(TodoItem.title.like('Sample%')).delete()
    
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    # Create some todos for today
    todos = [
        {
            'title': 'Complete project documentation',
            'description': 'Finish writing technical documentation',
            'priority': 'high',
            'due_date': today,
            'due_time': time(17, 0)
        },
        {
            'title': 'Review code changes',
            'description': 'Review and approve pending pull requests',
            'priority': 'medium',
            'due_date': today,
            'due_time': time(15, 30)
        },
        {
            'title': 'Update project timeline',
            'description': 'Adjust timeline based on recent progress',
            'priority': 'medium',
            'due_date': tomorrow,
            'due_time': time(10, 0)
        },
        {
            'title': 'Prepare client presentation',
            'description': 'Create slides for upcoming client meeting',
            'priority': 'high',
            'due_date': tomorrow,
            'due_time': time(9, 0)
        }
    ]
    
    # Add todos to database
    for todo_data in todos:
        # Create calendar event for todos with specific times
        if todo_data.get('due_time'):
            start_datetime = datetime.combine(todo_data['due_date'], todo_data['due_time'])
            calendar_event = CalendarEvent(
                title=f"TODO: {todo_data['title']}",
                description=todo_data['description'],
                start_time=start_datetime,
                event_type='deadline'
            )
            db.session.add(calendar_event)
            db.session.flush()
            todo_data['calendar_event_id'] = calendar_event.id
        
        todo = TodoItem(**todo_data)
        db.session.add(todo)
    
    print(f"Created {len(todos)} sample todos with calendar integration")

def main():
    """Initialize sample data"""
    with app.app_context():
        print("Initializing sample calendar events and todos...")
        
        create_sample_calendar_events()
        create_sample_todos()
        
        db.session.commit()
        print("Sample data initialization complete!")

if __name__ == '__main__':
    main()