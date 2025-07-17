"""
AI Calendar Assistant using Google Gemini API
Handles natural language processing for calendar events
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@dataclass
class EventRequest:
    """Structure for parsed event requests"""
    action: str  # 'create', 'list', 'update', 'delete', 'query'
    title: str = ""
    description: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[int] = None  # in minutes
    location: str = ""
    attendees: List[str] = None
    event_type: str = "meeting"
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []

@dataclass
class ConflictInfo:
    """Information about scheduling conflicts"""
    has_conflict: bool
    conflicting_events: List[Dict]
    reasoning: str
    suggested_times: List[str] = None
    
    def __post_init__(self):
        if self.suggested_times is None:
            self.suggested_times = []

def parse_natural_language(user_input: str) -> EventRequest:
    """
    Parse natural language input into structured event request
    """
    try:
        # Create system prompt for event parsing
        system_prompt = """
        You are an AI calendar assistant. Parse the user's natural language input into a structured event request.
        
        Analyze the input for:
        - Action: create, list, update, delete, or query
        - Event title/description
        - Date and time information
        - Duration if specified
        - Location if mentioned
        - Attendees if mentioned
        - Event type (meeting, appointment, reminder, deadline, etc.)
        
        For dates and times:
        - Convert relative dates like "tomorrow", "next week", "Friday" to ISO format
        - Use current time as reference: {current_time}
        - Default to 1 hour duration if not specified
        - Use 24-hour format for times
        
        Return JSON in this exact format:
        {{
            "action": "create|list|update|delete|query",
            "title": "event title",
            "description": "additional details",
            "start_time": "2025-07-18T14:00:00",
            "end_time": "2025-07-18T15:00:00",
            "duration": 60,
            "location": "location if mentioned",
            "attendees": ["person1", "person2"],
            "event_type": "meeting|appointment|reminder|deadline|task",
            "confidence": 0.8
        }}
        
        Examples:
        - "Schedule a meeting with John tomorrow at 2pm" → action: "create", title: "Meeting with John", start_time: tomorrow 2pm
        - "What do I have on Friday?" → action: "list", confidence: 0.9
        - "Remind me to call mom at 5pm today" → action: "create", title: "Call mom", event_type: "reminder"
        """
        
        current_time = datetime.now().isoformat()
        prompt = system_prompt.format(current_time=current_time)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_input)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=1000
            )
        )
        
        if response.text:
            data = json.loads(response.text)
            return EventRequest(
                action=data.get('action', 'query'),
                title=data.get('title', ''),
                description=data.get('description', ''),
                start_time=data.get('start_time'),
                end_time=data.get('end_time'),
                duration=data.get('duration'),
                location=data.get('location', ''),
                attendees=data.get('attendees', []),
                event_type=data.get('event_type', 'meeting'),
                confidence=data.get('confidence', 0.5)
            )
        
    except Exception as e:
        logging.error(f"Error parsing natural language: {e}")
    
    # Fallback parsing
    return EventRequest(
        action='query',
        title=user_input,
        confidence=0.3
    )

def detect_conflicts(event_request: EventRequest, existing_events: List[Dict]) -> ConflictInfo:
    """
    Detect scheduling conflicts and suggest alternatives
    """
    try:
        if not event_request.start_time or not existing_events:
            return ConflictInfo(
                has_conflict=False,
                conflicting_events=[],
                reasoning="No conflicts detected"
            )
        
        # Check for time overlaps
        event_start = datetime.fromisoformat(event_request.start_time)
        event_end = datetime.fromisoformat(event_request.end_time) if event_request.end_time else event_start + timedelta(hours=1)
        
        conflicts = []
        for existing in existing_events:
            existing_start = datetime.fromisoformat(existing['start_time'])
            existing_end = datetime.fromisoformat(existing['end_time']) if existing['end_time'] else existing_start + timedelta(hours=1)
            
            # Check for overlap
            if (event_start < existing_end and event_end > existing_start):
                conflicts.append(existing)
        
        if conflicts:
            conflict_titles = [c['title'] for c in conflicts]
            reasoning = f"Conflicts with: {', '.join(conflict_titles)}"
            
            # Suggest alternative times
            suggested_times = []
            for offset in [1, 2, -1, -2]:  # Try 1-2 hours before/after
                alt_start = event_start + timedelta(hours=offset)
                alt_end = event_end + timedelta(hours=offset)
                
                # Check if alternative time is free
                has_alt_conflict = False
                for existing in existing_events:
                    existing_start = datetime.fromisoformat(existing['start_time'])
                    existing_end = datetime.fromisoformat(existing['end_time']) if existing['end_time'] else existing_start + timedelta(hours=1)
                    
                    if (alt_start < existing_end and alt_end > existing_start):
                        has_alt_conflict = True
                        break
                
                if not has_alt_conflict:
                    suggested_times.append(alt_start.strftime('%I:%M %p'))
            
            return ConflictInfo(
                has_conflict=True,
                conflicting_events=conflicts,
                reasoning=reasoning,
                suggested_times=suggested_times[:3]  # Limit to 3 suggestions
            )
        
        return ConflictInfo(
            has_conflict=False,
            conflicting_events=[],
            reasoning="No conflicts detected"
        )
        
    except Exception as e:
        logging.error(f"Error detecting conflicts: {e}")
        return ConflictInfo(
            has_conflict=False,
            conflicting_events=[],
            reasoning="Error checking conflicts"
        )

def handle_conversation(user_input: str, context: Dict) -> str:
    """
    Handle general conversation about calendar and scheduling
    """
    try:
        system_prompt = """
        You are KITEDESK Calendar, a helpful AI assistant for calendar management.
        
        You help users with:
        - Scheduling events and meetings
        - Managing their calendar
        - Answering questions about their schedule
        - Providing scheduling suggestions
        
        Keep responses:
        - Helpful and conversational
        - Concise but informative
        - Focused on calendar-related tasks
        - Professional but friendly
        
        Context: {context}
        
        If the user asks about scheduling, guide them to use specific commands like:
        - "Schedule a meeting with [person] at [time]"
        - "What do I have on [day]?"
        - "Remind me to [task] at [time]"
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_input)])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt.format(context=json.dumps(context)),
                temperature=0.7,
                max_output_tokens=500
            )
        )
        
        if response.text:
            return response.text
        
    except Exception as e:
        logging.error(f"Error handling conversation: {e}")
    
    return "I'm here to help with your calendar! Try asking me to schedule something or check your upcoming events."

def extract_voice_command(transcript: str) -> EventRequest:
    """
    Extract calendar commands from voice transcript
    """
    # For now, use the same parsing as text input
    return parse_natural_language(transcript)

def test_gemini_api() -> bool:
    """
    Test if Gemini API is working correctly
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello, are you working?"
        )
        
        if response.text:
            logging.info(f"Gemini API test successful: {response.text}")
            return True
        
    except Exception as e:
        logging.error(f"Gemini API test failed: {e}")
        return False
    
    return False