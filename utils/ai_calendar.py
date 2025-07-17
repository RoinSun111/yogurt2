import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class EventRequest(BaseModel):
    """Structured event data from AI parsing"""
    action: str  # create, update, delete, list, find
    title: str = ""
    description: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[int] = None  # in minutes
    location: str = ""
    event_type: str = "meeting"
    attendees: List[str] = []
    confidence: float = 0.0

class ConflictInfo(BaseModel):
    """Information about schedule conflicts"""
    has_conflict: bool
    conflicting_events: List[Dict]
    suggested_times: List[str]
    reasoning: str

class AICalendarAssistant:
    """AI-powered calendar assistant using Gemini for natural language processing"""
    
    def __init__(self):
        self.client = client
        self.conversation_history = []
        
        # System prompt for calendar management
        self.system_prompt = """You are KITEDESK Calendar, an AI assistant specialized in calendar management and scheduling.

Your capabilities:
- Parse natural language requests for calendar events
- Create, update, delete, and find calendar events
- Detect scheduling conflicts and suggest alternatives
- Provide intelligent scheduling recommendations
- Handle voice commands and conversational interactions

Guidelines:
- Always confirm important details before creating events
- Suggest optimal meeting times based on availability
- Be proactive about conflict detection
- Use friendly, professional language
- Ask clarifying questions when requests are ambiguous
- Remember context from previous interactions in the conversation

When parsing calendar requests, extract:
- Action (create/update/delete/list/find)
- Event title and description
- Date and time (convert relative terms like "tomorrow", "next week")
- Duration or end time
- Location (if mentioned)
- Event type (meeting/appointment/deadline/reminder)
- Attendees (if mentioned)

Respond with structured JSON when processing calendar operations, and conversational text for general interactions."""

    def parse_natural_language(self, user_input: str, current_time: datetime = None) -> EventRequest:
        """Parse natural language input into structured event data"""
        if current_time is None:
            current_time = datetime.now()
            
        try:
            prompt = f"""Parse this calendar request into structured data:
            
User Request: "{user_input}"
Current Date/Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %A')}

Extract the following information and respond with JSON:
{{
    "action": "create|update|delete|list|find",
    "title": "event title",
    "description": "event description or details",
    "start_time": "YYYY-MM-DD HH:MM" or null,
    "end_time": "YYYY-MM-DD HH:MM" or null,
    "date": "YYYY-MM-DD" or null,
    "duration": minutes as integer or null,
    "location": "location if mentioned",
    "event_type": "meeting|appointment|deadline|reminder",
    "attendees": ["list", "of", "attendees"],
    "confidence": 0.0-1.0 (how confident you are in the parsing)
}}

Rules:
- Convert relative dates ("tomorrow", "next week", "Monday") to actual dates
- Convert relative times ("at 2", "in the afternoon") to 24-hour format
- If no year specified, assume current year
- If no time specified for meetings, suggest appropriate business hours
- Default meeting duration is 60 minutes if not specified
- Set confidence based on how clear and complete the request is
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                event_data = json.loads(response.text)
                return EventRequest(**event_data)
            else:
                raise ValueError("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"Error parsing natural language: {e}")
            # Return a basic parse attempt
            return EventRequest(
                action="create" if any(word in user_input.lower() for word in ["schedule", "book", "create", "add"]) else "list",
                title=user_input[:50],
                confidence=0.1
            )

    def detect_conflicts(self, event_request: EventRequest, existing_events: List[Dict]) -> ConflictInfo:
        """Detect scheduling conflicts and suggest alternatives"""
        try:
            # Prepare conflict detection prompt
            prompt = f"""Analyze this scheduling request for conflicts:

New Event Request:
Title: {event_request.title}
Start Time: {event_request.start_time}
End Time: {event_request.end_time}
Duration: {event_request.duration} minutes

Existing Events:
{json.dumps(existing_events, indent=2)}

Analyze for conflicts and respond with JSON:
{{
    "has_conflict": true/false,
    "conflicting_events": [list of conflicting events],
    "suggested_times": ["alternative time slots in YYYY-MM-DD HH:MM format"],
    "reasoning": "explanation of conflicts and suggestions"
}}

Consider:
- Direct time overlaps
- Buffer time between meetings (suggest 15-min gaps)
- Reasonable meeting hours (9 AM - 6 PM for business)
- Travel time if locations are different
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                conflict_data = json.loads(response.text)
                return ConflictInfo(**conflict_data)
            else:
                return ConflictInfo(has_conflict=False, conflicting_events=[], suggested_times=[], reasoning="No conflicts detected")
                
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return ConflictInfo(has_conflict=False, conflicting_events=[], suggested_times=[], reasoning="Unable to analyze conflicts")

    def generate_smart_suggestions(self, user_context: str, existing_events: List[Dict]) -> List[str]:
        """Generate intelligent scheduling suggestions based on user patterns and context"""
        try:
            prompt = f"""Based on the user's calendar and context, generate 3-5 intelligent scheduling suggestions:

User Context: {user_context}

Recent Calendar Events:
{json.dumps(existing_events[-10:], indent=2)}  # Last 10 events for pattern analysis

Generate helpful suggestions like:
- Optimal meeting times based on their schedule patterns
- Reminders for upcoming deadlines
- Suggestions to block focus time
- Travel time considerations
- Meeting preparation time

Respond with a JSON array of suggestion strings:
["suggestion 1", "suggestion 2", "suggestion 3"]
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                suggestions = json.loads(response.text)
                return suggestions if isinstance(suggestions, list) else []
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

    def handle_conversation(self, user_input: str, context: Dict = None) -> str:
        """Handle conversational interactions with the calendar assistant"""
        if context is None:
            context = {}
            
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
        
        try:
            # Create conversation context
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])
            
            prompt = f"""You are KITEDESK Calendar Assistant. Respond to this user message conversationally and helpfully.

{self.system_prompt}

Conversation History:
{conversation_context}

Current Context:
{json.dumps(context, indent=2)}

User Message: "{user_input}"

Provide a helpful, conversational response. If the user is asking about calendar operations, explain what you can do and ask clarifying questions if needed.
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            assistant_response = response.text or "I'm here to help with your calendar. What would you like to do?"
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_response, "timestamp": datetime.now()})
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in conversation handling: {e}")
            return "I'm having trouble processing that request. Could you try rephrasing it?"

    def optimize_schedule(self, events: List[Dict], preferences: Dict = None) -> Dict:
        """Optimize daily/weekly schedule based on user preferences and best practices"""
        if preferences is None:
            preferences = {}
            
        try:
            prompt = f"""Analyze this schedule and provide optimization suggestions:

Events:
{json.dumps(events, indent=2)}

User Preferences:
{json.dumps(preferences, indent=2)}

Provide optimization suggestions as JSON:
{{
    "optimization_score": 0.0-1.0,
    "issues": ["list of issues found"],
    "suggestions": ["specific improvement suggestions"],
    "ideal_schedule": ["proposed schedule changes"],
    "focus_time_blocks": ["suggested focus time slots"],
    "break_recommendations": ["break and buffer time suggestions"]
}}

Consider:
- Meeting clustering vs. spread
- Focus time blocks
- Travel time between meetings
- Lunch and break times
- Energy levels throughout the day
- Meeting-free time for deep work
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                return json.loads(response.text)
            else:
                return {"optimization_score": 0.5, "issues": [], "suggestions": [], "ideal_schedule": [], "focus_time_blocks": [], "break_recommendations": []}
                
        except Exception as e:
            logger.error(f"Error optimizing schedule: {e}")
            return {"optimization_score": 0.5, "issues": ["Unable to analyze schedule"], "suggestions": [], "ideal_schedule": [], "focus_time_blocks": [], "break_recommendations": []}

    def extract_voice_command(self, audio_transcript: str) -> EventRequest:
        """Extract calendar commands from voice input transcript"""
        # Add voice-specific processing
        prompt = f"""Extract calendar command from this voice transcript:

Voice Input: "{audio_transcript}"

This is from a voice command, so:
- Account for speech recognition errors
- Handle casual speech patterns
- Interpret incomplete sentences
- Handle commands like "schedule", "book", "cancel", "what's next"

{self.parse_natural_language.__doc__}"""
        
        return self.parse_natural_language(audio_transcript)

# Global instance
ai_calendar = AICalendarAssistant()