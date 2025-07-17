// AI Calendar JavaScript - Handle calendar UI and AI interactions

class AICalendar {
    constructor() {
        this.currentDate = new Date();
        this.events = [];
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.renderCalendar();
        this.loadEvents();
        this.setupVoiceRecording();
        
        // Initialize feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }
    
    setupEventListeners() {
        // Calendar navigation
        document.getElementById('prev-month').addEventListener('click', () => {
            this.currentDate.setMonth(this.currentDate.getMonth() - 1);
            this.renderCalendar();
        });
        
        document.getElementById('next-month').addEventListener('click', () => {
            this.currentDate.setMonth(this.currentDate.getMonth() + 1);
            this.renderCalendar();
        });
        
        document.getElementById('today-btn').addEventListener('click', () => {
            this.currentDate = new Date();
            this.renderCalendar();
        });
        
        // Chat interface
        document.getElementById('send-chat').addEventListener('click', () => {
            this.sendChatMessage();
        });
        
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });
        
        // Voice recording
        document.getElementById('voice-btn').addEventListener('mousedown', () => {
            this.startVoiceRecording();
        });
        
        document.getElementById('voice-btn').addEventListener('mouseup', () => {
            this.stopVoiceRecording();
        });
        
        document.getElementById('voice-btn').addEventListener('mouseleave', () => {
            this.stopVoiceRecording();
        });
        
        // Quick actions
        document.getElementById('schedule-meeting').addEventListener('click', () => {
            this.addChatMessage('Schedule a 1-hour meeting with the team tomorrow at 2pm', 'user');
            this.processChatMessage('Schedule a 1-hour meeting with the team tomorrow at 2pm');
        });
        
        document.getElementById('add-reminder').addEventListener('click', () => {
            this.addChatMessage('Set a reminder to review monthly reports on Friday at 3pm', 'user');
            this.processChatMessage('Set a reminder to review monthly reports on Friday at 3pm');
        });
        
        document.getElementById('view-today').addEventListener('click', () => {
            this.addChatMessage('What\'s on my schedule today?', 'user');
            this.processChatMessage('What\'s on my schedule today?');
        });
        
        document.getElementById('find-time').addEventListener('click', () => {
            this.addChatMessage('Find me free time for a 2-hour work session this week', 'user');
            this.processChatMessage('Find me free time for a 2-hour work session this week');
        });
    }
    
    renderCalendar() {
        const monthYear = document.getElementById('calendar-month-year');
        const calendarGrid = document.getElementById('calendar-grid');
        
        // Update month/year display
        const options = { year: 'numeric', month: 'long' };
        monthYear.textContent = this.currentDate.toLocaleDateString('en-US', options);
        
        // Clear previous calendar
        calendarGrid.innerHTML = '';
        
        // Get first day of month and number of days
        const firstDay = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), 1);
        const lastDay = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth() + 1, 0);
        const startDate = new Date(firstDay);
        startDate.setDate(startDate.getDate() - firstDay.getDay());
        
        // Generate calendar days (6 weeks = 42 days)
        for (let i = 0; i < 42; i++) {
            const date = new Date(startDate);
            date.setDate(startDate.getDate() + i);
            
            const dayDiv = document.createElement('div');
            dayDiv.className = 'calendar-day';
            
            // Add classes for styling
            if (date.getMonth() !== this.currentDate.getMonth()) {
                dayDiv.classList.add('other-month');
            }
            
            if (this.isToday(date)) {
                dayDiv.classList.add('today');
            }
            
            // Day number
            const dayNumber = document.createElement('div');
            dayNumber.className = 'calendar-day-number';
            dayNumber.textContent = date.getDate();
            dayDiv.appendChild(dayNumber);
            
            // Add events for this day
            const dayEvents = this.getEventsForDay(date);
            dayEvents.forEach(event => {
                const eventDiv = document.createElement('div');
                eventDiv.className = `calendar-event ${event.type}`;
                if (event.is_ai_created) {
                    eventDiv.classList.add('ai-created');
                }
                eventDiv.textContent = event.title;
                eventDiv.onclick = () => this.showEventDetails(event);
                dayDiv.appendChild(eventDiv);
            });
            
            // Click handler for adding events
            dayDiv.addEventListener('click', (e) => {
                if (e.target === dayDiv || e.target === dayNumber) {
                    this.onDayClick(date);
                }
            });
            
            calendarGrid.appendChild(dayDiv);
        }
    }
    
    isToday(date) {
        const today = new Date();
        return date.toDateString() === today.toDateString();
    }
    
    getEventsForDay(date) {
        const dayStart = new Date(date);
        dayStart.setHours(0, 0, 0, 0);
        const dayEnd = new Date(date);
        dayEnd.setHours(23, 59, 59, 999);
        
        return this.events.filter(event => {
            const eventStart = new Date(event.start);
            return eventStart >= dayStart && eventStart <= dayEnd;
        });
    }
    
    onDayClick(date) {
        const dateStr = date.toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        const message = `What do I have scheduled for ${dateStr}?`;
        this.addChatMessage(message, 'user');
        this.processChatMessage(message);
    }
    
    showEventDetails(event) {
        const modalBody = document.querySelector('#eventModal .modal-body #event-details');
        const startTime = new Date(event.start);
        const endTime = event.end ? new Date(event.end) : null;
        
        modalBody.innerHTML = `
            <h6>${event.title}</h6>
            <p><strong>Time:</strong> ${startTime.toLocaleString()}</p>
            ${endTime ? `<p><strong>End:</strong> ${endTime.toLocaleString()}</p>` : ''}
            ${event.description ? `<p><strong>Description:</strong> ${event.description}</p>` : ''}
            ${event.location ? `<p><strong>Location:</strong> ${event.location}</p>` : ''}
            <p><strong>Type:</strong> ${event.type}</p>
            ${event.is_ai_created ? '<span class="badge bg-secondary">AI Created</span>' : ''}
        `;
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('eventModal'));
        modal.show();
    }
    
    async loadEvents() {
        try {
            const startDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), 1);
            const endDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth() + 1, 0);
            
            const response = await fetch(`/api/calendar/events?start=${startDate.toISOString().split('T')[0]}&end=${endDate.toISOString().split('T')[0]}`);
            const data = await response.json();
            
            if (data.success) {
                this.events = data.events;
                this.renderCalendar();
            }
        } catch (error) {
            console.error('Error loading events:', error);
        }
    }
    
    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addChatMessage(message, 'user');
        input.value = '';
        
        // Process message
        await this.processChatMessage(message);
    }
    
    async processChatMessage(message) {
        try {
            // Show typing indicator
            this.addTypingIndicator();
            
            const response = await fetch('/api/calendar/ai-chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            if (data.success) {
                this.addChatMessage(data.response, 'ai');
                
                // If an event was created, refresh the calendar
                if (data.event_created) {
                    await this.loadEvents();
                }
            } else {
                this.addChatMessage('Sorry, I had trouble processing that request. Could you try again?', 'ai');
            }
        } catch (error) {
            console.error('Error processing chat message:', error);
            this.removeTypingIndicator();
            this.addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }
    }
    
    addChatMessage(message, sender) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
        
        messageDiv.innerHTML = `
            <div class="message-content">
                ${sender === 'ai' ? '<strong>KITEDESK Calendar:</strong> ' : ''}${message}
            </div>
            <div class="message-time">${timeStr}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    addTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <strong>KITEDESK Calendar:</strong> <span class="typing-dots">...</span>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    async setupVoiceRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processVoiceRecording();
            };
        } catch (error) {
            console.warn('Microphone access denied or not available:', error);
        }
    }
    
    startVoiceRecording() {
        if (!this.mediaRecorder || this.isRecording) return;
        
        this.isRecording = true;
        this.audioChunks = [];
        
        const voiceBtn = document.getElementById('voice-btn');
        const voiceBtnText = document.getElementById('voice-btn-text');
        
        voiceBtn.classList.add('voice-recording');
        voiceBtnText.textContent = 'Recording...';
        
        this.mediaRecorder.start();
    }
    
    stopVoiceRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        const voiceBtn = document.getElementById('voice-btn');
        const voiceBtnText = document.getElementById('voice-btn-text');
        
        voiceBtn.classList.remove('voice-recording');
        voiceBtnText.textContent = 'Hold to Speak';
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }
    
    async processVoiceRecording() {
        if (this.audioChunks.length === 0) return;
        
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'voice_command.wav');
            
            // Show processing message
            this.addChatMessage('🎤 Processing voice command...', 'ai');
            
            const response = await fetch('/api/calendar/voice', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Add the transcript as user message
                this.addChatMessage(`Voice: "${data.transcript}"`, 'user');
                
                // Process the command
                await this.processChatMessage(data.transcript);
            } else {
                this.addChatMessage('Sorry, I couldn\'t understand that. Please try speaking clearly.', 'ai');
            }
        } catch (error) {
            console.error('Error processing voice recording:', error);
            this.addChatMessage('Error processing voice command. Please try again.', 'ai');
        }
    }
}

// Initialize the AI Calendar when the page loads
document.addEventListener('DOMContentLoaded', function() {
    new AICalendar();
});