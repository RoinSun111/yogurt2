// Activity Monitor - Track and display user activity states

document.addEventListener('DOMContentLoaded', function() {
    // Set up initial state
    initializeActivityMonitor();
    
    // Update the activity status every 5 seconds
    setInterval(updateActivityStatus, 5000);
    
    // Initial update
    updateActivityStatus();
});

// Initialize the activity monitor UI
function initializeActivityMonitor() {
    // Set up state icons
    const stateIcons = {
        working: 'edit-3',
        not_working: 'pause',
        distracted_by_others: 'users',
        on_break: 'coffee',
        idle: 'moon',
        unknown: 'help-circle'
    };
    
    // Set up substate icons
    const substateIcons = {
        typing: 'type',
        writing: 'pen-tool',
        reading: 'book-open'
    };
    
    // Preload icons
    for (const state in stateIcons) {
        const iconElement = document.createElement('span');
        iconElement.setAttribute('data-feather', stateIcons[state]);
        iconElement.style.display = 'none';
        document.body.appendChild(iconElement);
    }
    
    for (const substate in substateIcons) {
        const iconElement = document.createElement('span');
        iconElement.setAttribute('data-feather', substateIcons[substate]);
        iconElement.style.display = 'none';
        document.body.appendChild(iconElement);
    }
    
    // Initialize feather icons (done in layout.html)
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

// Update the activity status from the API
function updateActivityStatus() {
    fetch('/api/activity_status')
        .then(response => response.json())
        .then(data => {
            displayActivityStatus(data);
        })
        .catch(error => {
            console.error('Error fetching activity status:', error);
        });
}

// Display the activity status in the UI
function displayActivityStatus(data) {
    // State badges and colors
    const stateBadges = {
        working: { text: 'Working', class: 'bg-success' },
        not_working: { text: 'Not Working', class: 'bg-warning' },
        distracted_by_others: { text: 'Distracted', class: 'bg-warning' },
        on_break: { text: 'On Break', class: 'bg-info' },
        idle: { text: 'Idle', class: 'bg-secondary' },
        unknown: { text: 'Unknown', class: 'bg-secondary' }
    };
    
    // State icons
    const stateIcons = {
        working: 'edit-3',
        not_working: 'pause',
        distracted_by_others: 'users',
        on_break: 'coffee',
        idle: 'moon',
        unknown: 'help-circle'
    };
    
    // Substate labels and icons
    const substateLabels = {
        typing: 'Typing',
        writing: 'Writing',
        reading: 'Reading'
    };
    
    const substateIcons = {
        typing: 'type',
        writing: 'pen-tool',
        reading: 'book-open'
    };
    
    // Get the status badge and icon elements
    const activityStateBadge = document.getElementById('activity-state');
    const activityIconContainer = document.getElementById('activity-icon');
    const activityDescription = document.getElementById('activity-description');
    const activityTimer = document.getElementById('activity-timer');
    const activityProgressBar = document.getElementById('activity-progress');
    const activityMetrics = document.getElementById('activity-metrics');
    
    // Safety check - if elements don't exist yet, don't proceed
    if (!activityStateBadge || !activityIconContainer) {
        console.error('Activity monitor elements not found in DOM');
        return;
    }
    
    // Get the activity state and substate
    const state = data.activity_state || 'unknown';
    const substate = data.working_substate;
    
    // Update the state badge
    const badge = stateBadges[state] || stateBadges.unknown;
    activityStateBadge.textContent = badge.text;
    activityStateBadge.className = `badge ${badge.class}`;
    
    // Update the icon
    activityIconContainer.innerHTML = '';
    const icon = document.createElement('i');
    icon.setAttribute('data-feather', stateIcons[state]);
    activityIconContainer.appendChild(icon);
    
    // Update the description
    let description = '';
    
    if (state === 'working' && substate) {
        description = `${substateLabels[substate] || substate}`;
        
        // Add substate icon next to description
        if (substateIcons[substate]) {
            const substateIcon = document.createElement('i');
            substateIcon.setAttribute('data-feather', substateIcons[substate]);
            substateIcon.classList.add('feather-icon', 'ms-2');
            description += ` <span id="substate-icon"></span>`;
        }
    } else {
        // Descriptions for non-working states
        const descriptions = {
            not_working: 'Taking a short break but still at desk',
            distracted_by_others: 'Interacting with someone else',
            on_break: 'Away from the desk',
            idle: 'Present but inactive',
            unknown: 'Status not determined yet'
        };
        description = descriptions[state] || 'Monitoring activity...';
    }
    activityDescription.innerHTML = description;
    
    // Add substate icon if applicable
    if (state === 'working' && substate && substateIcons[substate]) {
        const substateIconEl = document.getElementById('substate-icon');
        if (substateIconEl) {
            substateIconEl.innerHTML = '';
            const substateIcon = document.createElement('i');
            substateIcon.setAttribute('data-feather', substateIcons[substate]);
            substateIcon.classList.add('feather-icon-sm');
            substateIconEl.appendChild(substateIcon);
        }
    }
    
    // Format time in state
    let timeInState = data.time_in_state || 0;
    const minutes = Math.floor(timeInState / 60);
    const seconds = timeInState % 60;
    const timeFormatted = `${minutes}m ${seconds}s`;
    activityTimer.textContent = timeFormatted;
    
    // Update progress bar based on state
    // Different states have different ideal durations
    let maxDuration = 0;
    if (state === 'working') {
        maxDuration = 25 * 60; // 25 minutes (pomodoro-like)
    } else if (state === 'not_working') {
        maxDuration = 5 * 60; // 5 minutes
    } else if (state === 'on_break') {
        maxDuration = 15 * 60; // 15 minutes
    } else if (state === 'idle') {
        maxDuration = 2 * 60; // 2 minutes
    } else {
        maxDuration = 10 * 60; // 10 minutes default
    }
    
    // Calculate percentage
    const percentage = Math.min(100, (timeInState / maxDuration) * 100);
    activityProgressBar.style.width = `${percentage}%`;
    activityProgressBar.setAttribute('aria-valuenow', percentage);
    
    // Set progress bar color based on state
    activityProgressBar.className = 'progress-bar';
    if (badge.class) {
        activityProgressBar.classList.add(badge.class);
    }
    
    // Update additional metrics
    let metricsHTML = '';
    
    // Add head angle information
    if (data.head_angle !== undefined) {
        metricsHTML += `<div class="metric">
            <span class="metric-label">Head Angle:</span>
            <span class="metric-value">${Math.round(data.head_angle)}°</span>
        </div>`;
    }
    
    // Add movement level information
    if (data.movement_level !== undefined) {
        metricsHTML += `<div class="metric">
            <span class="metric-label">Movement:</span>
            <span class="metric-value">${Math.round(data.movement_level)}%</span>
        </div>`;
    }
    
    activityMetrics.innerHTML = metricsHTML;
    
    // Re-initialize feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

// Helper function to get activity context
function getActivityContext(state, substate) {
    const contexts = {
        working: {
            base: "You're being productive.",
            typing: "You're typing. Keep up the good work!",
            writing: "You're writing. Maintain your focus.",
            reading: "You're reading. Take in the information."
        },
        not_working: "Taking a short break. Remember to stay on task.",
        distracted_by_others: "You're interacting with someone. Try to return to your work soon.",
        on_break: "You're on a break. Remember to return refreshed.",
        idle: "You haven't moved in a while. Time to re-engage with your work.",
        unknown: "Monitoring your activity..."
    };
    
    if (state === 'working' && substate) {
        return contexts.working[substate] || contexts.working.base;
    }
    
    return contexts[state] || contexts.unknown;
}